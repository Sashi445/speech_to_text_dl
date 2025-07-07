### ✅ caption_api.py (Blueprint)
from flask import Blueprint, request, jsonify, send_file, render_template, redirect, url_for
import uuid
import os
import threading
from database import SessionLocal
from db.models import CaptionJob
from services.caption_generator.index import CaptionGeneratorService

caption_api = Blueprint("caption_api", __name__, template_folder="../templates")


def caption_worker(job_id, input_path, output_path):
    db = SessionLocal()
    job = db.query(CaptionJob).get(job_id)
    job.status = "processing"
    db.commit()

    try:
        service = CaptionGeneratorService()
        service.process_video(input_path, output_path)
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)
    finally:
        db.commit()
        db.close()


@caption_api.route("/", methods=["GET", "POST"])
def dashboard():
    if request.method == "POST":
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        try:
            job_id = str(uuid.uuid4())
            input_path = f"uploads/{job_id}.mp4"
            output_path = f"outputs/{job_id}_captioned.mp4"
            os.makedirs("uploads", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)

            request.files['video'].save(input_path)

            db = SessionLocal()
            job = CaptionJob(
                id=job_id,
                status="pending",
                video_path=input_path,
                output_path=output_path
            )
            db.add(job)
            db.commit()
            db.close()

            threading.Thread(target=caption_worker, args=(job_id, input_path, output_path), daemon=True).start()

            return redirect(url_for("caption_api.dashboard"))

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    db = SessionLocal()
    jobs = db.query(CaptionJob).order_by(CaptionJob.id.desc()).all()
    db.close()
    return render_template("caption_generation.html", jobs=jobs)


@caption_api.route("/status/<job_id>")
def get_status(job_id):
    db = SessionLocal()
    job = db.query(CaptionJob).get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job.status,
        "error": job.error
    })


@caption_api.route("/download/<job_id>")
def download(job_id):
    db = SessionLocal()
    job = db.query(CaptionJob).get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.status != "done":
        return jsonify({"error": "Job not completed yet"}), 400
    return send_file(job.output_path, as_attachment=True)


@caption_api.route("/jobs", methods=["GET", "POST"])
def handle_jobs():
    db = SessionLocal()
    try:
        if request.method == "GET":
            jobs = db.query(CaptionJob).all()
            result = [
                {
                    "id": job.id,
                    "status": job.status,
                    "video_path": job.video_path,
                    "output_path": job.output_path,
                    "error": job.error
                } for job in jobs
            ]
            return jsonify(result)

        elif request.method == "POST":
            data = request.get_json()
            job_id = data.get("id")
            status = data.get("status")
            job = db.query(CaptionJob).get(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            job.status = status
            db.commit()
            return jsonify({"message": "Job updated successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()
