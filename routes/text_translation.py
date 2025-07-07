# from flask import Blueprint, render_template, request, jsonify

# text_translation_bp = Blueprint("text_translation", __name__)

# @text_translation_bp.route("/", methods=["GET"])
# def index():
#     return render_template("text_translation.html")

# /routes/text_translation.py

from flask import Blueprint, render_template, request, jsonify
from services.text_translation.index import get_translator

text_translation_bp = Blueprint("text_translation", __name__)

@text_translation_bp.route("/", methods=["GET", "POST"])
def index():
    translated_text = None
    error = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        if not input_text:
            error = "Please enter some text to translate."
        else:
            try:
                translated_text = get_translator().translate(input_text)
            except Exception as e:
                error = "Translation failed."
    return render_template(
        "text_translation.html",
        input_text=input_text,
        translated_text=translated_text,
        error=error
    )


@text_translation_bp.route("/api", methods=["POST"])
def api_translate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Send 'text'"}), 400
    input_text = data["text"]
    try:
        translated_text = get_translator().translate(input_text)
        return jsonify({"original_text": input_text, "translated_text": translated_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Translation failed"}), 500
