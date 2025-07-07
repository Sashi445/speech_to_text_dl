from flask import Blueprint, request, jsonify, render_template
from services.speech_to_text.index import SpeechToTextService
from werkzeug.utils import secure_filename
import os
import tempfile
import torchaudio
import subprocess
import os


speech_to_text_bp = Blueprint('speech_to_text', __name__)
speech_to_text_service = SpeechToTextService()

@speech_to_text_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get model_type from form
        model_type = request.form.get('model_type')
        if not model_type:
            return render_template('speech_to_text.html', error='Please select a model type')
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return render_template('speech_to_text.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('speech_to_text.html', error='No file selected')
        
        # Check audio file format
        allowed_extensions = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac', 'aif', 'aiff'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return render_template('speech_to_text.html', 
                                error=f'Invalid file format. Allowed formats: {", ".join(allowed_extensions)}')
        
        # Save file with secure filename
        filename = secure_filename(file.filename)
        upload_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(upload_path)
        
        # Transcribe with model_type
        transcripts = speech_to_text_service.transcribe(upload_path, model_type)
        
        return render_template('speech_to_text.html', transcripts=transcripts, filename=filename)
    
    return render_template('speech_to_text.html')



@speech_to_text_bp.route("/transcribe_chunk", methods=["POST"])
def transcribe_chunk():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        audio_file = request.files['audio']

        # Save uploaded .webm audio chunk
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_webm:
            audio_file.save(tmp_webm.name)
            webm_path = tmp_webm.name

        # Convert to .wav using ffmpeg
        wav_path = webm_path.replace(".webm", ".wav")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            error_message = result.stderr.decode()
            print("FFmpeg failed:", error_message)
            os.remove(webm_path)
            return jsonify({"error": "FFmpeg failed to convert audio", "details": error_message}), 500

        # Load the converted WAV audio
        waveform, sample_rate = torchaudio.load(wav_path)

        # Transcribe using your RNN + CTC model
        transcript = speech_to_text_service.transcribe_chunk(waveform, sample_rate)

        # Cleanup temp files
        os.remove(webm_path)
        os.remove(wav_path)

        return jsonify({"transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500