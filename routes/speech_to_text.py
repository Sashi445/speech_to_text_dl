from flask import Blueprint, request, jsonify, render_template
from services.speech_to_text.index import SpeechToTextService
from werkzeug.utils import secure_filename
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