from flask import Flask
import os

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configuration
    # app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Register blueprints
    from routes.main import main_bp
    from routes.speech_to_text import speech_to_text_bp
    from routes.text_translation import text_translation_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(speech_to_text_bp, url_prefix='/speech-to-text')
    app.register_blueprint(text_translation_bp, url_prefix='/text-translation')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
