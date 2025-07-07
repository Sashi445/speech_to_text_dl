from flask import Flask
import os

from database import engine, Base
from db.models import CaptionJob  # import all models you want to create

def create_app():
    app = Flask(__name__)

    # Register Blueprints
    from routes.main import main_bp
    from routes.speech_to_text import speech_to_text_bp
    from routes.text_translation import text_translation_bp
    from routes.caption_generator import caption_api

    app.register_blueprint(main_bp)
    app.register_blueprint(speech_to_text_bp, url_prefix='/speech-to-text')
    app.register_blueprint(text_translation_bp, url_prefix='/text-translation')
    app.register_blueprint(caption_api, url_prefix='/caption-generator')

    from database import engine, Base
    from db.models import CaptionJob

    def init_db():
        Base.metadata.create_all(bind=engine)

    app.before_request(init_db)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
