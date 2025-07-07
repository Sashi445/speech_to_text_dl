# /services/text_translation/index.py

from utils.text_translation.preprocess import TranslationPreprocessor
import os

_base_dir = os.path.join(os.path.dirname(__file__), "../../utils/text_translation")
_preprocessor = None

def get_translator():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TranslationPreprocessor(os.path.abspath(_base_dir))
    return _preprocessor
