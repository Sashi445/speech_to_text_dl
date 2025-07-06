from flask import Blueprint, render_template, request, jsonify

text_translation_bp = Blueprint("text_translation", __name__)

@text_translation_bp.route("/", methods=["GET"])
def index():
    return render_template("text_translation.html")