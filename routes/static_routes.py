import os
from flask import Blueprint, send_from_directory, current_app

static_bp = Blueprint('static', __name__)

# 원본 이미지 제공
@static_bp.route('/images/original/<filename>')
def serve_original_image(filename):
    return send_from_directory(
        current_app.config['UPLOAD_FOLDER_ORIGINAL'],
        filename
    )

# 모델 1 마스크 이미지 제공
@static_bp.route('/images/model1/<filename>')
def serve_model1_image(filename):
    return send_from_directory(
        current_app.config['PROCESSED_FOLDER_MODEL1'],
        filename
    )

# 모델 2 마스크 이미지 제공
@static_bp.route('/images/model2/<filename>')
def serve_model2_image(filename):
    return send_from_directory(
        current_app.config['PROCESSED_FOLDER_MODEL2'],
        filename
    )

# 모델 3 마스크 이미지 제공
@static_bp.route('/images/model3/<filename>')
def serve_model3_image(filename):
    return send_from_directory(
        current_app.config['PROCESSED_FOLDER_MODEL3'],
        filename
    )

# X-ray 모델 1 마스크 이미지 제공
@static_bp.route('/images/xmodel1/<filename>')
def serve_xmodel1_image(filename):
    return send_from_directory(
        current_app.config['PROCESSED_FOLDER_XMODEL1'],
        filename
    )

# X-ray 모델 2 마스크 이미지 제공
@static_bp.route('/images/xmodel2/<filename>')
def serve_xmodel2_image(filename):
    return send_from_directory(
        current_app.config['PROCESSED_FOLDER_XMODEL2'],
        filename
    )

#이미지 타입            접근 URL 예시
#원본                   /images/original/파일명.png
#모델1 마스크	         /images/model1/파일명.png
#모델2 마스크	         /images/model2/파일명.png
#모델3 마스크	         /images/model3/파일명.png
#xray모델1 마스크       /images/xmodel1/파일명.png
#xray모델2 마스크       /images/xmodel2/파일명.png