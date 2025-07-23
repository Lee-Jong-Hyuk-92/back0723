import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image

from ai_model.predictor import predict_overlayed_image              # model1: 질병
from ai_model import hygiene_predictor, tooth_number_predictor      # model2: 위생, model3: 치아번호
from models.model import MongoDBClient

upload_bp = Blueprint('upload', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@upload_bp.route('/upload_image', methods=['POST'])
def upload_image_from_flutter():
    return upload_masked_image()

@upload_bp.route('/upload', methods=['POST'])
def upload_plain_image():
    return upload_masked_image()

@upload_bp.route('/upload_masked_image', methods=['POST'])
def upload_masked_image():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 필요합니다.'}), 400

    file = request.files['file']
    user_id = request.form.get('user_id', 'anonymous')

    yolo_results_json_str = request.form.get('yolo_results_json')
    yolo_inference_data = []
    if yolo_results_json_str:
        try:
            yolo_inference_data = json.loads(yolo_results_json_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'YOLO 결과 JSON 형식 오류: {e}'}), 400

    if file.filename == '':
        return jsonify({'error': '파일명이 비어 있습니다.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        # 파일 저장 준비
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        original_filename = secure_filename(file.filename)
        base_name = f"{user_id}_{timestamp}_{original_filename}"

        upload_dir = current_app.config['UPLOAD_FOLDER_ORIGINAL']
        processed_dir_1 = current_app.config['PROCESSED_FOLDER_MODEL1']
        processed_dir_2 = current_app.config['PROCESSED_FOLDER_MODEL2']
        processed_dir_3 = current_app.config['PROCESSED_FOLDER_MODEL3']

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(processed_dir_1, exist_ok=True)
        os.makedirs(processed_dir_2, exist_ok=True)
        os.makedirs(processed_dir_3, exist_ok=True)

        # 원본 이미지 저장
        original_path = os.path.join(upload_dir, base_name)
        file.save(original_path)

        image = Image.open(original_path).convert("RGB")

        # ✅ model1: 질병 세그멘테이션
        processed_path_1 = os.path.join(processed_dir_1, base_name)
        masked_image_1, lesion_points, backend_model_confidence, backend_model_name, disease_label = predict_overlayed_image(image)
        masked_image_1.save(processed_path_1)

        # ✅ model2: 위생 세그멘테이션
        processed_path_2 = os.path.join(processed_dir_2, base_name)
        hygiene_predictor.predict_mask_and_overlay_only(image, processed_path_2)
        hygiene_class_id, hygiene_conf, hygiene_label = hygiene_predictor.get_main_class_and_confidence_and_label(image)

        # ✅ model3: 치아번호 세그멘테이션
        processed_path_3 = os.path.join(processed_dir_3, base_name)
        tooth_number_predictor.predict_mask_and_overlay_only(image, processed_path_3)
        tooth_info = tooth_number_predictor.get_main_class_info_json(image)

        # ✅ MongoDB 저장
        mongo_client = MongoDBClient()
        mongo_client.insert_result({
            'user_id': user_id,
            'original_image_path': f"/images/original/{base_name}",
            'original_image_yolo_detections': yolo_inference_data,

            'model1_image_path': f"/images/model1/{base_name}",
            'model1_inference_result': {
                'message': 'model1 마스크 생성 완료',
                'lesion_points': lesion_points,
                'confidence': backend_model_confidence,
                'used_model': backend_model_name,
                'label': disease_label
            },

            'model2_image_path': f"/images/model2/{base_name}",
            'model2_inference_result': {
                'message': 'model2 마스크 생성 완료',
                'class_id': hygiene_class_id,
                'confidence': hygiene_conf,
                'label': hygiene_label
            },

            'model3_image_path': f"/images/model3/{base_name}",
            'model3_inference_result': {
                'message': 'model3 마스크 생성 완료',
                'class_id': tooth_info['class_id'],
                'confidence': tooth_info['confidence'],
                'tooth_number_fdi': tooth_info['tooth_number_fdi']
            },

            'timestamp': datetime.now()
        })

        # ✅ 응답 반환
        return jsonify({
            'message': '3개 모델 처리 및 저장 완료',
            'original_image_path': f"/images/original/{base_name}",
            'original_image_yolo_detections': yolo_inference_data,

            'model1_image_path': f"/images/model1/{base_name}",
            'model1_inference_result': {
                'message': 'model1 마스크 생성 완료',
                'lesion_points': lesion_points,
                'confidence': backend_model_confidence,
                'used_model': backend_model_name,
                'label': disease_label
            },

            'model2_image_path': f"/images/model2/{base_name}",
            'model2_inference_result': {
                'message': 'model2 마스크 생성 완료',
                'class_id': hygiene_class_id,
                'confidence': hygiene_conf,
                'label': hygiene_label
            },

            'model3_image_path': f"/images/model3/{base_name}",
            'model3_inference_result': {
                'message': 'model3 마스크 생성 완료',
                'class_id': tooth_info['class_id'],
                'confidence': tooth_info['confidence'],
                'tooth_number_fdi': tooth_info['tooth_number_fdi']
            },

            'timestamp': datetime.now()
        }), 200

    except Exception as e:
        return jsonify({'error': f'서버 처리 중 오류: {str(e)}'}), 500
