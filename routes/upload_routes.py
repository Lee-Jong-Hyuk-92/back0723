import os
import json
import time
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image
from flask_jwt_extended import jwt_required, get_jwt_identity
from ai_model.predictor import predict_overlayed_image              # model1: 질병
from ai_model import hygiene_predictor, tooth_number_predictor      # model2: 위생, model3: 치아번호
from models.model import MongoDBClient

# ✅ 업로드 전용 로거 분리 설정
upload_logger = logging.getLogger("upload_logger")
upload_logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "inference_times.log")

if not upload_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    upload_logger.addHandler(fh)

upload_bp = Blueprint('upload', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@upload_bp.route('/upload_image', methods=['POST'])
@jwt_required()
def upload_image_from_flutter():
    return upload_masked_image()

@upload_bp.route('/upload', methods=['POST'])
@jwt_required()  # ✅ 이 줄 추가
def upload_plain_image():
    return upload_masked_image()

@upload_bp.route('/upload_masked_image', methods=['POST'])
def upload_masked_image():
    user_id = get_jwt_identity()  # ✅ JWT에서 user_id 추출
    start_total = time.perf_counter()

    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 필요합니다.'}), 400

    file = request.files['file']

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
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        original_filename = secure_filename(file.filename)
        base_name = f"{user_id}_{timestamp}_{original_filename}"
        base_name = os.path.splitext(base_name)[0] + ".png"  # ✅ 확장자 강제 변경

        upload_dir = current_app.config['UPLOAD_FOLDER_ORIGINAL']
        processed_dir_1 = current_app.config['PROCESSED_FOLDER_MODEL1']
        processed_dir_2 = current_app.config['PROCESSED_FOLDER_MODEL2']
        processed_dir_3 = current_app.config['PROCESSED_FOLDER_MODEL3']

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(processed_dir_1, exist_ok=True)
        os.makedirs(processed_dir_2, exist_ok=True)
        os.makedirs(processed_dir_3, exist_ok=True)

        original_path = os.path.join(upload_dir, base_name)
        file.save(original_path)

        image = Image.open(original_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ✅ model1: 질병
        t1 = time.perf_counter()
        processed_path_1 = os.path.join(processed_dir_1, base_name)
        masked_image_1, lesion_points, backend_model_confidence, backend_model_name, disease_label = predict_overlayed_image(image)
        masked_image_1.save(processed_path_1, format='PNG')  # ✅ JPEG 저장 오류 방지
        upload_logger.info(f"[🧠 모델1] 질병 세그멘테이션 추론 시간: {time.perf_counter() - t1:.4f}s")

        # ✅ model2: 위생
        t2 = time.perf_counter()
        processed_path_2 = os.path.join(processed_dir_2, base_name)
        hygiene_predictor.predict_mask_and_overlay_only(image, processed_path_2)
        hygiene_class_id, hygiene_conf, hygiene_label = hygiene_predictor.get_main_class_and_confidence_and_label(image)
        upload_logger.info(f"[🧠 모델2] 위생 세그멘테이션 추론 시간: {time.perf_counter() - t2:.4f}s")

        # ✅ model3: 치아번호
        t3 = time.perf_counter()
        processed_path_3 = os.path.join(processed_dir_3, base_name)
        tooth_number_predictor.predict_mask_and_overlay_only(image, processed_path_3)
        tooth_info = tooth_number_predictor.get_main_class_info_json(image)
        upload_logger.info(f"[🧠 모델3] 치아번호 세그멘테이션 추론 시간: {time.perf_counter() - t3:.4f}s")

        total_elapsed = time.perf_counter() - start_total
        upload_logger.info(f"[📸 전체 모델 추론 완료] 총 소요 시간: {total_elapsed:.4f}s (user_id={user_id})")

        mongo_client = MongoDBClient()
        inserted_id = mongo_client.insert_result({
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

        return jsonify({
            'message': '3개 모델 처리 및 저장 완료',
            'inference_result_id': str(inserted_id),  # ✅ 추가됨
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
