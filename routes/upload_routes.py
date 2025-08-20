import os
import json
import time
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from flask_jwt_extended import jwt_required, get_jwt_identity
from ai_model.predictor import predict_overlayed_image
from ai_model import hygiene_predictor, tooth_number_predictor
from ai_model.combiner import combine_results
from models.model import MongoDBClient

import numpy as np

try:
    import torch
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
except Exception:
    def _sync_cuda():
        pass

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

def _load_font(size: int = 18):
    """플랫폼별 한글 폰트 폴백."""
    try:
        if os.name == "nt" and os.path.exists("C:/Windows/Fonts/malgun.ttf"):
            return ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
        if os.path.exists("/mnt/c/Windows/Fonts/malgun.ttf"):
            return ImageFont.truetype("/mnt/c/Windows/Fonts/malgun.ttf", size)
        if os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
            return ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size)
    except Exception:
        pass
    return ImageFont.load_default()

def _convert_for_mongo(data):
    """MongoDB에 저장할 수 있도록 NumPy 객체를 변환합니다."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, dict):
        return {k: _convert_for_mongo(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_for_mongo(item) for item in data]
    return data

@upload_bp.route('/upload_image', methods=['POST'])
@jwt_required()
def upload_image_from_flutter():
    return upload_masked_image()

@upload_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_plain_image():
    return upload_masked_image()

@upload_bp.route('/upload_masked_image', methods=['POST'])
@jwt_required()
def upload_masked_image():
    user_id = get_jwt_identity()
    start_total = time.perf_counter()

    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 필요합니다.'}), 400
    file = request.files['file']
    image_type = request.form.get('image_type', 'normal')

    yolo_results_json_str = request.form.get('yolo_results_json')
    yolo_inference_data = []
    if yolo_results_json_str:
        try:
            yolo_inference_data = json.loads(yolo_results_json_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'YOLO 결과 JSON 형식 오류: {e}'}), 400

    survey_json_str = request.form.get('survey')
    survey_data = {}
    if survey_json_str:
        try:
            survey_data = json.loads(survey_json_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'survey JSON 형식 오류: {e}'}), 400

    if file.filename == '':
        return jsonify({'error': '파일명이 비어 있습니다.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        original_filename = secure_filename(file.filename)
        base_name = f"{user_id}_{timestamp}_{original_filename}"
        base_name = os.path.splitext(base_name)[0] + ".png"

        upload_dir = current_app.config['UPLOAD_FOLDER_ORIGINAL']
        processed_dir_1 = current_app.config['PROCESSED_FOLDER_MODEL1']
        processed_dir_2 = current_app.config['PROCESSED_FOLDER_MODEL2']
        processed_dir_3 = current_app.config['PROCESSED_FOLDER_MODEL3']
        xmodel1_dir = current_app.config['PROCESSED_FOLDER_XMODEL1']
        xmodel2_dir = current_app.config['PROCESSED_FOLDER_XMODEL2']

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(processed_dir_1, exist_ok=True)
        os.makedirs(processed_dir_2, exist_ok=True)
        os.makedirs(processed_dir_3, exist_ok=True)
        os.makedirs(xmodel1_dir, exist_ok=True)
        os.makedirs(xmodel2_dir, exist_ok=True)

        original_path = os.path.join(upload_dir, base_name)
        file.save(original_path)

        image = Image.open(original_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ───────────────────────────── X-ray 처리 ─────────────────────────────
        if image_type == 'xray':
            from ai_model.xray_detector import detect_xray
            from ai_model.predict_implant_manufacturer import classify_implants_from_xray

            detect_start = time.perf_counter()
            detect_result = detect_xray(original_path)
            _sync_cuda()
            detect_elapsed = int((time.perf_counter() - detect_start) * 1000)

            filtered_boxes = detect_result['detections']
            summary_text = detect_result.get('summary', '감지된 객체가 없습니다.')

            yolo_predictions = [
                {
                    "class_id": det['class_id'],
                    "class_name": det['class_name'],
                    "confidence": round(det['confidence'], 3),
                    "bbox": det['bbox']
                } for det in filtered_boxes
            ]

            impl_start = time.perf_counter()
            implant_classification_results = classify_implants_from_xray(original_path)
            _sync_cuda()
            impl_elapsed = int((time.perf_counter() - impl_start) * 1000)

            total_elapsed = int((time.perf_counter() - start_total) * 1000)
            upload_logger.info(
                f"[📸 추론 완료] 총 {total_elapsed}ms "
                f"(탐지: {detect_elapsed}ms, 임플란트분류: {impl_elapsed}ms, user_id={user_id})"
            )

            image_draw = image.copy()
            draw = ImageDraw.Draw(image_draw)
            font = _load_font(18)
            for det in filtered_boxes:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['class_name']} {det['confidence']:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                draw.text((x1, max(y1 - 20, 0)), label, font=font, fill="blue")
            processed_path_x1 = os.path.join(xmodel1_dir, base_name)
            image_draw.save(processed_path_x1)

            image_with_manufacturer = image.copy()
            draw = ImageDraw.Draw(image_with_manufacturer)
            for result in implant_classification_results:
                x1, y1, x2, y2 = result['bbox']
                name = result['predicted_manufacturer_name']
                conf = result['confidence'] * 100
                label = f"{name} ({conf:.1f}%)"
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(y1 - 22, 0)), label, fill="yellow", font=font)
            processed_path_x2 = os.path.join(xmodel2_dir, base_name)
            image_with_manufacturer.save(processed_path_x2)

            mongo_data = {
                'user_id': user_id,
                'image_type': image_type,
                'survey': survey_data,
                'original_image_path': f"/images/original/{base_name}",
                'model1_image_path': f"/images/xmodel1/{base_name}",
                'model2_image_path': f"/images/xmodel2/{base_name}",
                'model1_inference_result': {
                    'used_model': 'xray_detect_best.pt',
                    'predictions': yolo_predictions,
                    'summary': summary_text
                },
                'implant_classification_result': implant_classification_results,
                'timestamp': datetime.now()
            }
            mongo_client = MongoDBClient()
            inserted_id = mongo_client.insert_result(_convert_for_mongo(mongo_data))

            return jsonify({
                'message': 'X-ray 이미지 YOLO + 임플란트 분류 완료',
                'inference_result_id': str(inserted_id),
                'image_type': image_type,
                'original_image_path': f"/images/original/{base_name}",
                'model1_image_path': f"/images/xmodel1/{base_name}",
                'model2_image_path': f"/images/xmodel2/{base_name}",
                'model1_inference_result': {
                    'used_model': 'xray_detect_best.pt',
                    'predictions': yolo_predictions,
                    'summary': summary_text
                },
                'implant_classification_result': implant_classification_results
            }), 200

        # ─────────────────────────── 일반 이미지 처리 ───────────────────────────
        t1_start = time.perf_counter()
        processed_path_1 = os.path.join(processed_dir_1, base_name)
        (
            masked_image_1,
            disease_detections_list,
            backend_model_confidence,
            backend_model_name,
            disease_label,
        ) = predict_overlayed_image(image, processed_path_1)
        try:
            masked_image_1.save(processed_path_1)
        except Exception:
            pass
        t1_elapsed = int((time.perf_counter() - t1_start) * 1000)

        t2_start = time.perf_counter()
        processed_path_2 = os.path.join(processed_dir_2, base_name)
        (
            masked_image_2,
            hygiene_detections_list,
            hygiene_confidence,
            hygiene_model_name,
            hygiene_main_label,
        ) = hygiene_predictor.predict_mask_and_overlay_with_all(image, processed_path_2)
        try:
            Image.open(processed_path_2).close()
        except Exception as e:
            upload_logger.warning(f"[DEBUG] model2 이미지 확인 실패: {e}")
        t2_elapsed = int((time.perf_counter() - t2_start) * 1000)

        t3_start = time.perf_counter()
        processed_path_3 = os.path.join(processed_dir_3, base_name)
        tooth_number_predictor.predict_mask_and_overlay_only(image, processed_path_3)
        try:
            Image.open(processed_path_3).close()
        except Exception as e:
            upload_logger.warning(f"[DEBUG] model3 이미지 확인 실패: {e}")
        tooth_info_list = tooth_number_predictor.get_all_class_info_json(image)

        # 🦷 중복된 치아 번호 제거 (가장 높은 confidence만 유지)
        # 키를 tooth_number_fdi로만 설정하여 동일한 치아 번호는 하나만 남김
        unique_tooth_info = {}
        for tooth_info in tooth_info_list:
            tooth_number = tooth_info['tooth_number_fdi']
            if tooth_number not in unique_tooth_info or tooth_info['confidence'] > unique_tooth_info[tooth_number]['confidence']:
                unique_tooth_info[tooth_number] = tooth_info
        
        filtered_tooth_info_list = list(unique_tooth_info.values())
        
        t3_elapsed = int((time.perf_counter() - t3_start) * 1000)
        
        final_matched_results = combine_results(
            image.size,
            disease_detections_list,
            hygiene_detections_list,
            filtered_tooth_info_list # 수정된 리스트 전달
        )

        total_elapsed = int((time.perf_counter() - start_total) * 1000)
        upload_logger.info(
            f"[📸 추론 완료] 총 {total_elapsed}ms "
            f"(질병(disease): {t1_elapsed}ms, 위생(Hygiene): {t2_elapsed}ms, 치아번호(number): {t3_elapsed}ms, "
            f"user_id={user_id})"
        )

        # ⚠️ DB 저장을 위한 데이터에서 'mask_array' 필드를 제거
        disease_detections_for_db = _convert_for_mongo([
            {k: v for k, v in det.items() if k != 'mask_array'}
            for det in disease_detections_list
        ])
        hygiene_detections_for_db = _convert_for_mongo([
            {k: v for k, v in det.items() if k != 'mask_array'}
            for det in hygiene_detections_list
        ])
        
        mongo_client = MongoDBClient()
        inserted_id = mongo_client.insert_result({
            'user_id': user_id,
            'image_type': image_type,
            'survey': survey_data,
            'original_image_path': f"/images/original/{base_name}",
            'original_image_yolo_detections': yolo_inference_data,
            'model1_image_path': f"/images/model1/{base_name}",
            'model1_inference_result': {
                'message': 'model1 마스크 생성 완료',
                'confidence': backend_model_confidence,
                'used_model': backend_model_name,
                'label': disease_label,
                'detections': disease_detections_for_db,
            },
            'model2_image_path': f"/images/model2/{base_name}",
            'model2_inference_result': {
                'message': 'model2 마스크 생성 완료',
                'confidence': hygiene_confidence,
                'label': hygiene_main_label,
                'detections': hygiene_detections_for_db,
                'used_model': hygiene_model_name
            },
            'model3_image_path': f"/images/model3/{base_name}",
            'model3_inference_result': {
                'message': 'model3 마스크 생성 완료',
                'predicted_tooth_info': filtered_tooth_info_list # 필터링된 리스트 저장
            },
            'matched_results': _convert_for_mongo(final_matched_results),
            'timestamp': datetime.now()
        })

        # ⚠️ API 응답을 위한 데이터에서도 'mask_array' 필드를 제거
        disease_detections_for_api = [
            {k: v for k, v in det.items() if k != 'mask_array'}
            for det in disease_detections_list
        ]
        hygiene_detections_for_api = [
            {k: v for k, v in det.items() if k != 'mask_array'}
            for det in hygiene_detections_list
        ]

        return jsonify({
            'message': '3개 모델 처리 및 저장 완료',
            'inference_result_id': str(inserted_id),
            'image_type': image_type,
            'original_image_path': f"/images/original/{base_name}",
            'original_image_yolo_detections': yolo_inference_data,
            'model1_image_path': f"/images/model1/{base_name}",
            'model1_inference_result': {
                'message': 'model1 마스크 생성 완료',
                'confidence': backend_model_confidence,
                'used_model': backend_model_name,
                'label': disease_label,
                'detections': disease_detections_for_api,
            },
            'model2_image_path': f"/images/model2/{base_name}",
            'model2_inference_result': {
                'message': 'model2 마스크 생성 완료',
                'confidence': hygiene_confidence,
                'label': hygiene_main_label,
                'detections': hygiene_detections_for_api,
                'used_model': hygiene_model_name
            },
            'model3_image_path': f"/images/model3/{base_name}",
            'model3_inference_result': {
                'message': 'model3 마스크 생성 완료',
                'predicted_tooth_info': filtered_tooth_info_list # 필터링된 리스트 응답
            },
            'matched_results': _convert_for_mongo(final_matched_results)
        }), 200

    except Exception as e:
        upload_logger.exception("Upload error")
        return jsonify({'error': f'서버 처리 중 오류: {str(e)}'}), 500