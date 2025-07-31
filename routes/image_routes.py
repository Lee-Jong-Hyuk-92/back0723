import os
from datetime import datetime
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename

from ai_model.model import perform_inference  # AI 모델 임포트

# Blueprint 생성
image_bp = Blueprint('image', __name__)

# 파일 확장자 확인
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@image_bp.route('/upload_image', methods=['POST'])
def upload_image():
    print("📅 [요청 수신] /upload_image")

    if 'image' not in request.files:
        print("❌ [에러] 'image' 파일 파트 없음")
        return jsonify({'error': 'No image file part'}), 400

    image_file = request.files['image']
    user_id = request.form.get('user_id', 'anonymous')
    print(f"👤 [사용자 ID] {user_id}")

    if image_file.filename == '':
        print("❌ [에러] 선택된 이미지 파일 없음")
        return jsonify({'error': 'No selected image file'}), 400

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{user_id}_{timestamp}_{filename}"
        original_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)

        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"📂 [원본 저장 중] {original_path}")
        image_file.save(original_path)
        print(f"✅ [원본 저장 완료]")

        inference_output = perform_inference(original_path, current_app.config['PROCESSED_UPLOAD_FOLDER'])
        if inference_output.get("error"):
            print(f"❌ [AI 추론 에러] {inference_output['error']}")
            return jsonify({'error': f"Inference failed: {inference_output['error']}"}), 500

        result_json = {
            "prediction": inference_output.get("prediction"),
            "details": inference_output.get("details", [])
        }
        processed_path = inference_output.get("processed_image_path")
        if not processed_path:
            print("❌ [에러] 처리된 이미지 경로 없음")
            return jsonify({'error': 'Missing processed image path'}), 500

        try:
            mongo_client_instance = current_app.extensions['mongo_client']
            print("📜 [MongoDB 'history' 저장 중]")
            mongo_client_instance.insert_into_collection(
                collection_name='history',
                document={
                    'user_id': user_id,
                    'original_image_filename': unique_filename,
                    'original_image_full_path': original_path,
                    'upload_timestamp': datetime.now(),
                    'inference_result': result_json,
                    'processed_image_path': processed_path
                }
            )
            print("✅ [MongoDB 'history' 저장 완료]")

        except Exception as e:
            print(f"❌ [MongoDB 저장 에러] {str(e)}")
            return jsonify({'error': f'MongoDB insert error: {e}'}), 500

        return jsonify({
            'message': 'Image uploaded and processed',
            'image_url': f"/processed_uploads/{os.path.basename(processed_path)}",
            'inference_data': result_json,
            'user_id': user_id
        }), 200

    print("❌ [에러] 유효하지 않은 파일 타입")
    return jsonify({'error': 'Invalid file type'}), 400

@image_bp.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@image_bp.route('/processed_uploads/<filename>')
def serve_processed(filename):
    return send_from_directory(current_app.config['PROCESSED_UPLOAD_FOLDER'], filename)

# ✅ 추가된 경로: 챗봇에서 마스크 이미지 보여주기용
@image_bp.route('/images/<path:subpath>')
def serve_result_image(subpath):
    # 예: subpath = "model1/파일명.png" 또는 "original/파일명.png"
    full_path = os.path.join("images", subpath)
    dir_path = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(dir_path, filename)
