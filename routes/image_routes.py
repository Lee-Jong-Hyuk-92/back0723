# C:\Users\sptzk\Desktop\backend0709-1\routes\image_routes.py
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename

from ai_model.model import perform_inference # AI 모델 임포트
# from models.model import MongoDBClient # 이 줄은 이제 필요 없습니다. 아래에서 current_app.extensions를 통해 접근합니다.

# Blueprint 생성
image_bp = Blueprint('image', __name__)

# 파일 확장자 확인 (app.py의 함수를 직접 가져오기 어려우므로, 여기에 다시 정의)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@image_bp.route('/upload_image', methods=['POST'])
def upload_image():
    print("📅 [요청 수신] /upload_image") # 로그 추가

    if 'image' not in request.files:
        print("❌ [에러] 'image' 파일 파트 없음") # 로그 추가
        return jsonify({'error': 'No image file part'}), 400

    image_file = request.files['image']
    # user_id를 요청의 form 데이터에서 가져옵니다.
    # 클라이언트(모바일 앱)에서 'user_id'라는 키로 전송해야 합니다.
    user_id = request.form.get('user_id', 'anonymous') # user_id가 없으면 'anonymous'로 설정
    print(f"👤 [사용자 ID] {user_id}") # 로그 추가

    if image_file.filename == '':
        print("❌ [에러] 선택된 이미지 파일 없음") # 로그 추가
        return jsonify({'error': 'No selected image file'}), 400

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{user_id}_{timestamp}_{filename}" # user_id를 파일명에 추가
        original_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 원본 이미지 저장 경로 확인 및 생성
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"📂 [원본 저장 중] {original_path}") # 로그 추가
        image_file.save(original_path)
        print(f"✅ [원본 저장 완료]") # 로그 추가

        inference_output = perform_inference(original_path, current_app.config['PROCESSED_UPLOAD_FOLDER'])
        if inference_output.get("error"):
            print(f"❌ [AI 추론 에러] {inference_output['error']}") # 로그 추가
            return jsonify({'error': f"Inference failed: {inference_output['error']}"}), 500

        result_json = {
            "prediction": inference_output.get("prediction"),
            "details": inference_output.get("details", [])
        }
        processed_path = inference_output.get("processed_image_path")
        if not processed_path:
            print("❌ [에러] 처리된 이미지 경로 없음") # 로그 추가
            return jsonify({'error': 'Missing processed image path'}), 500

        try:
            # app.py에서 초기화된 MongoDBClient 인스턴스에 접근
            mongo_client_instance = current_app.extensions['mongo_client']
            
            # MongoDB 'history' 컬렉션에 데이터 저장
            print("📜 [MongoDB 'history' 저장 중]") # 로그 추가
            mongo_client_instance.insert_into_collection(
                collection_name='history', # 'history' 컬렉션 지정
                document={
                    'user_id': user_id,
                    'original_image_filename': unique_filename,
                    'original_image_full_path': original_path, # 원본 이미지의 전체 경로
                    'upload_timestamp': datetime.now(),
                    'inference_result': result_json,
                    'processed_image_path': processed_path # 처리된 이미지 경로도 함께 저장 (선택 사항)
                }
            )
            print("✅ [MongoDB 'history' 저장 완료]") # 로그 추가

        except Exception as e:
            print(f"❌ [MongoDB 저장 에러] {str(e)}") # 로그 추가
            return jsonify({'error': f'MongoDB insert error: {e}'}), 500

        return jsonify({
            'message': 'Image uploaded and processed',
            'image_url': f"/processed_uploads/{os.path.basename(processed_path)}",
            'inference_data': result_json,
            'user_id': user_id # 클라이언트에 user_id 반환 (선택 사항)
        }), 200

    print("❌ [에러] 유효하지 않은 파일 타입") # 로그 추가
    return jsonify({'error': 'Invalid file type'}), 400

@image_bp.route('/uploads/<filename>')
def serve_upload(filename):
    # 이 경로는 원본 이미지를 웹에서 접근할 수 있게 합니다.
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@image_bp.route('/processed_uploads/<filename>')
def serve_processed(filename):
    # 이 경로는 처리된 이미지를 웹에서 접근할 수 있게 합니다.
    return send_from_directory(current_app.config['PROCESSED_UPLOAD_FOLDER'], filename)