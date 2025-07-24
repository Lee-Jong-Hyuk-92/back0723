# 안쓰는 파일
from flask import Blueprint, request, jsonify, current_app as app
import time
import os
import re
from datetime import datetime
from PIL import Image
import io
from vertexai.preview.generative_models import Part

chatbot_med_bp = Blueprint('chatbot_medgemma', __name__)

def preprocess_image_for_medgemma(image_full_path):
    try:
        match = re.search(r'/images/original/(.+)', image_full_path)
        if not match:
            app.logger.error(f"이미지 URL 형식 오류: {image_full_path}")
            return None
        relative_path = match.group(1).replace('/', os.sep)
        image_file_path = os.path.join(app.config['UPLOAD_FOLDER_ORIGINAL'], relative_path)
        if not os.path.exists(image_file_path):
            app.logger.error(f"이미지 없음: {image_file_path}")
            return None
        img = Image.open(image_file_path).convert('RGB').resize((896, 896), Image.Resampling.LANCZOS)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='JPEG')
        return byte_arr.getvalue()
    except Exception as e:
        app.logger.error(f"MedGemma 전처리 실패: {e}", exc_info=True)
        return None

@chatbot_med_bp.route("/api/chat-medgemma", methods=["POST"])
def chat_with_medgemma():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message")

    if not user_id or not user_message:
        return jsonify({"error": "user_id와 message는 필수입니다."}), 400

    try:
        mongo_client = app.extensions['mongo_client']
        db = mongo_client.db
        records = list(db['inference_results'].find({"user_id": user_id}).sort("timestamp", -1))

        if not records:
            return jsonify({"response": "진료 기록이 없습니다. 이미지를 먼저 업로드해 주세요.", "image_url": None})

        # MedGemma 모델 준비
        medgemma_model = app.extensions.get('medgemma_model')
        if not medgemma_model:
            return jsonify({"error": "MedGemma 모델이 설정되지 않았습니다."}), 500

        # 특정 날짜나 파일명 파싱
        found_record = None
        target_date = re.search(r'(\d{4}[년\s.-]?\d{1,2}[월\s.-]?\d{1,2}[일])', user_message)
        target_file = re.search(r'(\d{14}_web_image\.(png|jpg))', user_message)

        if target_date:
            date_str = target_date.group(1).replace('년', '').replace('월', '').replace('일', '').replace('.', '-').replace(' ', '')
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                for record in records:
                    ts = record.get('timestamp', '')
                    if ts and datetime.fromisoformat(ts.replace('Z', '+00:00')).date() == date_obj:
                        found_record = record
                        break
            except:
                pass
        elif target_file:
            filename = target_file.group(1)
            for record in records:
                if os.path.basename(record.get('original_image_path', '')) == filename:
                    found_record = record
                    break

        if not found_record:
            found_record = records[0]  # 최신 기록

        # 이미지 준비
        image_path = found_record.get('original_image_path', '')
        image_url = f"http://192.168.0.19:5000{image_path}" if image_path else None
        image_bytes = preprocess_image_for_medgemma(image_path)
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg") if image_bytes else None

        # 시스템 프롬프트 구성
        m1 = found_record.get('model1_inference_result', {})
        m2 = found_record.get('model2_inference_result', {})
        m3 = found_record.get('model3_inference_result', {})

        instruction = f"""
        환자 진단 기록 ({found_record.get('timestamp')}):
        - 모델1 (질병): {m1.get('label', '없음')} ({m1.get('confidence', 0.0):.1%})
        - 모델2 (위생): {m2.get('label', '없음')} ({m2.get('confidence', 0.0):.1%})
        - 모델3 (치아 번호): {m3.get('tooth_number_fdi', '없음')} ({m3.get('confidence', 0.0):.1%})
        의사 코멘트: {found_record.get('doctor_comment', '없음')}
        위 정보를 참고하여 환자의 질문에 답변하세요.
        """

        contents = []
        if image_part:
            contents.append(image_part)
        contents.append(Part.from_text(instruction))
        contents.append(Part.from_text(user_message))

        chat = medgemma_model.start_chat()
        response = chat.send_message(contents)

        return jsonify({
            "response": response.text,
            "image_url": image_url
        })

    except Exception as e:
        app.logger.error(f"MedGemma 챗봇 오류: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
