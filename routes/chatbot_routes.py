from flask import Blueprint, request, jsonify, current_app as app
from pymongo.errors import ConnectionFailure
import time
import logging
import os
from flask_jwt_extended import jwt_required, get_jwt_identity

# ✅ 챗봇 전용 로거 분리
chatbot_logger = logging.getLogger("chatbot_logger")
chatbot_logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "chatbot_times.log")

if not chatbot_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    chatbot_logger.addHandler(fh)

chatbot_bp = Blueprint('chatbot', __name__)

# ✅ 이미지 관련 요청 판단 함수
def wants_image(user_message: str) -> bool:
    keywords = ["사진", "이미지", "보여", "그려", "그림", "사진 보여", "보여줘", "보여줄 수"]
    return any(kw in user_message for kw in keywords)

@chatbot_bp.route('/chatbot', methods=['POST'])
@jwt_required()  # ✅ 추가
def chatbot_reply():
    start_time = time.time()
    user_message = "알 수 없는 메시지"
    patient_id = get_jwt_identity()  # ✅ JWT에서 추출

    try:
        data = request.json
        user_message = data.get('message', '메시지 없음')

        app.logger.info(f"[💬 챗봇 요청] 사용자 메시지: '{user_message}', 환자 ID: '{patient_id}'")
        print(f"[💬 챗봇 요청] 사용자 메시지: '{user_message}', 환자 ID: '{patient_id}'")

        mongo_client = app.extensions.get("mongo_client")
        if not mongo_client:
            app.logger.error("[❌ MongoDB] mongo_client가 앱 익스텐션에 없습니다.")
            return jsonify({
                'response': '서버 오류: DB 클라이언트가 초기화되지 않았습니다.',
                'elapsed_time': round(time.time() - start_time, 2)
            }), 500

        try:
            db_collection = mongo_client.get_collection("inference_results")
        except ConnectionFailure as e:
            app.logger.error(f"[❌ MongoDB] MongoDB 연결 실패: {e}")
            return jsonify({
                'response': '데이터베이스 연결에 문제가 발생했습니다.',
                'elapsed_time': round(time.time() - start_time, 2)
            }), 500

        query_patient_id = str(patient_id)
        records = list(db_collection.find({"user_id": query_patient_id}))
        diagnosis_count = len(records)

        def summarize_record(r, index):
            ts = r.get('timestamp')
            date_str = ts.strftime('%Y-%m-%d %H:%M') if ts else '날짜 없음'
            label1 = r.get('model1_inference_result', {}).get('label', '없음')
            label2 = r.get('model2_inference_result', {}).get('label', '없음')
            label3 = r.get('model3_inference_result', {}).get('tooth_number_fdi', '없음')
            return f"- 기록 {index+1} ({date_str}) → 질병: {label1}, 위생: {label2}, 치아번호: {label3}"

        if diagnosis_count > 0:
            summaries = "\n".join([summarize_record(r, i) for i, r in enumerate(records)])
            record_summary = f"총 {diagnosis_count}건의 진단 기록 요약:\n{summaries}"
            record_status_log = f"✅ 진단 기록 {diagnosis_count}건 조회됨"
        else:
            record_summary = "진단 기록 없음"
            record_status_log = "ℹ️ 진단 기록 없음"

        app.logger.info(f"[🔍 DB 조회 결과] {record_status_log}")
        print(f"[🔍 DB 조회 결과] {record_status_log}")

        gemini_model = app.extensions.get("gemini_model")
        if not gemini_model:
            app.logger.error("[❌ Gemini] Gemini 모델이 앱 익스텐션에 없습니다.")
            return jsonify({
                'response': '서버 오류: AI 모델이 초기화되지 않았습니다.',
                'elapsed_time': round(time.time() - start_time, 2)
            }), 500

        chat = gemini_model.start_chat()

        prompt = f"""
        환자 ID '{query_patient_id}'는 지금까지 총 {diagnosis_count}건의 사진 진단 기록이 있습니다.

        {record_summary}

        사용자 질문:
        "{user_message}"

        위 내용을 참고하여 의료 기록 기반으로 정확하고 친절하게 답변해주세요.
        """
        app.logger.info(f"[🤖 Gemini 요청] 프롬프트 일부:\n{prompt[:500]}...")
        print(f"[🤖 Gemini 요청] 프롬프트 일부:\n{prompt[:500]}...")

        try:
            response = chat.send_message(prompt)
            reply = response.text
            app.logger.info(f"[✅ Gemini 응답] 길이: {len(reply)}자 / 내용:\n{reply[:500]}...")
            print(f"[✅ Gemini 응답] 길이: {len(reply)}자 / 내용:\n{reply[:500]}...")
        except Exception as e:
            app.logger.error(f"[❌ Gemini 오류] 응답 생성 실패: {e}")
            print(f"[❌ Gemini 오류] 응답 생성 실패: {e}")
            reply = "AI 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."

        # ✅ 이미지 조건부 반환
        image_urls = {}

        if diagnosis_count > 0 and wants_image(user_message):
            # ✅ "N번째 기록" 요청
            import re
            nth_match = re.search(r'(\d+)[번째\s]*기록', user_message)
            if nth_match:
                n = int(nth_match.group(1))
                if 1 <= n <= diagnosis_count:
                    selected_record = records[n - 1]
                else:
                    reply += f"\n\n⚠️ 총 {diagnosis_count}개의 기록 중 {n}번째 기록은 존재하지 않습니다."
                    selected_record = None
            elif "가장 오래된" in user_message:
                selected_record = records[0]
            elif "가장 최근" in user_message:
                selected_record = records[-1]
            else:
                reply += "\n\n⚠️ 진단 기록이 여러 건 존재합니다. 특정 기록을 확인하시려면 '가장 오래된 기록', '3번째 기록'과 같이 지정해주세요.\n\n또는 '이전 결과 보기' 화면에서 확인하실 수 있습니다."
                selected_record = None

            def to_url(path):
                return f"http://192.168.0.19:5000{path}" if path else None

            if selected_record:
                image_urls = {
                    k: to_url(selected_record.get(f"{k}_image_path"))
                    for k in ["original", "model1", "model2", "model3"]
                }
                image_urls = {k: v for k, v in image_urls.items() if v}

        elapsed_time = round(time.time() - start_time, 2)
        app.logger.info(f"[⏱️ 응답 시간] {elapsed_time}초")
        chatbot_logger.info(f"[🤖 챗봇 응답 시간] {elapsed_time:.2f}s (user_id={patient_id}, 메시지: {user_message})")

        return jsonify({
            'response': reply,
            'image_urls': image_urls,
            'elapsed_time': elapsed_time
        })

    except Exception as e:
        app.logger.error(f"[❌ 챗봇 오류] 예외 발생: {e}", exc_info=True)
        return jsonify({
            'response': '시스템 오류가 발생했습니다.',
            'elapsed_time': round(time.time() - start_time, 2)
        }), 500
