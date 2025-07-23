from flask import Blueprint, request, jsonify, current_app as app
from pymongo.errors import ConnectionFailure, OperationFailure
import time

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chatbot', methods=['POST'])
def chatbot_reply():
    start_time = time.time()
    user_message = "알 수 없는 메시지" # 로그를 위해 초기화
    patient_id = "알 수 없는 ID"     # 로그를 위해 초기화

    try:
        data = request.json
        user_message = data.get('message', '메시지 없음')
        patient_id = data.get('patient_id', 'ID 없음')

        app.logger.info(f"[💬 챗봇 요청] 사용자 메시지: '{user_message}', 환자 ID: '{patient_id}'")
        print(f"[💬 챗봇 요청] 사용자 메시지: '{user_message}', 환자 ID: '{patient_id}'")

        # ✅ MongoDB에서 환자 진료 기록 조회 시도
        mongo_client = app.extensions.get("mongo_client")
        if not mongo_client:
            app.logger.error("[❌ MongoDB] mongo_client가 앱 익스텐션에 없습니다.")
            print("[❌ MongoDB] mongo_client가 앱 익스텐션에 없습니다.")
            return jsonify({'response': '서버 오류: 데이터베이스 클라이언트가 초기화되지 않았습니다.', 'elapsed_time': round(time.time() - start_time, 2)}), 500

        try:
            db_collection = mongo_client.get_collection("inference_results")
            # MongoDB 연결 자체를 확인 (여기서는 클라이언트 초기화 시 확인했으므로 생략 가능하나, 추가적인 안정성을 위해 포함)
            # db_collection.find_one({}) # 간단한 쿼리로 연결 확인
        except ConnectionFailure as e:
            app.logger.error(f"[❌ MongoDB] MongoDB 연결 실패: {e}")
            print(f"[❌ MongoDB] MongoDB 연결 실패: {e}")
            return jsonify({'response': '데이터베이스 연결에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.', 'elapsed_time': round(time.time() - start_time, 2)}), 500
        except Exception as e:
            app.logger.error(f"[❌ MongoDB] MongoDB 컬렉션 접근 중 알 수 없는 오류: {e}")
            print(f"[❌ MongoDB] MongoDB 컬렉션 접근 중 알 수 없는 오류: {e}")
            return jsonify({'response': '데이터베이스 접근 중 알 수 없는 오류가 발생했습니다.', 'elapsed_time': round(time.time() - start_time, 2)}), 500

        # 여기서 patient_id가 문자열이 아닌 경우를 대비하여 명시적으로 문자열로 변환
        query_patient_id = str(patient_id)
        record = db_collection.find_one({"user_id": query_patient_id})

        record_status_log = ""
        if record:
            record_text = f"환자 기록: {record}"
            record_status_log = "✅ 환자 기록을 DB에서 찾았습니다."
        else:
            record_text = "환자 기록 없음"
            record_status_log = "ℹ️ 해당 환자 ID로 DB에서 기록을 찾지 못했습니다."
            if patient_id == 'ID 없음': # patient_id가 제대로 넘어오지 않은 경우
                record_status_log += " (환자 ID가 제대로 전달되지 않았을 수 있습니다.)"

        app.logger.info(f"[🔍 DB 조회 결과] {record_status_log} 조회된 기록: {record}")
        print(f"[🔍 DB 조회 결과] {record_status_log} 조회된 기록: {record}")


        # ✅ 이미 초기화된 Gemini 모델 사용
        gemini_model = app.extensions.get("gemini_model")
        if not gemini_model:
            app.logger.error("[❌ Gemini] Gemini 모델이 앱 익스텐션에 없습니다.")
            print("[❌ Gemini] Gemini 모델이 앱 익스텐션에 없습니다.")
            return jsonify({'response': '서버 오류: AI 모델이 초기화되지 않았습니다.', 'elapsed_time': round(time.time() - start_time, 2)}), 500

        chat = gemini_model.start_chat()

        prompt = f"""
        환자 기록은 다음과 같습니다:\n{record_text}\n\n
        환자가 다음과 같은 질문을 했습니다:\n"{user_message}"\n
        이에 대해 친절하게 설명해주세요.
        """
        app.logger.info(f"[🤖 Gemini 요청] Gemini 모델에 전달될 프롬프트:\n{prompt[:500]}...") # 프롬프트 일부만 로깅
        print(f"[🤖 Gemini 요청] Gemini 모델에 전달될 프롬프트:\n{prompt[:500]}...")

        try:
            response = chat.send_message(prompt)
            reply = response.text
            app.logger.info(f"[✅ Gemini 응답] Gemini 모델로부터 응답 받음. 내용 길이: {len(reply)} 문자")
            print(f"[✅ Gemini 응답] Gemini 모델로부터 응답 받음. 내용 길이: {len(reply)} 문자")
            app.logger.info(f"[✅ Gemini 응답] Gemini 모델의 실제 응답:\n{reply[:500]}...") # 응답 일부만 로깅
            print(f"[✅ Gemini 응답] Gemini 모델의 실제 응답:\n{reply[:500]}...")

        except Exception as e:
            app.logger.error(f"[❌ Gemini] Gemini 모델 응답 생성 중 오류 발생: {e}")
            print(f"[❌ Gemini] Gemini 모델 응답 생성 중 오류 발생: {e}")
            reply = "AI 모델 응답 생성 중 문제가 발생했습니다. 다시 시도해 주세요."

        elapsed_time = round(time.time() - start_time, 2)
        app.logger.info(f"[⏱️ 챗봇 응답] 총 응답 시간: {elapsed_time}초")
        print(f"[⏱️ 챗봇 응답] 총 응답 시간: {elapsed_time}초")

        return jsonify({'response': reply, 'elapsed_time': elapsed_time})

    except Exception as e:
        app.logger.error(f"[❌ 챗봇 오류] 챗봇 처리 중 예외 발생 (사용자 메시지: '{user_message}', 환자 ID: '{patient_id}'): {e}", exc_info=True)
        print(f"[❌ 챗봇 오류] 챗봇 처리 중 예외 발생 (사용자 메시지: '{user_message}', 환자 ID: '{patient_id}'): {e}")
        return jsonify({'response': '챗봇 시스템에 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.', 'elapsed_time': round(time.time() - start_time, 2)}), 500