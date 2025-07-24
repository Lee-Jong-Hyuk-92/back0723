from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import os
import time
import logging

# ✅ Gemini 전용 로거 분리 설정
gemini_logger = logging.getLogger("gemini_logger")
gemini_logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "gemini_times.log")

if not gemini_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    gemini_logger.addHandler(fh)

multimodal_gemini_bp = Blueprint('multimodal_gemini', __name__)

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

@multimodal_gemini_bp.route("/multimodal_gemini", methods=["POST"])
def handle_ai_opinion():
    start_time = time.perf_counter()  # ✅ 속도 측정 시작

    data = request.get_json()

    mongo_client = current_app.extensions.get("mongo_client")
    collection = mongo_client.get_collection("inference_results")

    image_url = data.get("image_url")
    inference_result_id = data.get("inference_result_id")  # ✅ MongoDB _id
    model1 = data.get("model1Label")
    conf1 = data.get("model1Confidence")
    model2 = data.get("model2Label")
    conf2 = data.get("model2Confidence")
    tooth_number = data.get("model3ToothNumber")
    conf3 = data.get("model3Confidence")

    # ✅ 1. 기존 AI_result 확인
    doc = collection.find_one({"_id": ObjectId(inference_result_id)})
    if not doc:
        return jsonify({"error": "해당 분석 결과를 찾을 수 없습니다."}), 404

    if "AI_result" in doc:
        print("📄 기존 AI_result 반환")
        return jsonify({"message": doc["AI_result"]})

    # ✅ 2. 없으면 Gemini로 생성
    print("🔍 기존 AI_result 없음 → Gemini 호출 시작")

    try:
        img_resp = requests.get(image_url)
        img = Image.open(BytesIO(img_resp.content))
    except Exception as e:
        return jsonify({"error": f"이미지 로드 실패: {str(e)}"}), 400

    prompt = f"""
너는 치과 전문의야. 아래는 AI가 구강 이미지를 분석한 결과야. 이 정보를 바탕으로 환자의 상태를 상세히 설명해줘.

- 질병 예측: {model1}, 확신도: {conf1:.2f}
- 위생 예측: {model2}, 확신도: {conf2:.2f}
- 치아 번호: {tooth_number}, 확신도: {conf3:.2f}

해당 이미지와 결과를 함께 고려해 설명해줘. 마지막엔 결론 한 줄로 요약해.
"""

    try:
        response = model.generate_content([prompt, img])
        result_text = response.text

        # ✅ 3. MongoDB에 결과 저장
        collection.update_one(
            {"_id": ObjectId(inference_result_id)},
            {"$set": {"AI_result": result_text}}
        )

        total_time = time.perf_counter() - start_time
        gemini_logger.info(f"[🧠 Gemini 멀티모달 추론 시간] {total_time:.4f}s (inference_result_id={inference_result_id})")

        print("✅ Gemini 응답 저장 완료")
        return jsonify({"message": result_text})
    except Exception as e:
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500
