from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import os
import time
import logging

# ✅ Gemini 로깅 설정
gemini_logger = logging.getLogger("gemini_logger_xray")
gemini_logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "gemini_xray_times.log")

if not gemini_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    gemini_logger.addHandler(fh)

multimodal_gemini_xray_bp = Blueprint('multimodal_gemini_xray', __name__)

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

@multimodal_gemini_xray_bp.route("/multimodal_gemini_xray", methods=["POST"])
def handle_xray_ai_opinion():
    start_time = time.perf_counter()
    data = request.get_json()

    mongo_client = current_app.extensions.get("mongo_client")
    collection = mongo_client.get_collection("inference_results")

    image_url = data.get("image_url")
    inference_result_id = data.get("inference_result_id")
    model1 = data.get("model1Label") or "unknown"
    conf1 = data.get("model1Confidence") or 0.0
    count = data.get("predictionCount") or 0

    doc = collection.find_one({"_id": ObjectId(inference_result_id)})
    if not doc:
        return jsonify({"error": "해당 분석 결과를 찾을 수 없습니다."}), 404

    if "AI_result" in doc:
        print("📄 기존 X-ray AI_result 반환")
        return jsonify({"message": doc["AI_result"]})

    print("🔍 Gemini X-ray 멀티모달 분석 시작")

    try:
        img_resp = requests.get(image_url, verify=False)
        img_resp.raise_for_status()
        img = Image.open(BytesIO(img_resp.content))
    except Exception as e:
        print("❌ 이미지 로드 실패:", str(e))
        return jsonify({"error": f"이미지 로드 실패: {str(e)}", "url": image_url}), 400

    prompt = f"""
너는 숙련된 치과 전문의야. 아래는 AI 탐지 모델이 X-ray 이미지를 분석한 결과야.

- 사용된 탐지 모델: {model1}
- 탐지된 객체 수: {count}

이 정보를 기반으로 환자의 구강 상태를 전문적으로 설명해줘.  
추론은 구체적으로 해석하고, 마지막에 환자에게 해줄 간단한 요약 조언 한 줄을 포함해줘.
"""

    try:
        response = model.generate_content([prompt, img])
        result_text = response.text

        collection.update_one(
            {"_id": ObjectId(inference_result_id)},
            {"$set": {"AI_result": result_text}}
        )

        elapsed_time_ms = int((time.perf_counter() - start_time) * 1000)
        gemini_logger.info(f"[🧠 Gemini X-ray 추론 시간] {elapsed_time_ms}ms (inference_result_id={inference_result_id})")

        print("✅ Gemini X-ray 응답 저장 완료")
        return jsonify({"message": result_text})

    except Exception as e:
        print("❌ Gemini X-ray 호출 실패:", str(e))
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500