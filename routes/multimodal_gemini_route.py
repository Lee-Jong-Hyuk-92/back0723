from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import os
import time
import logging

# ✅ Gemini 전용 로거 설정
gemini_logger = logging.getLogger("gemini_logger")
gemini_logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "gemini_times.log")

if not gemini_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    gemini_logger.addHandler(fh)

multimodal_gemini_bp = Blueprint('multimodal_gemini', __name__)

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")


@multimodal_gemini_bp.route("/multimodal_gemini", methods=["POST"])
def handle_ai_opinion():
    start_time = time.perf_counter()

    data = request.get_json()

    mongo_client = current_app.extensions.get("mongo_client")
    collection = mongo_client.get_collection("inference_results")

    image_url = data.get("image_url")
    inference_result_id = data.get("inference_result_id")

    # 값이 없을 수도 있으니 안전하게 기본값 처리
    model1 = data.get("model1Label") or "감지되지 않음"
    conf1 = float(data.get("model1Confidence") or 0.0)
    model2 = data.get("model2Label") or "감지되지 않음"
    conf2 = float(data.get("model2Confidence") or 0.0)
    tooth_number = data.get("model3ToothNumber") or "Unknown"
    conf3 = float(data.get("model3Confidence") or 0.0)

    doc = collection.find_one({"_id": ObjectId(inference_result_id)})
    if not doc:
        return jsonify({"error": "해당 분석 결과를 찾을 수 없습니다."}), 404

    # ✅ 문진 데이터 가져오기
    survey_data = doc.get("survey", {}) or {}

    # 이미 생성된 응답이 있으면 캐시 반환
    if "AI_result" in doc:
        print("📄 기존 AI_result 반환")
        return jsonify({"message": doc["AI_result"]})

    print("🔍 기존 AI_result 없음 → Gemini 호출 시작")

    # ✅ 2. 이미지 불러오기
    try:
        # NOTE: 내부 보호 리소스라면 인증이 필요할 수 있습니다.
        img_resp = requests.get(image_url, verify=False)
        img_resp.raise_for_status()
        img = Image.open(BytesIO(img_resp.content))
    except Exception as e:
        print("❌ 이미지 요청 실패:", str(e))
        return jsonify({"error": f"이미지 로드 실패: {str(e)}", "url": image_url}), 400

    # ✅ 3. 프롬프트 구성 (문진 포함)
    # 문진을 깔끔한 bullet 형태로 변환
    if isinstance(survey_data, dict) and survey_data:
        survey_lines = "\n".join([f"- {k}: {v}" for k, v in survey_data.items()])
    else:
        survey_lines = "제공되지 않음"

    prompt = f"""
너는 치과 전문의야. 아래는 환자의 문진 정보와, AI가 구강 이미지를 분석한 결과야.
이 모든 정보를 바탕으로 환자의 상태를 상세히 설명해줘. 마지막엔 결론 한 줄로 요약해.

[문진 정보]
{survey_lines}

[AI 분석 요약]
- 질병 예측: {model1}, 확신도: {conf1:.2f}
- 위생 예측: {model2}, 확신도: {conf2:.2f}
- 치아 번호: {tooth_number}, 확신도: {conf3:.2f}
"""

    # ✅ 4. Gemini 요청
    try:
        response = model.generate_content([prompt, img])
        result_text = response.text

        # ✅ 결과 저장
        collection.update_one(
            {"_id": ObjectId(inference_result_id)},
            {"$set": {"AI_result": result_text}}
        )

        elapsed_time_ms = int((time.perf_counter() - start_time) * 1000)
        gemini_logger.info(f"[🧠 Gemini 멀티모달 추론 시간] {elapsed_time_ms}ms (inference_result_id={inference_result_id})")

        print("✅ Gemini 응답 저장 완료")
        return jsonify({"message": result_text})

    except Exception as e:
        print("❌ Gemini 호출 실패:", str(e))
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500
