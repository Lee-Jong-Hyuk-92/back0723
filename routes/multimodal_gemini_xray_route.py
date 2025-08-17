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

    # 값이 없을 수도 있으니 기본값 처리
    model1 = data.get("model1Label") or "unknown"
    try:
        conf1 = float(data.get("model1Confidence") or 0.0)
    except Exception:
        conf1 = 0.0
    try:
        count = int(data.get("predictionCount") or 0)
    except Exception:
        count = 0

    # 📄 분석 결과 문서 조회
    try:
        doc = collection.find_one({"_id": ObjectId(inference_result_id)})
    except Exception:
        doc = None

    if not doc:
        return jsonify({"error": "해당 분석 결과를 찾을 수 없습니다."}), 404

    # ✅ 문진(survey) 데이터 가져오기
    survey_data = doc.get("survey", {}) or {}

    # 이미 생성된 결과가 있으면 캐시 반환
    if "AI_result" in doc:
        print("📄 기존 X-ray AI_result 반환")
        return jsonify({"message": doc["AI_result"]})

    print("🔍 Gemini X-ray 멀티모달 분석 시작")

    # ✅ 이미지 로드
    try:
        # 내부/자가 서명 인증서 환경 고려해 verify=False 유지
        img_resp = requests.get(image_url, verify=False)
        img_resp.raise_for_status()
        img = Image.open(BytesIO(img_resp.content))
    except Exception as e:
        print("❌ 이미지 로드 실패:", str(e))
        return jsonify({"error": f"이미지 로드 실패: {str(e)}", "url": image_url}), 400

    # ✅ 문진을 깔끔하게 bullet로 만들기
    if isinstance(survey_data, dict) and survey_data:
        try:
            # 키 순서를 고정하고 싶으면 sorted(survey_data.items())
            survey_lines = "\n".join([f"- {k}: {v}" for k, v in survey_data.items()])
        except Exception:
            survey_lines = str(survey_data)
    else:
        survey_lines = "제공되지 않음"

    # (선택) DB에 저장된 X-ray 탐지/분류 결과 요약 추가 정보
    xray_pred = doc.get("model1_inference_result", {})
    xray_summary = xray_pred.get("summary", "")
    implant_results = doc.get("implant_classification_result", []) or []
    implant_count = len(implant_results)

    # ✅ 프롬프트 구성 (문진 + X-ray 요약 포함)
    prompt = f"""
너는 숙련된 치과 전문의야. 아래는 환자의 문진 정보와 X-ray 분석 결과 요약이야.
원본 X-ray 픽셀 정보를 최우선으로 평가하되, 텍스트와 추가 정보는 참고용으로만 사용해.
충분한 근거 없이 과잉진단하지 말고, 마지막에는 환자가 이해할 수 있는 한 줄 조언을 붙여줘.

[문진 정보]
{survey_lines}

[X-ray 분석 요약(모델 출력)]
- 사용된 탐지 모델: {model1}
- 탐지된 객체 수: {count}
- 탐지 모델 확신도(평균/대표): {conf1:.2f}
- 내부 요약: {xray_summary if xray_summary else "요약 없음"}
- 임플란트 분류 결과 개수: {implant_count}
"""

    try:
        response = model.generate_content([prompt, img])
        result_text = response.text

        # ✅ 결과 저장 (캐시)
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