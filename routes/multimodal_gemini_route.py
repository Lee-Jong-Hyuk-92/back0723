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
    model1_label = data.get("model1Label") or "감지되지 않음"
    conf1 = float(data.get("model1Confidence") or 0.0)
    
    # ✅ 수정된 부분: model2Labels 리스트를 받습니다.
    model2_labels = data.get("model2Labels", [])
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

    print("🔍 기존 AI_result 없음 -> Gemini 호출 시작")

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

    # ✅ 수정된 부분: 위생 라벨 리스트를 콤마로 연결합니다.
    hygiene_summary = "감지되지 않음"
    if model2_labels and len(model2_labels) > 0:
        hygiene_summary = ", ".join(model2_labels)

    prompt = f"""
너는 치과 전문의로서, 환자의 문진 정보와 AI 분석 결과를 바탕으로 7줄 내외로 간결하게 환자에게 전달할 소견을 작성해줘. 노인들도 이해하기 쉬운 용어를 사용해. 마지막에는 1줄 결론도 같이.

[문진 정보]
{survey_lines}

[AI 분석 요약]
- 질병 예측: {model1_label}, 확신도: {conf1:.2f}
- 위생 예측: {hygiene_summary}, 확신도: {conf2:.2f}
- 치아 번호: {tooth_number}, 확신도: {conf3:.2f}
"""
    # ✅ 추가: 생성된 프롬프트를 로거로 출력
    gemini_logger.info(f"Generated Prompt for Gemini: \n{prompt}")

    # ✅ 4. Gemini 요청
    try:
        response = model.generate_content([prompt, img])
        result_text = response.text

        # ✅ 면책 조항을 AI 응답에 추가
        disclaimer = "\n\n주의: 이 AI 소견은 참고용이며, 실제 치과 진료를 대체할 수 없습니다. 최종 진단 및 치료 계획은 반드시 치과 의사와 상담하시기 바랍니다."
        final_result = result_text.strip() + disclaimer

        # ✅ 결과 저장
        collection.update_one(
            {"_id": ObjectId(inference_result_id)},
            {"$set": {"AI_result": final_result}}
        )

        elapsed_time_ms = int((time.perf_counter() - start_time) * 1000)
        gemini_logger.info(f"[🧠 Gemini 멀티모달 추론 시간] {elapsed_time_ms}ms (inference_result_id={inference_result_id})")

        print("✅ Gemini 응답 저장 완료")
        return jsonify({"message": final_result})

    except Exception as e:
        print("❌ Gemini 호출 실패:", str(e))
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500