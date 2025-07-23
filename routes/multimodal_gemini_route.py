from flask import Blueprint, request, jsonify
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import os
import time

multimodal_gemini_bp = Blueprint('multimodal_gemini', __name__)

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

@multimodal_gemini_bp.route("/api/multimodal_gemini", methods=["POST"])
def chat_gemini():
    print("✅ [요청 수신] /api/multimodal_gemini")

    data = request.get_json()
    print(f"📦 받은 데이터: {data}")

    image_url = data.get("image_url")
    model1 = data.get("model1Label")
    conf1 = data.get("model1Confidence")
    model2 = data.get("model2Label")
    conf2 = data.get("model2Confidence")
    tooth_number = data.get("model3ToothNumber")
    conf3 = data.get("model3Confidence")

    try:
        print(f"🌐 이미지 다운로드 중: {image_url}")
        img_resp = requests.get(image_url)
        img = Image.open(BytesIO(img_resp.content))
        print("✅ 이미지 로드 완료")
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return jsonify({"error": f"이미지 로드 실패: {str(e)}"}), 400

    prompt = f"""
너는 치과 전문의야. 아래는 AI가 구강 이미지를 분석한 결과야. 이 정보를 바탕으로 환자의 상태를 상세히 설명해줘.

- 질병 예측: {model1}, 확신도: {conf1:.2f}
- 위생 예측: {model2}, 확신도: {conf2:.2f}
- 치아 번호: {tooth_number}, 확신도: {conf3:.2f}

해당 이미지와 결과를 함께 고려해 설명해줘. 마지막엔 결론 한 줄로 요약해.
"""
    print("🧠 Gemini 요청 전송 시작...")

    try:
        start_time = time.time()
        response = model.generate_content([prompt, img])
        elapsed_time = time.time() - start_time
        print("✅ Gemini 응답 수신 완료")
        print(f"🕒 응답 시간: {elapsed_time:.2f}초")
        print(f"📝 응답 내용:\n{response.text}")
        return jsonify({"message": response.text})
    except Exception as e:
        print(f"❌ Gemini 호출 실패: {e}")
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500