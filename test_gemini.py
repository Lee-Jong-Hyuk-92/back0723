import google.generativeai as genai
from PIL import Image
import os
import time # time 모듈 임포트

# --- 1. Gemini API 키 설정 ---
API_KEY = "AIzaSyCHl0KRXd7oE4qzz0AbIzhGm_Ia2CpVNB0"
genai.configure(api_key=API_KEY)

# --- 2. 사용할 Gemini 모델 선택 ---
MODEL_NAME = "gemini-1.5-flash-latest"
model = genai.GenerativeModel(MODEL_NAME)

# --- 3. 분석할 이미지 파일 경로 지정 ---
IMAGE_PATH = r"C:\Users\302-1\Desktop\backend0709-1\images\original\121212_20250722160653484634_web_image.png"

# --- 4. 이미지 파일 로드 ---
try:
    img = Image.open(IMAGE_PATH)
    print(f"이미지 '{IMAGE_PATH}'를 성공적으로 로드했습니다.")
except FileNotFoundError:
    print(f"오류: '{IMAGE_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"이미지 로드 중 오류 발생: {e}")
    exit()

# --- 5. Gemini API 호출 ---
prompt = """
너는 치과 전문의야. 이 사진에 대해 자세히 설명해줘. 특히 어떤 상황인지, 주요 특징은 무엇인지 알려줘. 마지막에 한줄 결론도 적어줘.

_id
687f38913c7dfd2c80537995
user_id
"121212"
original_image_path
"/images/original/121212_20250722160653484634_web_image.png"

original_image_yolo_detections
Array (empty)
model1_image_path
"/images/model1/121212_20250722160653484634_web_image.png"

model1_inference_result
Object
message
"model1 마스크 생성 완료"

lesion_points
Array (613)
confidence
0.7220852375030518
used_model
"disease_model_saved_weight.pt"
label
"충치 초기"
model2_image_path
"/images/model2/121212_20250722160653484634_web_image.png"

model2_inference_result
Object
message
"model2 마스크 생성 완료"
class_id
-1
confidence
0
label
"Unknown"
model3_image_path
"/images/model3/121212_20250722160653484634_web_image.png"

model3_inference_result
Object
message
"model3 마스크 생성 완료"
class_id
31
confidence
0.6953080296516418
tooth_number_fdi
47
timestamp
2025-07-22T16:06:57.897+00:00
"""

print(f"\n[{MODEL_NAME}] 모델로 요청을 보냅니다...")

start_time = time.time() # 요청 보내기 직전 시간 기록

try:
    response = model.generate_content([prompt, img])

    end_time = time.time() # 응답 받은 직후 시간 기록
    elapsed_time = end_time - start_time # 소요 시간 계산

    # --- 6. 응답 출력 ---
    print("\n--- Gemini 모델의 응답 ---")
    print(response.text)
    print(f"\n응답 소요 시간: {elapsed_time:.2f}초") # 소요 시간 출력 (소수점 둘째 자리까지)

except genai.types.BlockedPromptException as e:
    print(f"오류: 프롬프트가 차단되었습니다. 사유: {e.response.prompt_feedback.block_reason}")
    print(f"안전 설정 확인: {e.response.prompt_feedback.safety_ratings}")
except Exception as e:
    print(f"API 호출 중 오류 발생: {e}")