import os
import vertexai
from google.cloud import aiplatform
from PIL import Image
import io
from google.cloud import storage
import time # time 모듈 임포트

# ✅ 환경변수로 GCP 인증 키 등록
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"meditooth-7ce9efd0794b.json"

# ✅ GCP 프로젝트 설정
PROJECT_ID = "meditooth"
LOCATION = "us-central1"

# ✅ Vertex AI 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ✅ MedGemma Endpoint ID
MEDGEMMA_ENDPOINT_ID = "7198930337072676864"

# ✅ 엔드포인트 연결
try:
    medgemma_endpoint = aiplatform.Endpoint(
        endpoint_name=MEDGEMMA_ENDPOINT_ID,
        project=PROJECT_ID,
        location=LOCATION,
    )
    print(f"✅ Connected to MedGemma Endpoint: {medgemma_endpoint.display_name}")
except Exception as e:
    print(f"❌ Endpoint connection failed: {e}")
    exit()

# ✅ GCS 업로드 함수
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # ⭐ RGBA → RGB 변환
    with Image.open(source_file_name) as img:
        img = img.resize((896, 896), Image.BICUBIC)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    blob.make_public()
    return blob.public_url

# ✅ 업로드할 이미지 및 GCS 설정
GCS_BUCKET_NAME = "meditooth-medgemma-images-temp"
GCS_IMAGE_DESTINATION_PATH = "oral_image_896x896.jpeg"
local_image_path = r"C:\Users\302-1\Desktop\back0723\images\model1\121212_20250724153839479471_web_image.png"

# ✅ 이미지 업로드
print(f"📤 Uploading image to GCS: {GCS_BUCKET_NAME}/{GCS_IMAGE_DESTINATION_PATH}")
try:
    gcs_image_url = upload_blob(GCS_BUCKET_NAME, local_image_path, GCS_IMAGE_DESTINATION_PATH)
    print(f"📎 Image URL: {gcs_image_url}")
except Exception as e:
    print(f"❌ GCS Upload Error: {e}")
    exit()

# ✅ 프롬프트 정의
system_instruction = "너는 치과 전문의야. 이 사진에 대해 자세히 설명해줘. 특히 어떤 상황인지, 주요 특징은 무엇인지 알려줘. 마지막에 한줄 결론도 적어줘."
user_prompt = """

_id
6881d4f35954b846b686a7c8
user_id
"121212"
original_image_path
"/images/original/121212_20250724153839479471_web_image.png"

original_image_yolo_detections
Array (empty)
model1_image_path
"/images/model1/121212_20250724153839479471_web_image.png"

model1_inference_result
Object
message
"model1 마스크 생성 완료"

lesion_points
Array (1600)
confidence
0.8067517280578613
used_model
"disease_model_saved_weight.pt"
label
"잇몸 염증 초기"
model2_image_path
"/images/model2/121212_20250724153839479471_web_image.png"

model2_inference_result
Object
message
"model2 마스크 생성 완료"
class_id
7
confidence
0.5540841817855835
label
"치석 단계2 (tar2)"
model3_image_path
"/images/model3/121212_20250724153839479471_web_image.png"

model3_inference_result
Object
message
"model3 마스크 생성 완료"
class_id
28
confidence
0.5090081691741943
tooth_number_fdi
44
timestamp
2025-07-24T15:38:43.229+00:00

"""

# ✅ Vertex AI 메시지 형식 구성
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            #{"type": "image_url", "image_url": {"url": gcs_image_url}}
        ]
    }
]

# ✅ 요청 인스턴스 구성
instances = [
    {
        "@requestFormat": "chatCompletions",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.4
    },
]

# ✅ MedGemma 추론 요청
print("\n🔮 Generating content from MedGemma...")
start_time = time.time() # 추론 요청 보내기 직전 시간 기록

try:
    result = medgemma_endpoint.predict(instances=instances)
    end_time = time.time() # 응답 받은 직후 시간 기록
    elapsed_time = end_time - start_time # 소요 시간 계산

    response = result.predictions["choices"][0]["message"]["content"]
    print("\n🦷 분석 결과:")
    print(response)
    print(f"\nMedGemma 추론 소요 시간: {elapsed_time:.2f}초") # 소요 시간 출력 (소수점 둘째 자리까지)

except Exception as e:
    print(f"❌ Prediction Error: {e}")