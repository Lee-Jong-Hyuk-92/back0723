# xray_detect_best.pt 모델을 이용한 추론코드, 아직 반영전
from ultralytics import YOLO
from PIL import Image
import torch

# ✅ 모델 로드
model_path = 'ai_model/xray_detect_best.pt'
model = YOLO(model_path)

# 클래스 ID와 이름 매핑
CLASS_NAMES = [
    '치아 우식증', '임플란트', '보철물',
    '근관치료', '정상치아', '상실치아'
]

def detect_xray(image_path: str):
    """
    X-ray 이미지를 받아 YOLOv11x 모델로 탐지 수행
    """
    results = model(image_path, conf=0.3)
    boxes = results[0].boxes  # 첫 번째 이미지 결과만 사용

    predictions = []

    for box in boxes:
        cls_id = int(box.cls.item())
        class_name = CLASS_NAMES[cls_id]
        
        # ✅ 정상치아는 제외
        if class_name == '정상치아':
            continue

        confidence = float(box.conf.item())
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        predictions.append({
            'class_id': cls_id,
            'class_name': class_name,
            'confidence': confidence,
            'bbox': coords
        })

    return {
        'image_path': image_path,
        'detections': predictions,
        'model': model_path
    }