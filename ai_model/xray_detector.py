from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch

# ✅ 모델 로드
model_path = 'ai_model/xray_detect_best.pt'
model = YOLO(model_path)

# 클래스 ID와 이름 매핑
CLASS_NAMES = [
    '치아 우식증', '임플란트', '보철물',
    '근관치료', '정상치아', '상실치아'
]

# 클래스별 색상 정의 (RGB)
CLASS_COLORS = {
    '치아 우식증': (255, 0, 0),      # 빨강
    '임플란트': (0, 0, 255),       # 파랑
    '보철물': (255, 255, 0),     # 노랑
    '근관치료': (0, 255, 0),  # ✅ 초록색으로 변경
    '상실치아': (0, 0, 0)       # ✅ 검은색으로 변경
}

def detect_xray(image_path: str):
    """
    X-ray 이미지를 받아 YOLOv11x 모델로 탐지 수행하고, 결과를 이미지에 그립니다.
    이 버전에서는 박스만 표시하고 라벨과 신뢰도는 표시하지 않습니다.
    """
    # YOLO 모델 추론. 결과는 결과 객체의 리스트를 반환함.
    results = model(image_path, conf=0.3)
    
    # 첫 번째 이미지의 결과 객체에서 boxes 속성을 가져옵니다.
    boxes = results[0].boxes

    predictions = []

    # 이미지 로드 및 그리기 객체 생성
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    line_thickness = 4

    for box in boxes:
        cls_id = int(box.cls.item())
        
        # CLASS_NAMES는 리스트이므로 인덱스로 접근합니다.
        class_name = CLASS_NAMES[cls_id]

        if class_name == '정상치아':
            continue

        confidence = float(box.conf.item())
        
        # CPU로 옮겨서 numpy 배열로 변환 후 리스트로 변환
        coords = box.xyxy.cpu().numpy()[0].tolist()
        x1, y1, x2, y2 = map(int, coords)

        # ✅ 박스만 그리기 (라벨 및 컨피던스 그리는 코드 제거)
        # 폰트 관련 설정 및 라벨 텍스트/배경 그리는 코드가 모두 제거되었습니다.
        color = CLASS_COLORS.get(class_name, (255, 255, 255)) # 클래스에 맞는 색상, 없으면 흰색
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)

        predictions.append({
            'class_id': cls_id,
            'class_name': class_name,
            'confidence': confidence,
            'bbox': coords
        })

    # 변경된 이미지를 새로운 파일로 저장
    output_path = image_path.replace(".png", "_detected.png").replace(".jpg", "_detected.jpg")
    image.save(output_path)

    return {
        'image_path': output_path, # 저장된 새 이미지 경로 반환
        'detections': predictions,
        'model': model_path
    }