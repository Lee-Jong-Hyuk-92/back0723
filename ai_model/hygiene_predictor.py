import os
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_masks
from typing import Tuple, List

# 모델 경로 및 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hygiene0728_best.pt')
model = YOLO(MODEL_PATH)

# 클래스 ID → 이름 매핑
YOLO_CLASS_MAP = {
    0: "교정장치",
    1: "금니 (골드 크라운)",
    2: "은니 (메탈 크라운)",
    3: "도자기소재 치아 덮개(세라믹 크라운)",
    4: "아말감 충전재",
    5: "도자기소재 치아 덮개(지르코니아 크라운)",
    6: "치석 1 단계",
    7: "치석 2 단계",
    8: "치석 3 단계"
}

# 시각화 색상 (RGBA)
PALETTE = {
    0: (138,  43, 226, 200),  # 교정장치: 보라
    1: (192, 192, 192, 200),  # 금니 (골드 크라운): 실버/회색
    2: (255, 215,   0, 200),  # 은니 (메탈 크라운): 골드
    3: (  0,   0,   0, 200),  # 도자기소재 치아 덮개(세라믹 크라운): 검정
    4: (  0,   0, 255, 200),  # 아말감 충전재: 파랑
    5: (  0, 255,   0, 200),  # 도자기소재 치아 덮개(지르코니아 크라운): 초록
    6: (255, 255,   0, 200),  # 치석 1 단계: 노랑
    7: (255, 165,   0, 200),  # 치석 2 단계: 주황
    8: (255,   0,   0, 200),  # 치석 3 단계: 빨강
}

def _prepare_image_for_yolo(pil_img: Image.Image, imgsz=640):
    """
    YOLO CLI와 동일한 letterbox 전처리
    """
    img_np = np.array(pil_img.convert("RGB"))
    lb = LetterBox(new_shape=(imgsz, imgsz))
    img_lb = lb(image=img_np)
    return img_lb, lb

def predict_mask_and_overlay_with_all(pil_img: Image.Image, overlay_save_path: str) -> Tuple[Image.Image, List[List[int]], float, str, str, List[str]]:
    """
    마스크 오버레이 이미지를 생성 및 저장하고,
    탐지된 모든 클래스 이름 배열과 기타 정보 반환.
    → 수정: 투명 배경 + 마스크만 저장
    """
    orig_w, orig_h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # LetterBox 전처리
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # 추론
    results = model(img_tensor, verbose=False)
    r = results[0]

    # 탐지 없으면 완전 투명 PNG 저장
    if r.masks is None or len(r.boxes.cls) == 0:
        Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0)).save(overlay_save_path, format="PNG")
        return pil_img, [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음", []

    # 마스크 크기 원본으로 복원
    masks_data = r.masks.data
    if masks_data.ndim == 3:
        masks_data = masks_data[:, None, :, :]
    r.masks.data = scale_masks(masks_data, (orig_h, orig_w))

    # 투명 배경에서 시작
    overlay_img = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    # 마스크별 색상 적용
    for seg, cls_t in zip(r.masks.data.squeeze(1), r.boxes.cls):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))
        mask = seg.cpu().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        colored = Image.composite(color_layer, Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0)), mask_img)
        overlay_img = Image.alpha_composite(overlay_img, colored)

    # 투명 PNG로 저장
    overlay_img.save(overlay_save_path, format="PNG")

    # 클래스명 리스트
    detected_classes = r.boxes.cls.tolist()
    detected_class_names = [YOLO_CLASS_MAP.get(int(c), "Unknown") for c in detected_classes]

    # 박스 중심 좌표
    box_centers = []
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        box_centers.append([cx, cy])

    # 평균 confidence
    avg_confidence = float(r.boxes.conf.mean().item()) if r.boxes.conf is not None else 0.0

    # 대표 클래스명 (첫번째)
    main_label = detected_class_names[0] if detected_class_names else "감지되지 않음"

    return overlay_img, box_centers, avg_confidence, os.path.basename(MODEL_PATH), main_label, detected_class_names

# 모든 디텍션과 confidence 리스트 반환
def get_all_classes_and_confidences(pil_img: Image.Image):
    img_lb, lb = _prepare_image_for_yolo(pil_img)
    results = model.predict(img_lb, verbose=False)
    r = results[0]

    classes_detected = []
    for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        label = YOLO_CLASS_MAP.get(int(cls_id), "Unknown")
        classes_detected.append((int(cls_id), float(conf), label))

    return classes_detected

# 가장 신뢰도 높은 클래스 1개 반환
def get_main_class_and_confidence_and_label(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    results = model(img_tensor, verbose=False)
    r = results[0]

    if not r.boxes or len(r.boxes.cls) == 0:
        return None, None, None

    best_idx = r.boxes.conf.argmax().item()
    class_id = int(r.boxes.cls[best_idx].item())
    conf = float(r.boxes.conf[best_idx].item())
    label = YOLO_CLASS_MAP.get(class_id, f"class_{class_id}")

    return class_id, conf, label
