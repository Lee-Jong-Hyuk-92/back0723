import os
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox   # ✅ 여기서 불러오기
from ultralytics.utils.ops import scale_masks, scale_boxes
from typing import Tuple, List

# ✅ 모델 경로 및 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hygiene0728_best.pt')
model = YOLO(MODEL_PATH)

# ✅ 클래스 ID → 이름 매핑
YOLO_CLASS_MAP = {
    0: "교정장치 (ortho)",
    1: "골드 (gcr)",
    2: "메탈크라운 (mcr)",
    3: "세라믹 (cecr)",
    4: "아말감 (am)",
    5: "지르코니아 (zircr)",
    6: "치석 단계1 (tar1)",
    7: "치석 단계2 (tar2)",
    8: "치석 단계3 (tar3)"    
}

# ✅ 시각화 색상
PALETTE = {
    0: (220, 20, 60, 200),
    1: (138, 43, 226, 200),
    2: (255, 215, 0, 200),
    3: (245, 245, 245, 200),
    4: (30, 30, 30, 200),
    5: (0, 255, 0, 200),
    6: (255, 140, 0, 200),
    7: (0, 0, 255, 200),
    8: (139, 69, 19, 200)
}

def _prepare_image_for_yolo(pil_img: Image.Image, imgsz=640):
    """
    YOLO CLI와 동일한 letterbox 전처리
    """
    img_np = np.array(pil_img.convert("RGB"))
    lb = LetterBox(new_shape=(imgsz, imgsz))
    img_lb = lb(image=img_np)
    return img_lb, lb

def predict_mask_and_overlay_only(pil_img: Image.Image, overlay_save_path: str) -> Image.Image:
    img_np = np.array(pil_img.convert("RGB"))

    # CLI와 동일하게 LetterBox 적용
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    results = model(img_tensor, verbose=False)
    result = results[0]

    # ✅ 탐지 없음 처리
    if result.masks is None or result.boxes is None or len(result.boxes.cls) == 0:
        pil_img.save(overlay_save_path)  # 원본 저장
        return pil_img

    # ✅ 스케일 복원
    masks_data = result.masks.data
    if masks_data.ndim == 3:
        masks_data = masks_data[:, None, :, :]  # [N, 1, H, W]
    result.masks.data = scale_masks(masks_data, img_np.shape[:2])

    # ✅ 오버레이 생성
    overlay_mask = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    for seg, cls in zip(result.masks.data.squeeze(1), result.boxes.cls):
        cls_id = int(cls.item())
        if cls_id not in PALETTE:
            continue
        mask = seg.cpu().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(pil_img.size, Image.NEAREST)
        color = PALETTE.get(cls_id, (0, 0, 0, 128))
        color_layer = Image.new("RGBA", pil_img.size, color)
        mask_rgba = Image.composite(color_layer, Image.new("RGBA", pil_img.size), mask_img)
        overlay_mask = Image.alpha_composite(overlay_mask, mask_rgba)

    overlayed = Image.alpha_composite(pil_img.convert("RGBA"), overlay_mask).convert("RGB")
    overlayed.save(overlay_save_path)
    return overlayed

# ✅ 모든 디텍션 표시
def get_all_classes_and_confidences(pil_img: Image.Image):
    img_lb, lb = _prepare_image_for_yolo(pil_img)
    results = model.predict(img_lb, verbose=False)
    result = results[0]

    classes_detected = []
    for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        label = YOLO_CLASS_MAP.get(int(cls_id), "Unknown")
        classes_detected.append((int(cls_id), float(conf), label))

    return classes_detected

def get_main_class_and_confidence_and_label(pil_img):
    """이미지에서 가장 신뢰도 높은 위생 클래스 1개를 반환"""
    img_np = np.array(pil_img.convert("RGB"))

    # CLI와 동일하게 LetterBox 적용
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    results = model(img_tensor, verbose=False)
    result = results[0]

    if not result.boxes or len(result.boxes.cls) == 0:
        return None, None, None

    # 가장 높은 confidence 선택
    best_idx = result.boxes.conf.argmax().item()
    class_id = int(result.boxes.cls[best_idx].item())
    conf = float(result.boxes.conf[best_idx].item())
    label = YOLO_CLASS_MAP.get(class_id, f"class_{class_id}")

    return class_id, conf, label
