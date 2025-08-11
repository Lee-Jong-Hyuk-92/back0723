import os
from PIL import Image, ImageDraw
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_masks
from typing import Tuple, List

# ✅ 모델 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'disease0728_best.pt')
model = YOLO(MODEL_PATH)

# ✅ 클래스 이름 (YOLO class index 기준)
YOLO_CLASS_MAP = {
    0: "충치 초기",
    1: "충치 중기",
    2: "충치 말기",
    3: "잇몸 염증 초기",
    4: "잇몸 염증 중기",
    5: "잇몸 염증 말기",
    6: "치주질환 초기",
    7: "치주질환 중기",
    8: "치주질환 말기"
}

# ✅ 색상 팔레트 (CLI 결과와 동일)
PALETTE = {
    0: (255, 0, 0, 128),
    1: (0, 255, 0, 128),
    2: (0, 0, 255, 128),
    3: (255, 255, 0, 128),
    4: (255, 0, 255, 128),
    5: (0, 255, 255, 128),
    6: (255, 165, 0, 128),
    7: (128, 0, 128, 128),
    8: (128, 128, 128, 128),
}

def predict_overlayed_image(pil_img: Image.Image):
    orig_w, orig_h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # ✅ CLI와 동일한 LetterBox 전처리
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # ✅ 추론
    results = model(img_tensor, verbose=False)
    r = results[0]

    # ✅ 마스크 복원
    if r.masks is not None:
        masks_data = r.masks.data
        if masks_data.ndim == 3:  # [N, H, W] → [N, 1, H, W]
            masks_data = masks_data[:, None, :, :]
        r.masks.data = scale_masks(masks_data, (orig_h, orig_w))

    # ✅ 탐지 없으면 원본 반환
    if r.masks is None or len(r.boxes.cls) == 0:
        return pil_img.copy(), [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음", []

    # ✅ 마스크 그리기
    overlay_img = pil_img.convert("RGBA")
    for seg, cls_t in zip(r.masks.data.squeeze(1), r.boxes.cls):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))
        mask = seg.cpu().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        overlay_img = Image.alpha_composite(overlay_img, Image.composite(color_layer, Image.new("RGBA", (orig_w, orig_h)), mask_img))

    # ✅ 클래스명 / 박스 중심
    detected_classes = r.boxes.cls.tolist()
    detected_class_names = [YOLO_CLASS_MAP.get(int(c), "Unknown") for c in detected_classes]

    box_centers = []
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        box_centers.append([cx, cy])

    return (
        overlay_img.convert("RGB"),
        box_centers,
        float(r.boxes.conf.mean().item()) if r.boxes is not None else 0.0,
        os.path.basename(MODEL_PATH),
        detected_class_names[0] if detected_class_names else "감지되지 않음",
        detected_class_names
    )
