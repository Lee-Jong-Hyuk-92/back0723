import os
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_masks

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

# ✅ 색상 팔레트 (클래스별 RGBA)
PALETTE = {
    0: (255, 255, 0, 220),     # 🟡 노랑 (충치 초기)
    1: (255, 165, 0, 220),     # 🟠 주황 (충치 중기)
    2: (255, 0, 0, 220),       # 🔴 빨강 (충치 말기)
    3: (144, 202, 249, 220),   # #90CAF9 (잇몸 염증 초기) 
    4: ( 30, 136, 229, 220),   # #1E88E5 (잇몸 염증 중기) 
    5: ( 13,  71, 161, 220),   # #0D47A1 (잇몸 염증 말기) 
    6: (178, 255, 158, 220),   # #B2FF9E (치주질환 초기) 
    7: (102, 187, 106, 220),   # #66BB6A (치주질환 중기) 
    8: ( 27,  94,  32, 220),   # #1B5E20 (치주질환 말기) 
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

    # ✅ 탐지 없으면 원본 반환
    if r.masks is None or len(r.boxes.cls) == 0:
        return pil_img.copy(), [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음", []

    # ✅ 마스크 복원
    if r.masks is not None:
        masks_data = r.masks.data
        if masks_data.ndim == 3:  # [N, H, W] → [N, 1, H, W]
            masks_data = masks_data[:, None, :, :]
        r.masks.data = scale_masks(masks_data, (orig_h, orig_w))

    # ✅ 원본 RGBA 복사
    overlay_img = pil_img.convert("RGBA")

    # ✅ 각 마스크를 해당 색상으로만 반투명 덮기
    for seg, cls_t in zip(r.masks.data.squeeze(1), r.boxes.cls):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))  # 기본 흰색
        mask = seg.cpu().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        overlay_img = Image.alpha_composite(
            overlay_img,
            Image.composite(color_layer, Image.new("RGBA", (orig_w, orig_h)), mask_img)
        )

    # ✅ 클래스명 / 박스 중심 계산
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
