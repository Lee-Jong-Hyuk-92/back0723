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
    """
    입력: PIL 이미지(원본)
    출력: (overlay_rgba, box_centers, avg_conf, model_name, main_label, detected_labels)
      - overlay_rgba: 원본과 동일 크기의 '투명 배경 RGBA PNG' 오버레이 (원본 위에 바로 얹어서 사용 가능)
    """
    orig_w, orig_h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # ✅ LetterBox 전처리 (YOLO CLI와 동일)
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # ✅ 추론
    results = model(img_tensor, verbose=False)
    r = results[0]

    # ✅ 탐지 없으면 '완전 투명 PNG' 오버레이 반환
    if r.masks is None or len(r.boxes.cls) == 0:
        transparent = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
        return transparent, [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음", []

    # ✅ 마스크 복원 (letterbox → 원본 크기)
    masks_data = r.masks.data
    if masks_data.ndim == 3:  # [N,H,W] → [N,1,H,W]
        masks_data = masks_data[:, None, :, :]
    r.masks.data = scale_masks(masks_data, (orig_h, orig_w))

    # ✅ 완전 투명 캔버스에만 색을 칠해 오버레이 생성 (원본은 건드리지 않음)
    overlay_rgba = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    for seg, cls_t in zip(r.masks.data.squeeze(1), r.boxes.cls):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))
        mask = (seg.cpu().numpy() * 255).astype(np.uint8)      # 0~255
        mask_img = Image.fromarray(mask, mode="L")             # 알파 마스크는 'L' 모드
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        overlay_rgba.paste(color_layer, (0, 0), mask_img)      # 마스크 영역만 칠함

    # ✅ 클래스/박스 정보 구성
    detected_classes = r.boxes.cls.tolist()
    detected_class_names = [YOLO_CLASS_MAP.get(int(c), "Unknown") for c in detected_classes]

    box_centers = []
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        box_centers.append([cx, cy])

    avg_conf = float(r.boxes.conf.mean().item()) if r.boxes is not None else 0.0
    main_label = detected_class_names[0] if detected_class_names else "감지되지 않음"

    # ✅ RGBA(알파 유지) 그대로 반환
    return overlay_rgba, box_centers, avg_conf, os.path.basename(MODEL_PATH), main_label, detected_class_names
