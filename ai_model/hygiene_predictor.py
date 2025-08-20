import os
from typing import Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_masks

# ── 모델 경로 및 로드 ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hygiene0728_best.pt')
model = YOLO(MODEL_PATH)

# ── 클래스 ID → 이름 매핑 ──────────────────────────────────────────────────────
YOLO_CLASS_MAP: Dict[int, str] = {
    0: "교정장치",
    1: "금니 (골드 크라운)",
    2: "은니 (메탈 크라운)",
    3: "도자기소재 치아 덮개(세라믹 크라운)",
    4: "아말감 충전재",
    5: "도자기소재 치아 덮개(지르코니아 크라운)",
    6: "치석 1 단계",
    7: "치석 2 단계",
    8: "치석 3 단계",
}

# ── 시각화 색상 (RGBA) ─────────────────────────────────────────────────────────
PALETTE: Dict[int, tuple] = {
    0: (138, 43, 226, 200),   # 교정장치: 보라
    1: (255, 215, 0, 200),    # 금니 (골드 크라운): 골드
    2: (192, 192, 192, 200),  # 은니 (메탈 크라운): 실버/회색
    3: (0, 0, 0, 200),        # 세라믹 크라운: 검정
    4: (0, 0, 255, 200),      # 아말감 충전재: 파랑
    5: (0, 255, 0, 200),      # 지르코니아 크라운: 초록
    6: (255, 255, 0, 200),    # 치석 1 단계: 노랑
    7: (255, 165, 0, 200),    # 치석 2 단계: 주황
    8: (255, 0, 0, 200),      # 치석 3 단계: 빨강
}

# ── 유틸: LetterBox 전처리 ─────────────────────────────────────────────────────
def _prepare_image_for_yolo(pil_img: Image.Image, imgsz: int = 640):
    """YOLO CLI와 동일한 letterbox 전처리."""
    img_np = np.array(pil_img.convert("RGB"))
    lb = LetterBox(new_shape=(imgsz, imgsz))
    img_lb = lb(image=img_np)
    return img_lb, lb

def predict_mask_and_overlay_with_all(
    pil_img: Image.Image,
    overlay_save_path: str
) -> Tuple[
    Image.Image,
    List[Dict],
    float,
    str,
    str,
]:
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
        empty = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
        empty.save(overlay_save_path, format="PNG")
        return empty, [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음"

    # 마스크 크기 원본으로 복원
    masks_data = r.masks.data
    if masks_data.ndim == 3:
        masks_data = masks_data[:, None, :, :]
    elif masks_data.ndim == 4 and masks_data.shape[1] != 1:
        masks_data = masks_data[:, :1, :, :]

    r.masks.data = scale_masks(masks_data, (orig_h, orig_w))
    masks_scaled = r.masks.data.squeeze(1)

    # 투명 배경에서 시작
    overlay_img = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    detections: List[Dict] = []
    for seg, cls_t, conf_t, box in zip(masks_scaled, r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))

        # 마스크 합성
        mask = seg.cpu().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        colored = Image.composite(color_layer, Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0)), mask_img)
        overlay_img = Image.alpha_composite(overlay_img, colored)

        # 디텍션 기록
        x1, y1, x2, y2 = map(float, box.tolist())
        detections.append({
            "class_id": cls_id,
            "label": YOLO_CLASS_MAP.get(cls_id, "Unknown"),
            "confidence": float(conf_t.item()),
            "bbox": [x1, y1, x2, y2],
            "mask_array": mask, # 마스크 데이터 추가
        })

    # 투명 PNG 저장
    overlay_img.save(overlay_save_path, format="PNG")

    # 평균 confidence
    avg_confidence = float(r.boxes.conf.mean().item()) if r.boxes.conf is not None else 0.0

    # 대표 클래스명 (첫번째)
    main_label = detections[0]['label'] if detections else "감지되지 않음"

    return overlay_img, detections, avg_confidence, os.path.basename(MODEL_PATH), main_label