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

# 시각화 색상 (RGBA)
PALETTE = {
    0: (30, 30, 30, 200),      # 0: 교정장치 (ortho) → 다크 그레이
    1: (255, 215, 0, 200),     # 1: 골드 (gcr) → 골드
    2: (169, 169, 169, 200),   # 2: 메탈크라운 (mcr) → 메탈
    3: (245, 245, 245, 200),   # 3: 세라믹 (cecr) → 화이트 스모크
    4: (192, 192, 192, 200),   # 4: 아말감 (am) → 실버
    5: (220, 20, 60, 200),     # 5: 지르코니아 (zircr) → 크림슨 레드
    6: (255, 255, 153, 200),   # 6: 치석 단계1 (tar1) → 연한 노랑
    7: (255, 204, 0, 200),     # 7: 치석 단계2 (tar2) → 진한 노랑/오렌지
    8: (204, 153, 0, 200),     # 8: 치석 단계3 (tar3) → 황갈색
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
    마스크 오버레이 이미지를 '투명 배경 RGBA'로 생성/저장하고,
    탐지된 모든 클래스 이름 배열과 기타 정보 반환.
    overlay_save_path는 PNG로 저장됨(확장자 강제 .png)
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

    # 저장 경로 확장자 PNG 강제
    root, _ext = os.path.splitext(overlay_save_path)
    overlay_save_path = root + ".png"

    # 탐지 없으면 '완전 투명 PNG' 저장 후 반환
    if r.masks is None or len(r.boxes.cls) == 0:
        transparent = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
        transparent.save(overlay_save_path, "PNG")
        return transparent, [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음", []

    # 마스크 크기 원본으로 복원
    masks_data = r.masks.data
    if masks_data.ndim == 3:
        masks_data = masks_data[:, None, :, :]
    r.masks.data = scale_masks(masks_data, (orig_h, orig_w))

    # ✅ 완전 투명 캔버스(원본 위에 직접 칠하지 않음)
    overlay_rgba = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    for seg, cls_t in zip(r.masks.data.squeeze(1), r.boxes.cls):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))
        mask = (seg.cpu().numpy() * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask, mode="L")
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        overlay_rgba.paste(color_layer, (0, 0), mask_img)

    # 클래스명 리스트
    detected_classes = r.boxes.cls.tolist()
    detected_class_names = [YOLO_CLASS_MAP.get(int(c), "Unknown") for c in detected_classes]

    # 박스 중심 좌표
    box_centers = []
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        box_centers.append([cx, cy])

    # 평균 confidence / 대표 클래스명
    avg_confidence = float(r.boxes.conf.mean().item()) if r.boxes.conf is not None else 0.0
    main_label = detected_class_names[0] if detected_class_names else "감지되지 않음"

    # ✅ 반드시 PNG(RGBA)로 저장
    overlay_rgba.save(overlay_save_path, "PNG")

    # (선택) 미리보기용 합성 이미지를 따로 저장하고 싶다면:
    # preview = Image.alpha_composite(pil_img.convert("RGBA"), overlay_rgba)
    # preview.save(root + "_preview.png", "PNG")

    return overlay_rgba, box_centers, avg_confidence, os.path.basename(MODEL_PATH), main_label, detected_class_names

# 모든 디텍션과 confidence 리스트 반환
def get_all_classes_and_confidences(pil_img: Image.Image):
    img_lb, lb = _prepare_image_for_yolo(pil_img)
    results = model.predict(img_lb, verbose=False)
    r = results[0]

    classes_detected = []
    if r.boxes is not None and len(r.boxes) > 0:
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
