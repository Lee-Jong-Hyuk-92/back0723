import os
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_masks
import time
import logging

# Set up logging
predictor_logger = logging.getLogger("predictor_logger")
predictor_logger.setLevel(logging.INFO)
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "predictor_times.log")
if not predictor_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    predictor_logger.addHandler(fh)

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
    0: (255, 255, 0, 220),   # 충치 초기      (노랑)
    1: (255, 165, 0, 220),   # 충치 중기      (주황)
    2: (255,   0, 0, 220),   # 충치 말기      (빨강)

    3: (255,   0, 255, 220), # 잇몸 염증 초기 (마젠타)
    4: (165,   0, 255, 220), # 잇몸 염증 중기 (보라빛)
    5: (  0,   0, 255, 220), # 잇몸 염증 말기 (파랑)

    6: (  0, 255, 255, 220), # 치주질환 초기  (시안)
    7: (  0, 255, 165, 220), # 치주질환 중기  (연두빛)
    8: (  0, 255,   0, 220), # 치주질환 말기  (초록)
}

def predict_overlayed_image(pil_img: Image.Image, overlay_save_path: str):
    start_time = time.perf_counter()
    orig_w, orig_h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # ✅ LetterBox 전처리
    lb = LetterBox(new_shape=(640, 640))
    img_lb = lb(image=img_np)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # ✅ 추론
    results = model(img_tensor, verbose=False)
    r = results[0]

    # ✅ 탐지 없으면 투명 PNG 저장 후 반환
    if r.masks is None or len(r.boxes.cls) == 0:
        Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0)).save(overlay_save_path, format="PNG")
        
        elapsed = int((time.perf_counter() - start_time) * 1000)
        predictor_logger.info(f"모델1 추론 (감지 없음): {elapsed}ms")

        # 반환 값 구조 변경
        return pil_img.copy(), [], 0.0, os.path.basename(MODEL_PATH), "감지되지 않음", []

    # ✅ 마스크 복원
    masks_data = r.masks.data
    if masks_data.ndim == 3:
        masks_data = masks_data[:, None, :, :]
    r.masks.data = scale_masks(masks_data, (orig_h, orig_w))

    # ✅ 완전 투명 배경에서 시작
    overlay_img = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    # ✅ 마스크 부분만 색상 합성
    for seg, cls_t in zip(r.masks.data.squeeze(1), r.boxes.cls):
        cls_id = int(cls_t.item())
        color = PALETTE.get(cls_id, (255, 255, 255, 128))
        mask = seg.cpu().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        color_layer = Image.new("RGBA", (orig_w, orig_h), color)
        colored = Image.composite(color_layer, Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0)), mask_img)
        overlay_img = Image.alpha_composite(overlay_img, colored)

    # ✅ overlay만 PNG로 저장
    overlay_img.save(overlay_save_path, format="PNG")

    # ✅ 클래스명/컨피던스/중심점 계산
    detected_results = []
    box_centers = []
    
    # zip을 사용하여 여러 리스트를 동시에 순회
    for cls_t, conf_t, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
        cls_id = int(cls_t.item())
        class_name = YOLO_CLASS_MAP.get(cls_id, "Unknown")
        confidence = float(conf_t.item())
        
        detected_results.append({
            "label": class_name,
            "confidence": confidence
        })

        x1, y1, x2, y2 = box.tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        box_centers.append([cx, cy])

    elapsed = int((time.perf_counter() - start_time) * 1000)
    predictor_logger.info(f"모델1 추론 완료: {elapsed}ms, 감지된 객체 수: {len(detected_results)}")
    
    # ✅ 최종 반환 값 구조 변경
    return (
        overlay_img, 
        box_centers,
        float(r.boxes.conf.mean().item()) if r.boxes is not None else 0.0, # 평균 컨피던스 값
        os.path.basename(MODEL_PATH),
        detected_results[0]['label'] if detected_results else "감지되지 않음",
        detected_results # 변경된 딕셔너리 리스트 반환
    )