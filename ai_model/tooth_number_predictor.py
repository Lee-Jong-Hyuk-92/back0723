import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ✅ 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai_model", "number_x_250806.pt")
model = YOLO(MODEL_PATH)

# ✅ FDI 치아 번호 매핑
FDI_CLASS_MAP = {
    0: '11', 1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18',
    8: '21', 9: '22', 10: '23', 11: '24', 12: '25', 13: '26', 14: '27', 15: '28',
    16: '31', 17: '32', 18: '33', 19: '34', 20: '35', 21: '36', 22: '37', 23: '38',
    24: '41', 25: '42', 26: '43', 27: '44', 28: '45', 29: '46', 30: '47', 31: '48'
}

# ✅ 예측 + 오버레이 이미지 저장
def predict_mask_and_overlay_only(pil_img, overlay_save_path):
    rgb_img = np.array(pil_img.convert("RGB"))
    results = model.predict(rgb_img, conf=0.25, imgsz=640)
    result = results[0]

    if result is None:
        print("❌ 예측 결과 없음")
        return None

    # YOLO 출력 이미지를 PIL로 변환 후 원본 크기로 맞춤
    overlay_img = Image.fromarray(result.plot()).resize(pil_img.size, Image.NEAREST)
    overlay_img.save(overlay_save_path, format="PNG")
    return overlay_img

# ✅ 모든 클래스 ID, confidence, FDI 번호 반환
def get_all_class_info_json(pil_img):
    rgb_img = np.array(pil_img.convert("RGB"))
    results = model.predict(rgb_img, conf=0.25, imgsz=640)
    result = results[0]

    class_info_list = []

    if result.masks is None or len(result.boxes.cls) == 0:
        return class_info_list  # 예측된 항목 없음

    for i in range(len(result.boxes.cls)):
        class_id = int(result.boxes.cls[i].item())
        confidence = float(result.boxes.conf[i].item())
        tooth_number = FDI_CLASS_MAP.get(class_id, "Unknown")

        class_info_list.append({
            "class_id": class_id,
            "confidence": confidence,
            "tooth_number_fdi": tooth_number
        })

    return class_info_list
