import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from segmentation_models_pytorch import FPN
import os

# ✅ 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai_model", "tooth_number_saved_weight.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ FDI 치아 번호 매핑
FDI_CLASS_MAP = {
    1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18,
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    17: 31, 18: 32, 19: 33, 20: 34, 21: 35, 22: 36, 23: 37, 24: 38,
    25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48
}

# ✅ 모델 정의 및 로드
model = FPN(
    encoder_name="efficientnet-b7",
    encoder_weights=None,
    in_channels=3,
    classes=33
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ 전처리
def preprocess(pil_img, size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(pil_img).unsqueeze(0)

# ✅ 후처리
def postprocess(output_tensor, target_size=(224, 224)):
    pred = torch.argmax(output_tensor.squeeze(0), dim=0).cpu().numpy()

    PALETTE = {
        0: (0, 0, 0, 0),  # background (그대로)
        1: (255, 0, 0, 200), 2: (0, 255, 0, 200), 3: (0, 0, 255, 200),
        4: (255, 255, 0, 200), 5: (255, 0, 255, 200), 6: (0, 255, 255, 200),
        7: (128, 0, 128, 200), 8: (255, 165, 0, 200), 9: (128, 128, 128, 200),
        10: (0, 128, 0, 200), 11: (0, 0, 128, 200), 12: (139, 0, 0, 200),
        13: (0, 139, 139, 200), 14: (75, 0, 130, 200), 15: (220, 20, 60, 200),
        16: (154, 205, 50, 200), 17: (255, 215, 0, 200), 18: (0, 191, 255, 200),
        19: (255, 105, 180, 200), 20: (70, 130, 180, 200), 21: (160, 82, 45, 200),
        22: (60, 179, 113, 200), 23: (255, 20, 147, 200), 24: (47, 79, 79, 200),
        25: (218, 165, 32, 200), 26: (255, 160, 122, 200), 27: (199, 21, 133, 200),
        28: (32, 178, 170, 200), 29: (46, 139, 87, 200), 30: (123, 104, 238, 200),
        31: (127, 255, 0, 200), 32: (255, 99, 71, 200)
    }

    h, w = pred.shape
    color_mask = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        color_mask[pred == class_id] = color

    return Image.fromarray(color_mask, mode="RGBA").resize(target_size)

# ✅ 예측 + 원본 위에 합성 저장
def predict_mask_and_overlay_only(pil_img, overlay_save_path):
    input_tensor = preprocess(pil_img).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        output = F.softmax(output, dim=1)

    mask_img = postprocess(output)
    resized_img = pil_img.resize(mask_img.size).convert("RGBA")
    overlayed = Image.alpha_composite(resized_img, mask_img)

    # ✅ RGBA → RGB 변환 후 PNG로 저장
    overlayed = overlayed.convert("RGB")
    overlayed.save(overlay_save_path, format="PNG")
    return overlayed

# ✅ 주요 클래스 ID, confidence, 치아번호 라벨 포함 JSON 반환
def get_main_class_info_json(pil_img):
    input_tensor = preprocess(pil_img).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        output = F.softmax(output, dim=1)

    pred = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
    output_np = output.squeeze(0).cpu().numpy()

    class_ids, counts = np.unique(pred, return_counts=True)

    best_class = -1
    best_conf = 0.0
    for cid, cnt in zip(class_ids, counts):
        if cid == 0:
            continue
        class_conf = output_np[cid][pred == cid].mean()
        if cnt > 0 and class_conf > best_conf:
            best_class = cid
            best_conf = class_conf

    return {
        "class_id": int(best_class),
        "confidence": float(best_conf),
        "tooth_number_fdi": FDI_CLASS_MAP.get(best_class, "Unknown")
    }