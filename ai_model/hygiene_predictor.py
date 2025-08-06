import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
import os

# ✅ 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai_model", "hygiene_model_saved_weight.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 클래스 ID → 라벨명 매핑
HYGIENE_CLASS_MAP = {
    1: "아말감 (am)",
    2: "세라믹 (cecr)",
    3: "골드 (gcr)",
    4: "메탈크라운 (mcr)",
    5: "교정장치 (ortho)",
    6: "치석 단계1 (tar1)",
    7: "치석 단계2 (tar2)",
    8: "치석 단계3 (tar3)",
    9: "지르코니아 (zircr)"
}

# ✅ 모델 정의 및 로드
model = UnetPlusPlus(
    encoder_name="efficientnet-b7",
    encoder_weights=None,
    in_channels=3,
    classes=10
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

# ✅ 후처리 (클래스 → RGBA 컬러맵)
def postprocess(output_tensor, target_size=(224, 224)):
    pred = torch.argmax(output_tensor.squeeze(0), dim=0).cpu().numpy()

    PALETTE = {
        0: (0, 0, 0, 0),              # background
        1: (220, 20, 60, 200),        # 아말감 (am): 🔴 진한 빨강
        2: (138, 43, 226, 200),       # 세라믹 (cecr): 🟣 보라
        3: (255, 215, 0, 200),        # 골드 (gcr): 🟡 노랑 (골드톤)
        4: (245, 245, 245, 200),      # 메탈크라운 (mcr): ⚪ 거의 흰색
        5: (30, 30, 30, 200),         # 교정장치 (ortho): ⚫ 진한 검정
        6: (0, 255, 0, 200),          # 치석 단계1 (tar1): 🟢 초록
        7: (255, 140, 0, 200),        # 치석 단계2 (tar2): 🟠 주황
        8: (0, 0, 255, 200),          # 치석 단계3 (tar3): 🔵 파랑
        9: (139, 69, 19, 200),        # 지르코니아 (zircr): 🟤 갈색
    }

    h, w = pred.shape
    color_mask = np.zeros((h, w, 4), dtype=np.uint8)

    for class_id, color in PALETTE.items():
        color_mask[pred == class_id] = color

    return Image.fromarray(color_mask, mode="RGBA").resize(target_size)

# ✅ 예측 + 합성 마스크 저장
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

# ✅ 주요 클래스 ID, confidence, 라벨명 반환
def get_main_class_and_confidence_and_label(pil_img):
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

    best_label = HYGIENE_CLASS_MAP.get(best_class, "Unknown")

    return int(best_class), float(best_conf), best_label
