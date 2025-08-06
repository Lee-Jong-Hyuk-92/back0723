import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_CLASS_MAP = {
    1: "충치 초기", 2: "충치 중기", 3: "충치 말기",
    4: "잇몸 염증 초기", 5: "잇몸 염증 중기", 6: "잇몸 염증 말기", # 치은염
    7: "치주질환 초기", 8: "치주질환 중기", 9: "치주질환 말기"  # 치주염, 치은염이 심해진것, 잇몸 염증 + 잇몸 뼈 염증
}

n_labels = 10
model = smp.UnetPlusPlus(
    encoder_name='efficientnet-b7',
    encoder_weights='imagenet',
    classes=n_labels,
    activation=None
)
model_path = os.path.join(os.path.dirname(__file__), 'disease_model_saved_weight.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

BACKEND_MODEL_NAME = os.path.basename(model_path)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

PALETTE = {
    0: (0, 0, 0, 0),        # background - 완전 투명
    1: (255, 0, 0, 180),
    2: (0, 255, 0, 180),
    3: (0, 0, 255, 180),
    4: (255, 255, 0, 180),
    5: (255, 0, 255, 180),
    6: (0, 255, 255, 180),
    7: (255, 165, 0, 180),
    8: (128, 0, 128, 180),
    9: (128, 128, 128, 180),
}

def predict_overlayed_image(pil_img: Image.Image) -> Tuple[Image.Image, List[List[int]], float, str, str]:
    original_img_resized = pil_img.resize((224, 224)).convert('RGBA')  # ✅ RGBA로 변환
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_logits = model(input_tensor)
        output_logits_np = output_logits.squeeze(0).cpu().numpy()
        probabilities = torch.softmax(torch.from_numpy(output_logits_np), dim=0).numpy()
        pred_mask = np.argmax(output_logits_np, axis=0)

    # ✅ RGBA 마스크 생성
    color_mask = np.zeros((224, 224, 4), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        color_mask[pred_mask == class_id] = color
    color_mask_img = Image.fromarray(color_mask, mode='RGBA')

    # ✅ 이미지 합성 (alpha_composite)
    overlay = Image.alpha_composite(original_img_resized, color_mask_img)

    lesion_coords = np.column_stack(np.where(pred_mask > 0))
    lesion_points = lesion_coords.tolist()

    backend_model_confidence = 0.0
    if lesion_coords.shape[0] > 0:
        lesion_pixel_confidences = [
            probabilities[pred_mask[y, x], y, x]
            for y, x in lesion_coords if pred_mask[y, x] > 0
        ]
        if lesion_pixel_confidences:
            backend_model_confidence = np.mean(lesion_pixel_confidences)

    # ✅ 추가: 등장한 class들을 set으로 추출
    lesion_labels = pred_mask[pred_mask > 0]
    detected_class_ids = sorted(list(set(lesion_labels.tolist())))
    detected_labels = [DISEASE_CLASS_MAP.get(cid, "Unknown") for cid in detected_class_ids]

    # ✅ 가장 많이 나온 클래스만 따로 지정
    if len(lesion_labels) > 0:
        most_common_class = np.bincount(lesion_labels).argmax()
        main_class_label = DISEASE_CLASS_MAP.get(most_common_class, "알 수 없음")
    else:
        main_class_label = "감지되지 않음"

    return overlay, lesion_points, float(backend_model_confidence), BACKEND_MODEL_NAME, main_class_label, detected_labels
