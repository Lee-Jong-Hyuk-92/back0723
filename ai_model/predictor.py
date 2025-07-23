import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 ID → 라벨명 매핑
DISEASE_CLASS_MAP = {
    1: "충치 초기",
    2: "충치 중기",
    3: "충치 말기",
    4: "잇몸 염증 초기",
    5: "잇몸 염증 중기",
    6: "잇몸 염증 말기",
    7: "치주질환 초기",
    8: "치주질환 중기",
    9: "치주질환 말기"
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
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (255, 165, 0),
    8: (128, 0, 128),
    9: (128, 128, 128),
}

def predict_overlayed_image(pil_img: Image.Image) -> Tuple[Image.Image, List[List[int]], float, str, str]:
    original_img_resized = pil_img.resize((224, 224)).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_logits = model(input_tensor)[0].cpu().numpy()
        probabilities = torch.softmax(torch.from_numpy(output_logits), dim=0).numpy()
        pred_mask = np.argmax(output_logits, axis=0)

    color_mask = np.zeros((224, 224, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        color_mask[pred_mask == class_id] = color
    color_mask_img = Image.fromarray(color_mask)

    overlay = Image.blend(original_img_resized, color_mask_img, alpha=0.78)

    lesion_coords = np.column_stack(np.where(pred_mask > 0))
    lesion_points = lesion_coords.tolist()

    backend_model_confidence = 0.0
    if lesion_coords.shape[0] > 0:
        lesion_pixel_confidences = [
            probabilities[pred_mask[y, x], y, x]
            for y, x in lesion_coords
            if pred_mask[y, x] > 0
        ]
        if lesion_pixel_confidences:
            backend_model_confidence = np.mean(lesion_pixel_confidences)

    lesion_labels = pred_mask[pred_mask > 0]
    if len(lesion_labels) > 0:
        most_common_class = np.bincount(lesion_labels).argmax()
        main_class_label = DISEASE_CLASS_MAP.get(most_common_class, "알 수 없음")
    else:
        main_class_label = "감지되지 않음"

    if os.environ.get("FLASK_DEBUG") == "1":
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(original_img_resized)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(color_mask_img)
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return overlay, lesion_points, float(backend_model_confidence), BACKEND_MODEL_NAME, main_class_label
