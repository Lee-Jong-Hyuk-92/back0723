import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

# ✅ 클래스 컬러맵 (0~9)
CLASS_COLORS = [
    (0, 0, 0),         # 0: background
    (255, 0, 0),       # 1
    (0, 255, 0),       # 2
    (0, 0, 255),       # 3
    (255, 255, 0),     # 4
    (255, 0, 255),     # 5
    (0, 255, 255),     # 6
    (128, 0, 128),     # 7
    (0, 128, 128),     # 8
    (128, 128, 0),     # 9
]

def apply_colormap(mask):
    """클래스 인덱스 마스크를 RGB 색상 마스크로 변환"""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == class_idx] = color
    return color_mask

# ✅ 모델 로드
model = smp.UnetPlusPlus('efficientnet-b7', encoder_weights='imagenet', classes=10, activation=None)
model.load_state_dict(torch.load("ai_model/hygiene_model_saved_weight.pt", map_location='cpu'))
model.eval()

# ✅ 이미지 로드
img_path = "C:/Users/302-1/4.png"
original = cv2.imread(img_path)               # BGR
resized = cv2.resize(original, (224, 224))
img = resized / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img = np.transpose(img, (2, 0, 1))
img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

# ✅ 예측
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# ✅ 컬러 마스크로 변환
colored_mask = apply_colormap(pred_mask)

# ✅ 오버레이 생성 (원본도 resize 해줘야 크기 맞음)
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(resized_rgb, 0.7, colored_mask, 0.3, 0)

# ✅ 저장
os.makedirs("C:/Users/302-1/Desktop/results", exist_ok=True)
cv2.imwrite("C:/Users/302-1/Desktop/results/original_resized.png", cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("C:/Users/302-1/Desktop/results/mask_colored.png", cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
cv2.imwrite("C:/Users/302-1/Desktop/results/overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
