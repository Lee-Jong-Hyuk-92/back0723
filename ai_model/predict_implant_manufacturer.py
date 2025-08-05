import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import ttach as tta
from ai_model.xray_detector import detect_xray  # YOLO 탐지 결과 사용

# 클래스 수
NUM_CLASSES = 42
IMPLANT_CLASS_NAME = '임플란트'

# 클래스 인덱스 → 제조사명 맵핑 (42개)
MANUFACTURER_MAP = {
    0: '바이오멧 3i', 1: 'ank', 2: '메가젠 Anyone external(aoe)', 3: '메가젠 Anyone internal(aoi)',
    4: '메가젠 any ridge', 5: '덴츠플라이 astra', 6: 'ati', 7: 'Biohorizon external',
    8: '스트라우만 Bone level', 9: '노벨바이오케어 branemark', 10: 'Cybermed core1', 11: '네오바이오텍 EB',
    12: 'exfeel internal', 13: '오스템 GS II', 14: '오스템 GS III', 15: '덴티움 implantium',
    16: '네오바이오텍 IS I', 17: '네오바이오텍 IS II', 18: '네오바이오텍 IS III', 19: 'it',
    20: '메가젠 IU', 21: '신흥 luna1', 22: '메가젠 luna2', 23: '메가젠 MII',
    24: '노벨바이오케어 Replace select', 25: '덴티움 S clean tapered', 26: '덴티움 Simpleline',
    27: '오스템 SS II', 28: '오스템 SS III', 29: '덴티움 Superline', 30: '오스템 TS III',
    31: '스트라우만 TS III', 32: '스트라우만 TS standard plus', 33: '짐머 TSV', 34: '디오임플란트 UF',
    35: '디오임플란트 UF2', 36: '오스템 US II', 37: '오스템 US III', 38: '메가젠 exfeel external',
    39: '덴츠플라이 Xive', 40: 'xi', 41: 'xin'
}

MODEL_PATH = "./ai_model/dm_nfnet_f0_best_acc_model_state.pt"
MODEL_NAME = "dm_nfnet_f0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

tta_transforms = tta.Compose([])

def predict_crop_image(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

def classify_implants_from_xray(xray_image_path: str):
    try:
        result = detect_xray(xray_image_path)
        image = Image.open(xray_image_path).convert("RGB")
    except Exception as e:
        print(f"❌ 오류: {e}")
        return []

    predictions = []

    for obj in result['detections']:
        if obj['class_name'] != IMPLANT_CLASS_NAME:
            continue

        x1, y1, x2, y2 = map(int, obj['bbox'])
        cropped = image.crop((x1, y1, x2, y2))

        try:
            pred_class, confidence = predict_crop_image(cropped)
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            continue

        predictions.append({
            "original_image": xray_image_path,
            "bbox": [x1, y1, x2, y2],
            "predicted_manufacturer_class": pred_class,
            "predicted_manufacturer_name": MANUFACTURER_MAP.get(pred_class, f"{pred_class}번"),
            "confidence": round(confidence, 3)
        })

    return predictions

# ✅ CLI 실행
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗ 사용법: python predict_implant_manufacturer.py [X-ray 이미지 경로]")
        sys.exit(1)

    test_image_path = sys.argv[1]

    if not os.path.exists(test_image_path):
        print(f"❌ 파일이 존재하지 않습니다: {test_image_path}")
        sys.exit(1)

    results = classify_implants_from_xray(test_image_path)

    if not results:
        print("🔍 감지된 임플란트 없음")
    else:
        print("📦 예측 결과:")
        for r in results:
            print(r)
