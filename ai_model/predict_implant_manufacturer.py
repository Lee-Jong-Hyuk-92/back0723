import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
import ttach as tta
from ai_model.xray_detector import detect_xray  # YOLO 탐지 결과 사용 (박스 정보만 활용)

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
        # detect_xray를 호출하여 임플란트 박스 정보를 얻습니다.
        # 이 단계에서 생성되는 이미지는 사용하지 않습니다.
        result_from_detector = detect_xray(xray_image_path) 
        
        # 임플란트 분류 이미지를 그릴 베이스로 원본 X-ray 이미지를 로드합니다.
        image = Image.open(xray_image_path).convert("RGB") 
    except Exception as e:
        print(f"❌ 오류: {e}")
        return [], None # 빈 리스트와 None을 반환

    predictions = []
    
    # 임플란트 박스와 클래스 번호를 그릴 Image 객체 (원본 이미지를 복사하여 사용)
    implant_overlay_image = image.copy() 
    draw = ImageDraw.Draw(implant_overlay_image)

    # 폰트 설정 (흰색 배경에 잘 보이도록 검은색 폰트 사용)
    try:
        # NanumGothicBold 폰트 경로를 사용합니다.
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 20)
    except IOError:
        font = ImageFont.load_default() # 폰트 로드 실패 시 기본 폰트 사용

    for obj in result_from_detector['detections']: # xray_detector의 탐지 결과 사용
        if obj['class_name'] != IMPLANT_CLASS_NAME:
            continue

        x1, y1, x2, y2 = map(int, obj['bbox'])
        cropped = image.crop((x1, y1, x2, y2)) # 원본 이미지에서 크롭

        try:
            pred_class, confidence = predict_crop_image(cropped)
        except Exception as e:
            print(f"❌ 임플란트 예측 실패: {e}")
            continue

        # ✅ 박스 위에 클래스 번호만 그리기 (검은색 글씨)
        label_text = str(pred_class) # 클래스 번호만
        
        # 텍스트 크기를 얻어 텍스트 박스 위치 계산
        # draw.textbbox((x, y), text, font)는 폰트에 따라 텍스트의 바운딩 박스를 반환합니다.
        # y1 - 22는 텍스트가 박스 위에 위치하도록 조정하는 예시이며, 폰트 크기에 따라 조정 필요
        text_bbox = draw.textbbox((x1, y1 - 22), label_text, font=font) 
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 텍스트 배경 (박스 위에 라벨이 잘 보이도록)
        # 텍스트가 시작되는 x1, y1-22에서 텍스트의 폭과 높이를 고려하여 배경 박스를 그립니다.
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - 5 # 텍스트가 박스 바로 위에 위치하도록 y 좌표 조정
        text_bg_x2 = x1 + text_width + 5
        text_bg_y2 = y1 - 5 # 텍스트 박스 아래쪽 y 좌표 조정
        
        # 임플란트 박스 자체는 빨간색 유지, 텍스트 배경은 노란색, 글씨는 검은색
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2) # 임플란트 박스 (빨간색)
        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill="yellow") # 텍스트 배경 (노란색)
        draw.text((text_bg_x1 + 2, text_bg_y1), label_text, font=font, fill=(0, 0, 0)) # 텍스트 (검은색)

        predictions.append({
            "original_image": xray_image_path,
            "bbox": [x1, y1, x2, y2],
            "predicted_manufacturer_class": pred_class,
            "predicted_manufacturer_name": MANUFACTURER_MAP.get(pred_class, f"{pred_class}번 알 수 없음"),
            "confidence": round(confidence, 3)
        })
    
    # 임플란트 박스와 클래스 번호가 그려진 이미지를 반환 (경로와 Image 객체 모두)
    # 임플란트 전용 이미지 저장 경로
    implant_output_path = xray_image_path.replace(".png", "_implant_classified.png").replace(".jpg", "_implant_classified.jpg")
    implant_overlay_image.save(implant_output_path)

    return predictions, implant_output_path # 예측 결과와 새로운 이미지 경로를 튜플로 반환

# ✅ CLI 실행
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗ 사용법: python predict_implant_manufacturer.py [X-ray 이미지 경로]")
        sys.exit(1)

    test_image_path = sys.argv[1]

    if not os.path.exists(test_image_path):
        print(f"❌ 파일이 존재하지 않습니다: {test_image_path}")
        sys.exit(1)

    results, output_image_path = classify_implants_from_xray(test_image_path) # 결과와 이미지 경로를 받음

    if not results:
        print("🔍 감지된 임플란트 없음")
    else:
        print("📦 예측 결과:")
        for r in results:
            print(r)
        print(f"🎨 결과 이미지가 저장되었습니다: {output_image_path}") # 저장된 이미지 경로 출력
