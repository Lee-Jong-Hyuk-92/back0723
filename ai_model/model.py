import os
import json
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# === [YOLO 모델 로딩] ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
_model = None

try:
    _model = YOLO(MODEL_PATH)
except Exception as e:
    _model = None

# === [AI 추론 함수] =================================
def perform_inference(image_path, processed_output_dir):
    """
    주어진 이미지에 대해 YOLOv11-seg 추론을 수행하고,
    결과가 그려진 이미지를 저장하고 JSON 데이터를 반환합니다.
    """
    if _model is None:
        return {
            "error": "AI 모델이 로드되지 않았습니다.",
            "prediction": "N/A",
            "details": [],
            "processed_image_path": None
        }

    print(f"🚀 YOLOv11-seg 추론 시작: {image_path}")

    os.makedirs(processed_output_dir, exist_ok=True)

    try:
        # 추론 수행
        results = _model(image_path, conf=0.25, iou=0.7, save=False)
        annotated_img = results[0].plot()

        # 처리된 이미지 저장
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        processed_filename = f"processed_{timestamp}_{name}{ext}"
        processed_full_path = os.path.join(processed_output_dir, processed_filename)
        cv2.imwrite(processed_full_path, annotated_img)

        print(f"📷 처리된 이미지 저장 완료: {processed_full_path}")

        # 상대 경로 반환용
        relative_processed_path = f"/processed_uploads/camera/{processed_filename}"

        # 결과 파싱
        inference_details = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]

                    detail = {
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "segmentation_info_present": bool(r.masks is not None)
                    }
                    inference_details.append(detail)

            elif r.masks:
                # 마스크만 있는 경우 처리
                inference_details.append({
                    "box": [],
                    "confidence": 0.0,
                    "class_id": -1,
                    "class_name": "unknown",
                    "segmentation_info_present": True
                })

        return {
            "prediction": "Objects detected" if inference_details else "No objects detected",
            "details": inference_details,
            "processed_image_path": relative_processed_path  # ✅ URL 접근 가능한 경로로 변경
        }

    except Exception as e:
        print(f"❌ 추론 중 오류 발생: {e}")
        return {
            "error": f"AI 추론 실패: {str(e)}",
            "prediction": "N/A",
            "details": [],
            "processed_image_path": None
        }

# === [단독 실행 테스트] ==============================
if __name__ == '__main__':
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')
    test_output_dir = os.path.join(os.path.dirname(__file__), 'test_processed_images')

    if not os.path.exists(test_image_path):
        print(f"⚠️ 테스트 이미지 없음: {test_image_path}")
    else:
        print("🧪 테스트 추론 시작...")
        result = perform_inference(test_image_path, test_output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))