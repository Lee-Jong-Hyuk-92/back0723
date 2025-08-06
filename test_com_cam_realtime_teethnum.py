import cv2
from ultralytics import YOLO

# ✅ 모델 경로 지정
model_path = r'C:\Users\302-1\Desktop\back0723\ai_model\number_x_250806.pt'
model = YOLO(model_path)

# ✅ 웹캠 열기 (0번 카메라, 다른 장치면 1 또는 2로 바꿔보세요)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("✅ 실시간 치아 번호 세그멘테이션 시작 (ESC 누르면 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # YOLO 예측 (stream=False면 더 빠르지만 약간 느려도 안정적)
    results = model.predict(source=frame, conf=0.25, stream=True)

    # 결과 시각화 및 출력
    for result in results:
        annotated_frame = result.plot()  # 마스크와 라벨이 포함된 이미지
        cv2.imshow('YOLO 치아 번호 예측', annotated_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()