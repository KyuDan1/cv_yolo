import cv2
import torch
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5로 객체 감지
    results = model(frame)

    # 결과 처리
    detections = results.xyxy[0].cpu().numpy()

    # 감지된 객체에 바운딩 박스 그리기
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # 신뢰도가 50% 이상인 경우만 표시
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('YOLO Real-time Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()