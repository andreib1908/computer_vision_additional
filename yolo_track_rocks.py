from ultralytics import YOLO

model = YOLO("runs/detect/train105/weights/best.pt")

model.track(
    source="https://145.126.46.69:8080/video", device=0, conf=0.6, imgsz=640, show=True
)
