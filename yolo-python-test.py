from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolo11n.pt")
model.predict(source=0, device=1, imgsz=640, conf=0.10, iou=0.45, show=True)
