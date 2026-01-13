from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=2, imgsz=640, device=0, workers=0)
