from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="datasets/rocks-good/data.yaml",  # or your fixed rock-only yaml
        epochs=50,
        imgsz=640,
        device=0,
        workers=0,  # Windows
    )
