from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="datasets/rocks-3/data.yaml",  # or your fixed rock-only yaml
        epochs=50,
        imgsz=640,
        device=0,
        workers=0,  # Windows
        project="runs/detect",
        name="train",  # always runs/detect/train
        exist_ok=True,  # overwrite the folder
        # augment knobs (start mild)
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
    )
