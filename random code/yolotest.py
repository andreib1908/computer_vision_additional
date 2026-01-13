from ultralytics import YOLO

def main():
    # Load a pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Train for 3 epochs on coco8
    results = model.train(data="coco8.yaml", epochs=3, device=0)

    # Validate
    model.val()

    # Run inference
    results = model("https://ultralytics.com/images/bus.jpg")

    # Export to ONNX
    model.export(format="onnx")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # ensures safe start on Windows
    main()
