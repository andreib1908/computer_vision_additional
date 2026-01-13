from ultralytics import YOLO

def main():
    # Load a pretrained YOLO model (transfer learning)
    model = YOLO("yolo11n.pt")  # nano version for Jetson-compatible training

    # Start training on your custom rock dataset
    model.train(
        data="datasets/rocks/data.yaml",  # dataset configuration file
        epochs=100,                       # increase if dataset is large
        imgsz=640,                        # resize training images
        batch=16,                         # adjust for your GPU RAM
        device=0,                         # use GPU
        name="train_rocks_yolo11n",       # run name
        workers=4,                        # reduce if on Windows
    )

    # Evaluate after training
    model.val()

    # Run a quick test prediction
    results = model.predict(
        source="datasets/rocks/images/val", 
        show=True,
        conf=0.25
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required on Windows
    main()
