from ultralytics import YOLO
import torch
import multiprocessing

def main():
    # Load model
    model = YOLO("yolo11n.pt")

    # Train with multiple workers for speed
    results = model.train(
        data=r"C:\Users\badea\OneDrive\Documents\Coding Projects\PyTorch ML\datasets\rocks-stolen\data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="rock_detector_fast",
        project="runs/train",
        workers=4,           # ✅ 2–8 depending on CPU cores
        device=0 if torch.cuda.is_available() else "cpu",
    )

    # Validate
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)

    # Predict on a video
    video_path = r"C:\Users\badea\Downloads\youtube_a bunch of rocks_.mp4"
    results = model.predict(source=video_path, save=True, conf=0.4)
    print("Saved predictions to:", results[0].save_dir)


if __name__ == "__main__":
    # ✅ Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
