from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("runs/detect/tune_rocks2/weights/best.pt")

    metrics = model.val(
        data="datasets/rocks-4/data.yaml",  # be explicit; don't rely on "remembered" settings
        device=0,
        workers=0,  # IMPORTANT on Windows
        batch=8,  # optional; keep small if VRAM is tight
    )

    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("per-class mAP50-95:", metrics.box.maps)


if __name__ == "__main__":
    freeze_support()  # safe on Windows; prevents spawn weirdness
    main()
