from ultralytics import YOLO
import yaml
from pathlib import Path

if __name__ == "__main__":
    data_yaml = "datasets/rocks-4/data.yaml"

    # 1) TUNE
    model = YOLO("yolo11n.pt")

    search_space = {
        "lr0": (1e-5, 1e-1),
        "momentum": (0.6, 0.98),
        "weight_decay": (0.0, 0.01),
        # augmentation knobs you actually care about
        "degrees": (0.0, 20.0),
        "translate": (0.0, 0.2),
        "scale": (0.0, 0.7),
        "shear": (0.0, 5.0),
        "perspective": (0.0, 0.001),
        "hsv_h": (0.0, 0.05),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.6),
        "mosaic": (0.0, 1.0),
        "mixup": (0.0, 0.2),
        "fliplr": (0.0, 0.5),
        "flipud": (0.0, 0.1),
    }

    tune_results = model.tune(
        data=data_yaml,
        imgsz=640,
        epochs=20,  # per trial (keep smaller than your real train)
        iterations=50,  # start like 20-80, not 300
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,  # IMPORTANT so it optimizes mAP on val set
        device=0,
        workers=0,  # Windows
        project="runs/detect",
        name="tune_rocks",
        exist_ok=True,
        space=search_space,
    )

    # 2) LOAD BEST HYPERPARAMETERS (Ultralytics writes a best YAML in the tune run folder)
    # Try common expected location:
    tune_dir = Path("runs/detect/tune_rocks")
    best_yaml = None
    for p in tune_dir.rglob("*best*.yaml"):
        best_yaml = p
        break

    best = {}
    if best_yaml and best_yaml.exists():
        best = yaml.safe_load(best_yaml.read_text())
        print("Loaded best hyp:", best_yaml)
    else:
        print(
            "Couldn't find best YAML automatically; will just train with your defaults."
        )

    # 3) TRAIN FINAL (your long run) using tuned params + your fixed settings
    final_args = dict(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        device=0,
        workers=0,
        project="runs/detect",
        name="train",
        exist_ok=True,
        optimizer="AdamW",
    )
    final_args.update(best)  # tuned hyperparams override defaults

    model = YOLO("yolo11n.pt")  # fresh start for the real run
    model.train(**final_args)
