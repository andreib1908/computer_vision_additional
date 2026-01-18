from ultralytics import YOLO

model = YOLO("runs/detect/tune_rocks2/weights/best.pt")

# conf interval -> greater value less false positives
# iou -> greater value less overlap between bounding boxes; what counts as a "correct" detection compared to the ground truth
# iou for single rocks -> (0.45, 0.6)
# iou for clusters of rocks -> (0.6, 0.75)
# See Params: https://docs.ultralytics.com/modes/predict/#inference-arguments
model.predict(
    source="http://10.10.0.157:8080/video",
    device=0,
    conf=0.55,
    iou=0.55,
    imgsz=640,
    show=True,
    # half=True,
    # max_det=150,
    # visualize=True,
    # augment=True,
)
