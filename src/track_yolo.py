import numpy as np
from ultralytics import YOLO

_model = None

def get_model(weights: str = "yolov8n.pt") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model

def track_people_yolo(
        frame: np.ndarray,
        conf: float = 0.35,
        weights: str = "yolov8m.pt",
        imgsz: int = 1280
) -> list[dict]:
    model = get_model(weights)
    results = model.track(
        frame,
        classes = [0],
        conf = conf,
        imgsz = imgsz,
        persist = True,
        tracker = "bytetrack_custom.yaml",
        verbose = False,
    )[0]

    detections = []
    if results.boxes is None or results.boxes.id is None:
        return detections

    ids = results.boxes.id.int().tolist()
    for box, tid in zip(results.boxes, ids):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detections.append({
            "id": tid,
            "cx": cx,
            "cy": cy,
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "conf": float(box.conf[0]),
        })
    return detections
