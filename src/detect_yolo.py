import numpy as np
import cv2
from ultralytics import YOLO

_model = None


def get_model(weights: str = "yolov8n.pt") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def detect_people_yolo(
    frame: np.ndarray,
    mask: np.ndarray,
    motion_threshold: int = 5000,
    conf: float = 0.35,
    weights: str = "yolov8n.pt",
) -> list[dict] | None:
    """
    Run YOLO person detection only when MOG2 mask has enough motion area.

    Returns list of dicts {cx, cy, bbox, conf} or None if motion below threshold
    (caller should skip tracking update on None).
    """
    motion_area = int(np.sum(mask > 0))
    if motion_area < motion_threshold:
        return None  # no significant motion — skip YOLO

    model = get_model(weights)
    results = model(frame, classes=[0], conf=conf, verbose=False)[0]  # class 0 = person

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        detections.append({
            "cx": cx, "cy": cy,
            "bbox": (x1, y1, w, h),
            "conf": float(box.conf[0]),
        })
    return detections
