import cv2
import numpy as np


def detect_people(mask: np.ndarray, min_area: int = 500, max_area: int = 50000) -> list[dict]:
    """
    Find people in a binary mask using contour detection.

    Returns a list of dicts with keys: cx, cy, bbox (x, y, w, h), area.
    Contours smaller than min_area are ignored as noise.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        detections.append({"cx": cx, "cy": cy, "bbox": (x, y, w, h), "area": area})
    return detections