import cv2
import numpy as np


def visualize(
    frame: np.ndarray,
    detections: list[dict],
    tracks: dict,
    count_in: int,
    count_out: int,
    line_y: int,
) -> np.ndarray:
    """Draw bounding boxes, centroids, counting line and counters on frame."""
    out = frame.copy()

    # Counting line
    cv2.line(out, (0, line_y), (out.shape[1], line_y), (0, 255, 255), 2)

    # Bounding boxes and centroids from detections
    for d in detections:
        x, y, w, h = d["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.circle(out, (d["cx"], d["cy"]), 4, (0, 0, 255), -1)

    # Track IDs
    for tid, t in tracks.items():
        cv2.putText(
            out, f"#{tid}", (t["cx"] + 6, t["cy"] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA,
        )

    # Counter overlay
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (200, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    cv2.putText(out, f"IN:  {count_in}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
    cv2.putText(out, f"OUT: {count_out}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    return out

def visualize_tracks(
    frame: np.ndarray,
    tracks: dict,
    count_in: int,
    count_out: int,
    line_y: int,
) -> np.ndarray:
    out = frame.copy()
    cv2.line(out, (0, line_y), (out.shape[1], line_y), (0, 255, 255), 2)

    for tid, t in tracks.items():
        x, y, w, h = t["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.circle(out, (t["cx"], t["cy"]), 4, (0, 0, 255), -1)
        cv2.putText(
            out, f"#{tid}", (t["cx"] + 6, t["cy"] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA,
        )

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (200, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    cv2.putText(out, f"IN:  {count_in}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
    cv2.putText(out, f"OUT: {count_out}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    return out