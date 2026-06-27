import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Remove noise and fill holes in a binary foreground mask.

    Opening (erosion then dilation) removes small noise blobs.
    Closing (dilation then erosion) fills holes inside silhouettes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def filter_components(mask: np.ndarray,
                      min_area: int = 500,
                      max_area: int = 50000,
                      min_aspect: float = 1.0,
                      max_aspect: float = 5.0) -> np.ndarray:
    """Keep only connected components that look like a standing person."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        aspect = h / w if w > 0 else 0
        if aspect < min_aspect or aspect > max_aspect:
            continue
        out[labels == i] = 255
    return out
