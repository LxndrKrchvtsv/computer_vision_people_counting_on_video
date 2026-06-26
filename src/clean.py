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