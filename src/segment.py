import cv2
import numpy as np


def create_subtractor(history: int = 500, var_threshold: float = 16.0, detect_shadows: bool = False):
    """Create a MOG2 background subtractor instance."""
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def segment_mog2(frame: np.ndarray, subtractor) -> np.ndarray:
    """Apply background subtraction to get a binary foreground mask."""
    mask = subtractor.apply(frame)
    # Threshold to strict binary (remove shadows if they were kept)
    _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    return binary