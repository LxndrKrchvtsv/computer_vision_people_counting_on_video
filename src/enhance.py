import cv2
import numpy as np


def enhance_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Equalize lighting using CLAHE on the L channel of LAB color space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def enhance_gamma(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply gamma correction to brighten or darken a frame."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)