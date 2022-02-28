import numpy as np


def to_8bit(img: np.ndarray) -> np.ndarray:
    img = img / img.max() * 255  # normalizes data in range 0 - 255
    img = img.astype(np.uint8)
    return img
