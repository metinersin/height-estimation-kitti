import random
import numpy as np


def segment(img: np.ndarray, prompt: str) -> np.ndarray:
    """
    Segment the image using the specified prompt.

    Parameters:
        img (np.ndarray): The input image as a numpy array.
        prompt (str): The prompt for segmentation.

    Returns:
        np.ndarray: The segmented image as a numpy array.
    """

    # Mock implementation
    mask = np.random.randint(0, 2, img.shape[:2])
    mask = np.asarray(mask, dtype=bool)
    return mask
