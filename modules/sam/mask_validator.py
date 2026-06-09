"""
Module: modules/sam/mask_validator.py
Description: Handles loading tracking masks from disk in both PNG and NPY serialization formats, 
             converting the array elements to binary float32 matrices, and validating structural 
             properties for compliance with SAM2 video predictor data type expectations.
"""

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class MaskValidator:
    """Provides utility methods for loading and validating segmentation mask data formats."""

    @staticmethod
    def load_png(path: str) -> np.ndarray:
        """
        Pre:
            - path must be a valid string pointing to an existing and readable PNG image file.
        Post:
            - Loads the grayscale image file, thresholds values above 0.5, and returns a binary 
              numpy.ndarray filled with float32 {0.0, 1.0} elements.
        """
        raw = Image.open(path).convert("L")
        arr = (np.array(raw) / 255.0 > 0.5).astype(np.float32)
        return arr

    @staticmethod
    def load_npy(path: str) -> Optional[np.ndarray]:
        """
        Pre:
            - path must be a valid string path representation.
        Post:
            - Returns the parsed matrix formatted as a float32 array if the target file exists.
            - Returns None if no file is found at the specified path.
        """
        if not os.path.isfile(path):
            return None
        arr = np.load(path)
        return arr.astype(np.float32)

    @staticmethod
    def validate(mask: np.ndarray) -> bool:
        """
        Pre:
            - mask must be an instantiated object.
        Post:
            - Returns True if mask is a 2D numpy array containing values belonging to a floating-point data type.
            - Returns False otherwise.
        """
        return (
            isinstance(mask, np.ndarray)
            and mask.ndim == 2
            and np.issubdtype(mask.dtype, np.floating)
        )
