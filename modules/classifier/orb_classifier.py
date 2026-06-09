"""
Module: modules/classifier/orb_classifier.py
Description: Classifies video frames as 'open' or 'close' using ORB (Oriented FAST and Rotated BRIEF) 
             feature keypoint descriptors and a Brute-Force Matcher (BFMatcher). Computes and compares 
             the highest number of valid localized texture matches against reference eye state templates.
"""

from typing import List, Optional

import cv2
import numpy as np

from config import MATCH_DISTANCE_THRESHOLD, MIN_OPEN_MATCHES, ORB_MAX_FEATURES


class ORBClassifier:
    """
    Classifies image frames by matching feature descriptors against open and closed template pools.

    Parameters
    ----------
    open_images : List[np.ndarray], optional
        List of BGR image arrays representing reference 'open' eye templates.
    close_images : List[np.ndarray], optional
        List of BGR image arrays representing reference 'close' eye templates.
    """

    def __init__(
        self,
        open_images: Optional[List[np.ndarray]] = None,
        close_images: Optional[List[np.ndarray]] = None,
    ):
        """
        Pre:
            - open_images and close_images must be either None or lists containing valid image arrays.
        Post:
            - Initializes the underlying cv2.ORB and cv2.BFMatcher instances.
            - Extracts and populates internal descriptor matrices if template lists are provided.
        """
        self._orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._des_open: List[np.ndarray] = []
        self._des_close: List[np.ndarray] = []

        if open_images:
            self.set_open_templates(open_images)
        if close_images:
            self.set_close_templates(close_images)

    def set_open_templates(self, images: List[np.ndarray]):
        """
        Pre:
            - images must be a list containing valid numpy.ndarray structures.
        Post:
            - Extracts, filters, and replaces the stored internal Open eye descriptor structures.
        """
        self._des_open = [d for d in (self._compute_des(img) for img in images) if d is not None]

    def set_close_templates(self, images: List[np.ndarray]):
        """
        Pre:
            - images must be a list containing valid numpy.ndarray structures.
        Post:
            - Extracts, filters, and replaces the stored internal Close eye descriptor structures.
        """
        self._des_close = [d for d in (self._compute_des(img) for img in images) if d is not None]

    def classify(self, img: np.ndarray) -> str:
        """
        Pre:
            - img must be a valid numpy.ndarray representing an image structure.
        Post:
            - Returns a string token ('open' or 'close') indicating the calculated eye state classification.
            - Defaults to 'close' if feature extraction fails.
        """
        des_target = self._compute_des(img)
        if des_target is None:
            return "close"

        max_open = self._max_matches(des_target, self._des_open)
        max_close = self._max_matches(des_target, self._des_close)

        if max_open > max_close and max_open > MIN_OPEN_MATCHES:
            return "open"
        return "close"

    def classify_path(self, img_path: str) -> str:
        """
        Pre:
            - img_path must be a valid string pointing to an existing and readable image file.
        Post:
            - Reads the file and returns its eye state token ('open' or 'close') by calling classify().
            - Defaults to 'close' if the image file cannot be read.
        """
        img = cv2.imread(img_path)
        if img is None:
            return "close"
        return self.classify(img)

    def _compute_des(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Pre:
            - img can be a valid numpy.ndarray image or None.
        Post:
            - Converts the image matrix to grayscale if needed and runs keypoint description matching.
            - Returns a numpy.ndarray containing the computed features, or None if extraction is impossible.
        """
        if img is None:
            return None
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = self._orb.detectAndCompute(gray, None)
        return des

    def _max_matches(
        self, des_target: np.ndarray, des_list: List[np.ndarray]
    ) -> int:
        """
        Pre:
            - des_target must be a valid numpy descriptor matrix.
            - des_list must be a list containing feature descriptor arrays extracted from reference templates.
        Post:
            - Returns an integer corresponding to the highest number of filtered proximity matches 
              found between des_target and any individual template descriptor in des_list.
        """
        best = 0
        for des_tmpl in des_list:
            if des_tmpl is None:
                continue
            matches = self._bf.match(des_tmpl, des_target)
            good = sum(1 for m in matches if m.distance < MATCH_DISTANCE_THRESHOLD)
            if good > best:
                best = good
        return best
