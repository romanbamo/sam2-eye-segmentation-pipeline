"""
Module: storage/folder_manager.py
Description: Defines and resolves all structured absolute or relative path workflows inside 
             the execution environment. Centralizes project directory tree configurations 
             for sessions, frames, tracking masks, and classification outputs.
"""

import os
from config import (
    BASE_DATA_DIR,
    FRAMES_SUBDIR,
    MASKS_SUBDIR,
    VISUALIZATIONS_SUBDIR,
    SESSIONS_SUBDIR,
)

VALID_EYES = ("left", "right")


class FolderManager:
    """Resolves and constructs all directory structures required for a specified video sequence."""

    def __init__(self, video_name: str):
        """
        Pre:
            - video_name must be a valid, non-empty string representing the base sequence folder.
        Post:
            - Initializes attributes and computes the root workspace pathway template using BASE_DATA_DIR.
        """
        self.video_name = video_name
        self.base = os.path.join(BASE_DATA_DIR, video_name)

    def video_root(self) -> str:
        """
        Pre:
            - None.
        Post:
            - Returns the string path corresponding to the base project directory.
        """
        return self.base

    def sessions_dir(self) -> str:
        """
        Pre:
            - None.
        Post:
            - Returns the string path corresponding to the internal system sessions subdirectory.
        """
        return os.path.join(self.base, SESSIONS_SUBDIR)

    def eye_root(self, eye: str) -> str:
        """
        Pre:
            - eye must be a string exactly matching 'left' or 'right'.
        Post:
            - Returns the string path pointing to the root subdirectory of the selected eye side.
            - Raises a ValueError if the side identifier is invalid.
        """
        self._check_eye(eye)
        return os.path.join(self.base, eye)

    def frames_dir(self, eye: str) -> str:
        """
        Pre:
            - eye must be a valid eye string selection token.
        Post:
            - Returns the string path pointing to the source frame storage location for the requested eye.
        """
        return os.path.join(self.eye_root(eye), FRAMES_SUBDIR)

    def masks_dir(self, eye: str, obj: str) -> str:
        """
        Pre:
            - eye must be a valid eye side token.
            - obj must be a valid non-empty target structure string token.
        Post:
            - Returns the string path targeting the output mask folder structure matching the parameters.
        """
        return os.path.join(self.eye_root(eye), MASKS_SUBDIR, obj)

    def visualizations_dir(self, eye: str, obj: str) -> str:
        """
        Pre:
            - eye must be a valid eye side token.
            - obj must be a valid non-empty target structure string token.
        Post:
            - Returns the string path targeting the visualization overlay image directory.
        """
        return os.path.join(self.eye_root(eye), VISUALIZATIONS_SUBDIR, obj)

    def classification_csv(self, eye: str) -> str:
        """
        Pre:
            - eye must be a valid eye side selection string.
        Post:
            - Returns the string destination pathway targeting the eye classification CSV log index file.
        """
        return os.path.join(
            self.eye_root(eye), f"{self.video_name}_{eye}_classification.csv"
        )

    def initial_mask_path(self, eye: str, obj: str) -> str:
        """
        Pre:
            - eye must be a valid eye side token.
            - obj must be a valid non-empty structural element string token.
        Post:
            - Returns the string pathway target configuration leading to the static initial reference mask PNG file.
        """
        from config import INITIAL_MASK_FILENAME
        return os.path.join(self.masks_dir(eye, obj), INITIAL_MASK_FILENAME)

    def last_mask_path(self, eye: str, obj: str) -> str:
        """
        Pre:
            - eye must be a valid eye side token.
            - obj must be a valid non-empty structural element string token.
        Post:
            - Returns the string pathway target configuration leading to the rolling binary checkpoint matrix NPY file.
        """
        from config import LAST_MASK_FILENAME
        return os.path.join(self.masks_dir(eye, obj), LAST_MASK_FILENAME)

    def ensure_structure(self, eye: str, obj: str):
        """
        Pre:
            - eye must be a valid eye side selector token.
            - obj must be a non-empty string representing a tracked target anatomical component.
        Post:
            - Verifies and safely constructs all subdirectories involved in the pipeline workflow loop on disk.
        """
        dirs = [
            self.sessions_dir(),
            self.frames_dir(eye),
            self.masks_dir(eye, obj),
            self.visualizations_dir(eye, obj),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def ensure_frames_dir(self, eye: str):
        """
        Pre:
            - eye must be a valid eye side selection token.
        Post:
            - Verifies and builds the target frames storage folder workspace on disk.
        """
        os.makedirs(self.frames_dir(eye), exist_ok=True)

    @staticmethod
    def _check_eye(eye: str):
        """
        Pre:
            - eye can be any string evaluation parameter.
        Post:
            - Does nothing if eye matches elements contained inside the VALID_EYES definition block.
            - Raises a ValueError if structural validation checks fail.
        """
        if eye not in VALID_EYES:
            raise ValueError(f"eye must be one of {VALID_EYES}, got '{eye}'")

    def frames_exist(self, eye: str) -> bool:
        """
        Pre:
            - eye must be a valid side selection token.
        Post:
            - Returns True if the frames directory contains at least one standard image asset format file.
            - Returns False if the folder path does not exist, is invalid, or contains no frames.
        """
        d = self.frames_dir(eye)
        if not os.path.isdir(d):
            return False
        return any(
            f.lower().endswith((".png", ".jpg", ".jpeg"))
            for f in os.listdir(d)
        )

    def classification_exists(self, eye: str) -> bool:
        """
        Pre:
            - eye must be a valid eye side token selection.
        Post:
            - Returns True if the designated classification log index file exists on disk, False otherwise.
        """
        return os.path.isfile(self.classification_csv(eye))

    def initial_mask_exists(self, eye: str, obj: str) -> bool:
        """
        Pre:
            - eye must be a valid side selector token.
            - obj must be a valid non-empty object name string parameter.
        Post:
            - Returns True if a baseline starting point image is found inside the target output path, False otherwise.
        """
        return os.path.isfile(self.initial_mask_path(eye, obj))
