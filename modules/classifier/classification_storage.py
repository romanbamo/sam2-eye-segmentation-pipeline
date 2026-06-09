"""
Module: modules/classifier/classification_storage.py
Description: Manages the in-memory storage of template frames labeled as 'open' or 'close' 
             during the interactive manual classification session. Provides utilities to 
             query statistics, format contents for serialization, and save template images to disk.
"""

import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np


class ClassificationStorage:
    """
    Stores open/close template images in memory and provides disk synchronization.

    Each template consists of a BGR image array, its frame identifier, and its filename.
    """

    def __init__(self, templates_dir: str):
        """
        Pre:
            - templates_dir must be a valid string representing a path to a directory.
        Post:
            - Initializes an empty storage dictionary and ensures the templates directory exists on disk.
        """
        self.templates_dir = templates_dir
        os.makedirs(templates_dir, exist_ok=True)

        self._store: Dict[int, dict] = {}

    def add(self, frame_id: int, filename: str, img: np.ndarray, state: str):
        """
        Pre:
            - frame_id must be an integer.
            - filename must be a valid non-empty string.
            - img must be a valid numpy.ndarray representing a BGR image.
            - state must be a string exactly matching 'open' or 'close'.
        Post:
            - Inserts or overwrites the template item for the given frame_id in the internal storage structure.
            - Raises a ValueError if the state parameter is invalid.
        """
        if state not in ("open", "close"):
            raise ValueError(f"state must be 'open' or 'close', got '{state}'")
        self._store[frame_id] = {
            "state": state,
            "filename": filename,
            "img": img,
        }

    def remove(self, frame_id: int):
        """
        Pre:
            - frame_id must be an integer.
        Post:
            - Removes the template item associated with frame_id from storage if it exists.
        """
        self._store.pop(frame_id, None)

    def get_by_state(self, state: str) -> List[Tuple[int, str, np.ndarray]]:
        """
        Pre:
            - state must be a string ('open' or 'close').
        Post:
            - Returns a sorted list of tuples containing (frame_id, filename, image_array) for all stored templates 
              matching the specified state.
        """
        return [
            (fid, v["filename"], v["img"])
            for fid, v in sorted(self._store.items())
            if v["state"] == state
        ]

    def open_templates(self) -> List[np.ndarray]:
        """
        Pre:
            - None.
        Post:
            - Returns a list containing the numpy.ndarray images of all templates classified as 'open'.
        """
        return [img for _, _, img in self.get_by_state("open")]

    def close_templates(self) -> List[np.ndarray]:
        """
        Pre:
            - None.
        Post:
            - Returns a list containing the numpy.ndarray images of all templates classified as 'close'.
        """
        return [img for _, _, img in self.get_by_state("close")]

    def count_open(self) -> int:
        """
        Pre:
            - None.
        Post:
            - Returns an integer corresponding to the total count of 'open' templates in memory.
        """
        return sum(1 for v in self._store.values() if v["state"] == "open")

    def count_close(self) -> int:
        """
        Pre:
            - None.
        Post:
            - Returns an integer corresponding to the total count of 'close' templates in memory.
        """
        return sum(1 for v in self._store.values() if v["state"] == "close")

    def all_items(self) -> List[Tuple[int, str, str]]:
        """
        Pre:
            - None.
        Post:
            - Returns a sorted list of tuples containing (frame_id, filename, state) for all stored templates.
        """
        return [
            (fid, v["filename"], v["state"])
            for fid, v in sorted(self._store.items())
        ]

    def is_empty(self) -> bool:
        """
        Pre:
            - None.
        Post:
            - Returns True if the storage contains no templates, False otherwise.
        """
        return len(self._store) == 0

    def to_csv_rows(self) -> List[Tuple[str, int, str]]:
        """
        Pre:
            - None.
        Post:
            - Returns a sorted list of tuples in the format (filename, frame_id, state) ready for CSV serialization.
        """
        return [
            (v["filename"], fid, v["state"])
            for fid, v in sorted(self._store.items())
        ]

    def persist_templates(self):
        """
        Pre:
            - self.templates_dir must be valid, and the internal storage must contain items to save.
        Post:
            - Clears all existing files in self.templates_dir.
            - Writes every image template from memory to disk using the filename pattern 'tmpl_{state}_{frame_id}.png'.
        """
        for f in os.listdir(self.templates_dir):
            fp = os.path.join(self.templates_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

        for fid, v in self._store.items():
            prefix = v["state"]  
            out_name = f"tmpl_{prefix}_{fid:07d}.png"
            cv2.imwrite(os.path.join(self.templates_dir, out_name), v["img"])
