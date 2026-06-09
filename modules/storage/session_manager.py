"""
Module: storage/session_manager.py
Description: Handles reading from and writing to the session.json configuration file.
             Maintains state attributes across executions to resume interrupted work, 
             tracking classification status, processed objects, and progress frames.
"""

import json
import os
from typing import Any, Dict, List, Optional


class SessionManager:
    def __init__(self, sessions_dir: str, video_name: str):
        """
        Pre:
            - sessions_dir must be a valid path string representation.
            - video_name must be a valid non-empty string.
        Post:
            - Ensures the sessions directory exists on disk and sets the targeted json file pathway.
            - Loads an existing session state or initializes a new state dictionary mapping structure.
        """
        os.makedirs(sessions_dir, exist_ok=True)
        self._path = os.path.join(sessions_dir, "session.json")
        self._data: Dict[str, Any] = self._load_or_create(video_name)

    def _load_or_create(self, video_name: str) -> Dict[str, Any]:
        """
        Pre:
            - video_name must be a valid non-empty string.
        Post:
            - Returns a dictionary populated with data loaded from the file if self._path exists.
            - Returns a new default session initialization dictionary schema structure if no file is found.
        """
        if os.path.isfile(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Session] Resumed → {self._path}")
            return data
        return {
            "video": video_name,
            "eye": None,
            "classification_done": False,
            "objects_processed": [],
            "last_processed_frame": {},
        }

    def save(self):
        """
        Pre:
            - None.
        Post:
            - Serializes and writes the internal self._data state dictionary mapping configuration 
              into self._path using a 2-space json indentation structure.
        """
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    @property
    def classification_done(self) -> bool:
        """
        Pre:
            - None.
        Post:
            - Returns a boolean indicating whether the open/close classification state process is completed.
        """
        return self._data.get("classification_done", False)

    @classification_done.setter
    def classification_done(self, value: bool):
        """
        Pre:
            - value must be a boolean flag.
        Post:
            - Updates the internal state mapping value for classification completion and triggers save().
        """
        self._data["classification_done"] = value
        self.save()

    @property
    def objects_processed(self) -> List[str]:
        """
        Pre:
            - None.
        Post:
            - Returns a list containing the string labels of all anatomical elements processed so far.
        """
        return self._data.get("objects_processed", [])

    def mark_object_processed(self, obj: str):
        """
        Pre:
            - obj must be a valid non-empty string token representing a target tracking structure.
        Post:
            - Appends the object label token to the internal processing history array if not present 
              and updates the json data store on disk.
        """
        if obj not in self._data["objects_processed"]:
            self._data["objects_processed"].append(obj)
        self.save()

    def get_last_frame(self, obj: str) -> Optional[int]:
        """
        Pre:
            - obj must be a valid non-empty string structure identifier.
        Post:
            - Returns the integer frame ID corresponding to the last fully tracked frame index for the 
              specified object if recorded; otherwise, returns None.
        """
        return self._data.get("last_processed_frame", {}).get(obj)

    def set_last_frame(self, obj: str, frame_id: int):
        """
        Pre:
            - obj must be a valid non-empty string structural label component.
            - frame_id must be a non-negative integer representing the progress checkpoint.
        Post:
            - Records the frame_id number mapping key under the specified object section and triggers save().
        """
        if "last_processed_frame" not in self._data:
            self._data["last_processed_frame"] = {}
        self._data["last_processed_frame"][obj] = frame_id
        self.save()

    @property
    def eye(self) -> Optional[str]:
        """
        Pre:
            - None.
        Post:
            - Returns the string configuration value denoting the active eye side selection ('left'|'right'), 
              or None if unconfigured.
        """
        return self._data.get("eye")

    @eye.setter
    def eye(self, value: str):
        """
        Pre:
            - value must be a string parameter ('left' or 'right').
        Post:
            - Updates the active processing eye tracking property in memory and executes save().
        """
        self._data["eye"] = value
        self.save()
