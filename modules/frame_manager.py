"""
Module: modules/frame_manager.py
Description: Encapsulates operations over a collection of video frame files stored on disk.
             Provides utilities for listing, indexing, frame_id lookup, range filtering,
             and downsampling based on frame classification states.
"""

import os
import re
from typing import Dict, List, Optional, Tuple


_FRAME_RE = re.compile(r"_frame_(\d+)_?([LRlr]?)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


class FrameManager:
    """
    Manages the collection of frames stored within a directory.

    Parameters
    ----------
    frames_dir : str
        Directory containing the frame images.
    """

    def __init__(self, frames_dir: str):
        """
        Pre:
            - frames_dir must be a valid string representing a path to a directory.
        Post:
            - Initializes the FrameManager instance with frames_dir and sets the internal index to None.
        """
        self.frames_dir = frames_dir
        self._index: Optional[Dict[int, str]] = None  

    def _build_index(self):
        """
        Pre:
            - self.frames_dir must exist and be readable.
        Post:
            - Populates the internal self._index dictionary with {frame_id: filename} pairs
              extracted from valid image names matching the _FRAME_RE regex pattern.
        """
        index: Dict[int, str] = {}
        for fname in sorted(os.listdir(self.frames_dir)):
            m = _FRAME_RE.search(fname)
            if m:
                fid = int(m.group(1))
                index[fid] = fname
        self._index = index

    def _get_index(self) -> Dict[int, str]:
        """
        Pre:
            - None.
        Post:
            - Returns the internal index dictionary. Rebuilds it using _build_index if it is None.
        """
        if self._index is None:
            self._build_index()
        return self._index  

    def invalidate(self):
        """
        Pre:
            - None.
        Post:
            - Clears the internal frame index cache, forcing it to be rebuilt on the next query.
        """
        self._index = None

    def all_frame_ids(self) -> List[int]:
        """
        Pre:
            - None.
        Post:
            - Returns a sorted list of all available integer frame IDs in the index.
        """
        return sorted(self._get_index().keys())

    def filename(self, frame_id: int) -> Optional[str]:
        """
        Pre:
            - frame_id must be an integer.
        Post:
            - Returns the string filename corresponding to the provided frame_id if it exists; 
              otherwise, returns None.
        """
        return self._get_index().get(frame_id)

    def full_path(self, frame_id: int) -> Optional[str]:
        """
        Pre:
            - frame_id must be an integer.
        Post:
            - Returns the absolute or relative absolute string path to the file corresponding 
              to frame_id if found; otherwise, returns None.
        """
        fn = self.filename(frame_id)
        return os.path.join(self.frames_dir, fn) if fn else None

    def frames_in_range(
        self, start: int, end: Optional[int]
    ) -> List[Tuple[int, str]]:
        """
        Pre:
            - start must be an integer representing the lower inclusive bound.
            - end can be an integer representing the upper inclusive bound, or None.
        Post:
            - Returns a list of (frame_id, filename) tuples for all indexed frames falling within 
              the range [start, end]. If end is None, includes all frames up to the last index.
        """
        result = []
        for fid in self.all_frame_ids():
            if fid < start:
                continue
            if end is not None and fid > end:
                break
            fn = self._get_index()[fid]
            result.append((fid, fn))
        return result

    def sampled_frames(self, display_fps: int, video_fps: float) -> List[Tuple[int, str]]:
        """
        Pre:
            - display_fps must be an integer (target sampling rate, or -1 for all frames).
            - video_fps must be a float representing the original video frame rate.
        Post:
            - Returns a list of downsampled (frame_id, filename) tuples based on the step size calculated 
              from video_fps and display_fps. If display_fps is -1 or video_fps is <= 0, returns all frames.
        """
        ids = self.all_frame_ids()
        if display_fps == -1 or video_fps <= 0:
            return [(fid, self._get_index()[fid]) for fid in ids]

        step = max(1, int(video_fps / display_fps))
        return [(fid, self._get_index()[fid]) for i, fid in enumerate(ids) if i % step == 0]

    def count(self) -> int:
        """
        Pre:
            - None.
        Post:
            - Returns an integer indicating the total number of indexed frames.
        """
        return len(self._get_index())

    def first_open_from(
        self, start_frame: int, classification: Dict[int, str]
    ) -> Optional[int]:
        """
        Pre:
            - start_frame must be an integer.
            - classification must be a dictionary mapping integer frame IDs to their state string.
        Post:
            - Returns the first integer frame_id greater than or equal to start_frame whose classification 
              state equals 'open'. Returns None if no such frame is found.
        """
        for fid in self.all_frame_ids():
            if fid < start_frame:
                continue
            if classification.get(fid, "open") == "open":
                return fid
        return None
