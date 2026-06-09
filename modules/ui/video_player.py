"""
Module: modules/ui/video_player.py
Description: A generic OpenCV-based image sequence and video player wrapper. Handles
             sequential frame rendering, trackbar updates, and standard playback navigation 
             while forwarding unhandled keystrokes to a custom operational user callback.
"""

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


_KEY_SPACE = ord(" ")
_KEY_LEFT = 81   
_KEY_RIGHT = 83  
_KEY_HOME = 80
_KEY_END = 87
_KEY_ESC = 27

_KEY_LEFT_WIN = 2424832
_KEY_RIGHT_WIN = 2555904


class VideoPlayer:
    """
    OpenCV media loop wrapper for rendering a structured list of image frames.

    Parameters
    ----------
    frames : List[Tuple[int, str]]
        Sorted collection sequence containing tuples formatted as (frame_id, full_path).
    window_name : str, optional
        Target structural layout identification title for the rendering interface.
    fps : float, optional
        Target processing frame rate speed (0.0 triggers strict manual key stepping).
    loop : bool, optional
        If True, resets the player index back to zero upon hitting the final frame bound.
    overlay_fn : Optional[Callable], optional
        Custom canvas draw hook parameter structured as (img, frame_id, idx) -> img.
    key_callback : Optional[Callable], optional
        Custom input hook structured as (key, frame_id, idx) -> bool.
    """

    WINDOW = "Video Player"

    def __init__(
        self,
        frames: List[Tuple[int, str]],
        window_name: str = "Video Player",
        fps: float = 0.0,
        loop: bool = False,
        overlay_fn: Optional[Callable] = None,
        key_callback: Optional[Callable] = None,
    ):
        """
        Pre:
            - frames must be a non-empty list of (frame_id, full_path) tuples.
        Post:
            - Configures class properties and tracking variables, ready to allocate windows.
            - Raises a ValueError if the input frames collection dataset is empty.
        """
        if not frames:
            raise ValueError("frames list is empty")

        self.frames = frames
        self.window_name = window_name
        self.fps = fps
        self.loop = loop
        self.overlay_fn = overlay_fn
        self.key_callback = key_callback

        self._idx: int = 0
        self._paused: bool = fps == 0.0
        self._running: bool = False

    def run(self, start_idx: int = 0):
        """
        Pre:
            - self.frames must be validated and fully accessible.
        Post:
            - Builds the window canvas interface layout, registers interactive trackbars, 
              and enters the playback loop.
            - Blocks execution until an exit key command sequence fires or loop boundaries close.
        """
        self._idx = max(0, min(start_idx, len(self.frames) - 1))
        self._running = True

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar(
            "Frame", self.window_name, self._idx, len(self.frames) - 1,
            self._on_trackbar,
        )

        delay_ms = int(1000 / self.fps) if self.fps > 0 else 30

        while self._running:
            frame_id, path = self.frames[self._idx]
            img = self._load(path)
            display = self._apply_overlay(img, frame_id, self._idx)
            self._draw_hud(display, frame_id, self._idx)

            cv2.imshow(self.window_name, display)
            cv2.setTrackbarPos("Frame", self.window_name, self._idx)

            key = cv2.waitKey(delay_ms) & 0xFF

            stop = self._handle_key(key, frame_id)
            if stop:
                break

            if not self._paused:
                self._advance()

        cv2.destroyWindow(self.window_name)

    def seek(self, idx: int):
        """
        Pre:
            - idx must be an integer.
        Post:
            - Clamps and updates the tracking workspace index location counter within active limits.
        """
        self._idx = max(0, min(idx, len(self.frames) - 1))

    def seek_by_frame_id(self, frame_id: int):
        """
        Pre:
            - frame_id must be an integer target tracking label.
        Post:
            - Performs a linear search scan. Updates internal index markers matching frame_id if found.
        """
        for i, (fid, _) in enumerate(self.frames):
            if fid == frame_id:
                self._idx = i
                return
        print(f"[VideoPlayer] frame_id {frame_id} not found.")

    def current_frame_id(self) -> int:
        """
        Pre:
            - None.
        Post:
            - Returns the integer database sequence frame identifier at the active index location.
        """
        return self.frames[self._idx][0]

    def current_idx(self) -> int:
        """
        Pre:
            - None.
        Post:
            - Returns the active integer internal tracking loop sequence array index location.
        """
        return self._idx

    def _handle_key(self, key: int, frame_id: int) -> bool:
        """
        Pre:
            - key and frame_id must be integers.
        Post:
            - Evaluates internal navigation shortcut keystrokes (toggling play states or index shifts).
            - Delegates unhandled inputs to self.key_callback if registered.
            - Returns True if an explicit exit signal token command intercepts tracking, False otherwise.
        """
        if key == _KEY_ESC:
            return True

        elif key == _KEY_SPACE:
            self._paused = not self._paused

        elif key == _KEY_RIGHT:
            self._advance()
            self._paused = True

        elif key == _KEY_LEFT:
            self._rewind()
            self._paused = True

        elif key == _KEY_HOME:
            self._idx = 0
            self._paused = True

        elif key == _KEY_END:
            self._idx = len(self.frames) - 1
            self._paused = True

        elif key != 255 and self.key_callback is not None:
            return self.key_callback(key, frame_id, self._idx)

        return False

    def _advance(self):
        """
        Pre:
            - None.
        Post:
            - Increments the internal data structure sequence frame array layout tracker index counter.
            - Handles wrapping boundaries if loop options are active, or sets running status flags to False.
        """
        if self._idx < len(self.frames) - 1:
            self._idx += 1
        elif self.loop:
            self._idx = 0
        else:
            self._running = False

    def _rewind(self):
        """
        Pre:
            - None.
        Post:
            - Decrements internal array list indices tracking variables, clamping bounds safely at zero.
        """
        self._idx = max(0, self._idx - 1)

    def _on_trackbar(self, pos: int):
        """
        Pre:
            - pos must be a valid non-negative integer within active list limits.
        Post:
            - Relocates current rendering canvas indexes to match the trackbar slider, forcing a pause.
        """
        self._idx = pos
        self._paused = True

    def _load(self, path: str) -> np.ndarray:
        """
        Pre:
            - path must be a string file location template line.
        Post:
            - Reads image assets from files. Returns an error box array frame matrix on parsing failure.
        """
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                img, f"Missing: {path}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )
        return img

    def _apply_overlay(
        self, img: np.ndarray, frame_id: int, idx: int
    ) -> np.ndarray:
        """
        Pre:
            - img must be a valid image matrix representation array.
            - frame_id and idx must be structural integer contextual tracking data values.
        Post:
            - Forwards elements to custom draw callback functions if a hook is active.
            - Returns the annotated numpy array layout frame.
        """
        if self.overlay_fn is not None:
            return self.overlay_fn(img, frame_id, idx)
        return img

    def _draw_hud(self, img: np.ndarray, frame_id: int, idx: int):
        """
        Pre:
            - img must be a valid BGR canvas layout destination array target.
            - frame_id and idx must reflect the active tracking position parameters.
        Post:
            - Burns text strings containing indexes, tracking counters, state variables, 
              and guidance bars directly over the input matrix.
        """
        status = "PAUSED" if self._paused else "PLAYING"
        text = f"{status}  [{idx+1}/{len(self.frames)}]  frame_id={frame_id}"
        hint = "SPACE=pause  ←/→=step  HOME/END  ESC=exit"

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(img, hint, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
