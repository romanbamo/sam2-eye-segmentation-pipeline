"""
Module: modules/ui/mask_viewer.py
Description: Provides an interactive media playback window to review SAM2 tracking sequences.
             Supports multi-view modes including color-blended overlays, raw binary masks, 
             and side-by-side comparative views with explicit image capturing triggers.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


_KEY_ESC = 27
_KEY_SPACE = ord(" ")
_KEY_LEFT = 81
_KEY_RIGHT = 83
_KEY_M = ord("m")
_KEY_S = ord("s")


class MaskViewer:
    """
    Interactive results visualization player for viewing overlay outputs and raw binary tracking masks.

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        Collection of matching path tuples structured as (visual_path, mask_path).
    window_name : str, optional
        Target rendering identifier title for the OpenCV window.
    fps : float, optional
        Automatic continuous play frame rate (0.0 initializes manual key step mode).
    export_dir : Optional[str], optional
        Destination directory pathway used to save frame screenshots.
    """

    WINDOW = "Mask Preview"

    MODE_OVERLAY = 0    
    MODE_MASK = 1       
    MODE_SIDE = 2       
    MODE_NAMES = ["OVERLAY", "MASK", "SIDE-BY-SIDE"]

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        window_name: str = "Mask Preview",
        fps: float = 0.0,
        export_dir: Optional[str] = None,
    ):
        """
        Pre:
            - pairs must be a non-empty list of path string tuples.
        Post:
            - Initializes index counters, playback control flags, and active view modes.
            - Raises a ValueError if pairs collection parameters are empty.
        """
        if not pairs:
            raise ValueError("pairs list is empty")

        self.pairs = pairs
        self.window_name = window_name
        self.fps = fps
        self.export_dir = export_dir or "."

        self._idx: int = 0
        self._mode: int = self.MODE_OVERLAY
        self._paused: bool = fps == 0.0

    def run(self, start_idx: int = 0):
        """
        Pre:
            - start_idx must be an integer.
        Post:
            - Instantiates the graphics canvas window, attaches control trackbars, and processes frames.
            - Terminate loops and destroys window contexts on ESC keystrokes or on reaching the sequence end.
        """
        self._idx = max(0, min(start_idx, len(self.pairs) - 1))

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "Frame", self.window_name, self._idx, len(self.pairs) - 1,
            self._on_trackbar,
        )

        delay_ms = int(1000 / self.fps) if self.fps > 0 else 30

        while True:
            visual_path, mask_path = self.pairs[self._idx]
            display = self._build_display(visual_path, mask_path)
            self._draw_hud(display, self._idx)

            cv2.imshow(self.window_name, display)
            cv2.setTrackbarPos("Frame", self.window_name, self._idx)

            key = cv2.waitKey(delay_ms) & 0xFF
            stop = self._handle_key(key, visual_path)
            if stop:
                break

            if not self._paused:
                if self._idx < len(self.pairs) - 1:
                    self._idx += 1
                else:
                    self._paused = True  

        cv2.destroyWindow(self.window_name)

    def _build_display(self, visual_path: str, mask_path: str) -> np.ndarray:
        """
        Pre:
            - visual_path and mask_path must be valid string file location lines.
        Post:
            - Compiles and scales loaded canvas structures based on the active viewing profile mode.
            - Returns a single formatted output display numpy array frame.
        """
        visual = self._load_bgr(visual_path)
        mask_gray = self._load_gray(mask_path)

        if self._mode == self.MODE_OVERLAY:
            return visual

        elif self._mode == self.MODE_MASK:
            return cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

        else:  
            mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

            h1, w1 = visual.shape[:2]
            h2, w2 = mask_bgr.shape[:2]
            if h1 != h2:
                mask_bgr = cv2.resize(mask_bgr, (w2, h1))

            sep = np.full((h1, 4, 3), 200, dtype=np.uint8)
            return np.hstack([visual, sep, mask_bgr])

    def _handle_key(self, key: int, visual_path: str) -> bool:
        """
        Pre:
            - key must be an integer, visual_path must be a valid path string.
        Post:
            - Alters playback boundaries or viewport modalities depending on captured layout keycodes.
            - Returns True if an exit request action is caught, False otherwise.
        """
        if key == _KEY_ESC:
            return True

        elif key == _KEY_SPACE:
            self._paused = not self._paused

        elif key == _KEY_RIGHT:
            self._idx = min(self._idx + 1, len(self.pairs) - 1)
            self._paused = True

        elif key == _KEY_LEFT:
            self._idx = max(self._idx - 1, 0)
            self._paused = True

        elif key == _KEY_M:
            self._mode = (self._mode + 1) % 3

        elif key == _KEY_S:
            self._save_screenshot(visual_path)

        return False

    def _on_trackbar(self, pos: int):
        """
        Pre:
            - pos must be a valid non-negative index integer inside self.pairs limits.
        Post:
            - Synchronizes internal frame tracker indices to the trackbar coordinate and sets execution state to pause.
        """
        self._idx = pos
        self._paused = True

    def _draw_hud(self, img: np.ndarray, idx: int):
        """
        Pre:
            - img must be a valid image matrix target, idx must match current frame loop counter.
        Post:
            - Computes and burns status bars, tracking information strings, and instructions blocks directly 
              over the input display matrix.
        """
        status = "PAUSED" if self._paused else "PLAYING"
        mode_name = self.MODE_NAMES[self._mode]
        visual_path, mask_path = self.pairs[idx]

        line1 = (
            f"{status}  [{idx+1}/{len(self.pairs)}]  "
            f"MODE={mode_name}  |  {os.path.basename(visual_path)}"
        )
        line2 = "←/→=step  SPACE=play  M=mode  S=save  ESC=exit"

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        cv2.putText(img, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
        cv2.putText(img, line2, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    def _save_screenshot(self, visual_path: str):
        """
        Pre:
            - visual_path must be a non-empty path string representation.
        Post:
            - Exports current workspace configuration imagery to the designated export target location path.
        """
        base = os.path.splitext(os.path.basename(visual_path))[0]
        out_path = os.path.join(self.export_dir, f"screenshot_{base}.png")
        visual_path_cur, mask_path_cur = self.pairs[self._idx]
        display = self._build_display(visual_path_cur, mask_path_cur)
        cv2.imwrite(out_path, display)
        print(f"[MaskViewer] Screenshot saved → {out_path}")

    @staticmethod
    def _load_bgr(path: str) -> np.ndarray:
        """
        Pre:
            - path must be a string path representation.
        Post:
            - Reads the target image asset file from disk. Returns a fallback black notification box 
              if loading fails.
        """
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                img, f"Missing: {os.path.basename(path)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )
        return img

    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        """
        Pre:
            - path must be a string path representation.
        Post:
            - Reads the grayscale representation matrix from disk. Returns an empty zero fallback matrix 
              on parsing failure.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((240, 320), dtype=np.uint8)
        return img

    @classmethod
    def from_dirs(
        cls,
        visuals_dir: str,
        masks_dir: str,
        window_name: str = "Mask Preview",
        fps: float = 0.0,
        export_dir: Optional[str] = None,
    ) -> "MaskViewer":
        """
        Pre:
            - visuals_dir and masks_dir must target existing project subdirectories.
        Post:
            - Scans filenames recursively matching visual outputs to their corresponding binary tracking mask counterparts.
            - Returns an instantiated MaskViewer class ready to enter run processing loops.
            - Raises a FileNotFoundError if the masks folder is missing, or a ValueError if matching targets fail.
        """
        pairs: List[Tuple[str, str]] = []

        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        for mask_file in sorted(os.listdir(masks_dir)):
            if not mask_file.lower().endswith(".png"):
                continue

            if mask_file.startswith("mask_"):
                base = mask_file[len("mask_"):]
            else:
                base = mask_file

            mask_path = os.path.join(masks_dir, mask_file)

            visual_name = f"final_{base}"
            visual_path = os.path.join(visuals_dir, visual_name)

            if not os.path.isfile(visual_path):
                visual_path = os.path.join(visuals_dir, base)

            pairs.append((visual_path, mask_path))

        if not pairs:
            raise ValueError(
                f"No mask/visual pairs found in:\n  masks={masks_dir}\n  visuals={visuals_dir}"
            )

        return cls(
            pairs=pairs,
            window_name=window_name,
            fps=fps,
            export_dir=export_dir or visuals_dir,
        )
