"""
Module: modules/classifier/classifier_ui.py
Description: Interactive OpenCV-based video player interface designed for manually labeling 
             frames as 'open' or 'close' using keystrokes. Features multi-frame batch range 
             selection, review loops for template modification/deletion, and validation screens.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import KEY_BATCH, KEY_CLOSE, KEY_ESC, KEY_NEXT, KEY_OPEN, KEY_REPEAT
from modules.classifier.classification_storage import ClassificationStorage


class ClassifierUI:
    """
    Interactive player interface for open/close frame classification.

    Parameters
    ----------
    storage : ClassificationStorage
        The data collection store where labeled template items are recorded.
    frames : List[Tuple[int, str, str]]
        Sequence data structures containing tuples of (frame_id, filename, full_path).
    """

    WINDOW = "Video Classifier"

    def __init__(
        self,
        storage: ClassificationStorage,
        frames: List[Tuple[int, str, str]],
    ):
        """
        Pre:
            - storage must be a valid instantiation of ClassificationStorage.
            - frames must be a list containing tuples structured as (frame_id, filename, full_path).
        Post:
            - Initializes the player configuration, resets internal tracking for batch boundaries, 
              and leaves the interface ready to start rendering windows.
        """
        self.storage = storage
        self.frames = frames  

        self._batch_start: Optional[int] = None  
        self._batch_mode: bool = False

    def run(self) -> bool:
        """
        Pre:
            - self.frames must contain a valid population of downsampled frames.
        Post:
            - Manages windows, rendering, and loops until explicit user decision.
            - Returns True if data collection is finalized, saved, and successfully confirmed.
            - Returns False if the session is abandoned, canceled via ESC, or closed without exporting.
        """
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

        while True:
            confirmed = self._run_labeling_pass()
            if confirmed is None:
                cv2.destroyWindow(self.WINDOW)
                return False

            action = self._review_loop()
            if action == "confirm":
                cv2.destroyWindow(self.WINDOW)
                return True
            elif action == "repeat":
                self.storage = ClassificationStorage(self.storage.templates_dir)
                continue
            else:
                cv2.destroyWindow(self.WINDOW)
                return False

    def _run_labeling_pass(self) -> Optional[bool]:
        """
        Pre:
            - The UI window must be initialized and display properties must be ready.
        Post:
            - Iterates sequentially through self.frames, intercepting keyboard user commands.
            - Returns True if iteration completes naturally up to the last available frame.
            - Returns None if the process is explicitly aborted via the ESC key.
        """
        self._batch_start = None
        self._batch_mode = False

        for idx, (frame_id, filename, full_path) in enumerate(self.frames):
            img = cv2.imread(full_path)
            if img is None:
                continue

            self._draw_frame(img, frame_id, filename, idx)

            key = cv2.waitKey(0) & 0xFF

            if key == KEY_ESC:
                return None

            elif key == KEY_OPEN:
                self.storage.add(frame_id, filename, img, "open")
                print(f"[Classifier] OPEN  → {filename}")

            elif key == KEY_CLOSE:
                self.storage.add(frame_id, filename, img, "close")
                print(f"[Classifier] CLOSE → {filename}")

            elif key == KEY_BATCH:
                result = self._handle_batch(idx, img, frame_id, filename)
                if result == "skip":
                    continue

            elif key == KEY_NEXT:
                pass

        return True

    def _handle_batch(
        self,
        idx: int,
        img: np.ndarray,
        frame_id: int,
        filename: str,
    ) -> str:
        """
        Pre:
            - idx must be an integer representing the current frame collection index context.
            - img must be the raw numpy array of the frame currently displayed.
            - frame_id and filename must accurately match the underlying frame item details.
        Post:
            - Toggles the internal batch status flags. Sets limits if it is the opening toggle.
            - Opens submenus and automatically applies states to frames inside the interval if closing.
            - Returns 'skip' if further sequential rendering is paused waiting for inputs, or 
              'done' once range bulk operations are applied.
        """
        if not self._batch_mode:
            self._batch_start = idx
            self._batch_mode = True
            print(f"[Batch] Start set at frame {frame_id}")
            self._draw_frame(img, frame_id, filename, idx, batch_active=True)
            return "skip"

        else:
            batch_end = idx
            print(f"[Batch] End set at frame {frame_id}")

            state = self._ask_batch_state()
            if state is None:
                self._batch_mode = False
                self._batch_start = None
                return "skip"

            start = self._batch_start  
            for b_idx in range(start, batch_end + 1):
                b_frame_id, b_filename, b_path = self.frames[b_idx]
                b_img = cv2.imread(b_path)
                if b_img is not None:
                    self.storage.add(b_frame_id, b_filename, b_img, state)

            print(
                f"[Batch] Applied '{state}' to {batch_end - start + 1} frames "
                f"({self.frames[start][0]} → {self.frames[batch_end][0]})"
            )

            self._batch_mode = False
            self._batch_start = None
            return "done"

    def _ask_batch_state(self) -> Optional[str]:
        """
        Pre:
            - Display output targets must be functional.
        Post:
            - Displays a prompt on an empty temporary canvas window, blocking until a choice is selected.
            - Returns 'open' or 'close' matching the chosen configuration string, or 
              None if canceled via ESC.
        """
        blank = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Batch: press O = OPEN  |  C = CLOSE  |  ESC = cancel",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
        )
        cv2.imshow(self.WINDOW, blank)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == KEY_OPEN:
                return "open"
            elif key == KEY_CLOSE:
                return "close"
            elif key == KEY_ESC:
                return None

    def _review_loop(self) -> str:
        """
        Pre:
            - Storage must contain the records gathered during the initial manual phase.
        Post:
            - Processes custom inputs during multi-item inspection screens, enabling deletion.
            - Returns a navigation string action token ('confirm' | 'repeat' | 'exit') matching the choice.
        """
        print(
            f"\n[Review] Open: {self.storage.count_open()} | "
            f"Close: {self.storage.count_close()}"
        )

        items = self.storage.all_items()
        review_idx = 0

        while True:
            if not items:
                action = self._show_empty_review()
                return action

            fid, fname, state = items[review_idx]
            full_path = self._find_path(fname)
            img = cv2.imread(full_path) if full_path else None
            display = img if img is not None else np.zeros((300, 400, 3), np.uint8)

            label = f"[{review_idx+1}/{len(items)}] {fname}  STATE={state.upper()}"
            overlay = display.copy()
            cv2.putText(
                overlay, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
            )
            cv2.putText(
                overlay,
                "DEL=delete | N=next | P=prev | ENTER=confirm | R=repeat | ESC=exit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )
            cv2.imshow(self.WINDOW, overlay)

            key = cv2.waitKey(0) & 0xFF

            if key in (ord("n"), KEY_NEXT):
                review_idx = (review_idx + 1) % len(items)
            elif key == ord("p"):
                review_idx = (review_idx - 1) % len(items)
            elif key == ord("\r") or key == 13:  
                return "confirm"
            elif key == KEY_REPEAT:
                return "repeat"
            elif key == KEY_ESC:
                return "exit"
            elif key == ord("d") or key == 127:  
                self.storage.remove(fid)
                items = self.storage.all_items()
                review_idx = min(review_idx, max(0, len(items) - 1))
                print(f"[Review] Deleted frame {fid}")

    def _show_empty_review(self) -> str:
        """
        Pre:
            - None.
        Post:
            - Renders a black fallback banner screen indicating that no templates exist.
            - Returns 'repeat' or 'exit' based on the caught structural keyboard choice.
        """
        blank = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(
            blank, "No templates. Press R to repeat or ESC to exit.",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1,
        )
        cv2.imshow(self.WINDOW, blank)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == KEY_REPEAT:
                return "repeat"
            elif key == KEY_ESC:
                return "exit"

    def _draw_frame(
        self,
        img: np.ndarray,
        frame_id: int,
        filename: str,
        idx: int,
        batch_active: bool = False,
    ):
        """
        Pre:
            - img must be a valid BGR array structure.
            - frame_id, filename, and idx details must accurately correspond to current loop metadata.
        Post:
            - Renders text guidelines, frame position identifiers, and item counts directly over 
              the active image view.
        """
        display = img.copy()
        status = f"[{idx+1}/{len(self.frames)}] Frame {frame_id}"
        if batch_active:
            status += "  [BATCH START]"
        n_open = self.storage.count_open()
        n_close = self.storage.count_close()
        info = f"O={n_open}  C={n_close}  |  O=open  C=close  B=batch  N=skip  ESC=done"

        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(display, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(self.WINDOW, display)

    def _find_path(self, filename: str) -> Optional[str]:
        """
        Pre:
            - filename must be a non-empty string.
        Post:
            - Searches inside the self.frames index parameter collection structure.
            - Returns the string path matching the filename if found, otherwise returns None.
        """
        for _, fn, fp in self.frames:
            if fn == filename:
                return fp
        return None
