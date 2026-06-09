"""
Module: modules/sam/manual_mask_creator.py
Description: Provides an interactive GUI for generating initial segmentation reference masks 
             using SAM2ImagePredictor. Captures manual user mouse clicks to assign positive 
             or negative coordinates, infers three candidate masks, and prompts selection.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from config import SAM2_CHECKPOINT, SAM2_MODEL_CONFIG


class ManualMaskCreator:
    """
    Creates an initial binary reference mask via interactive coordinate annotations.

    Parameters
    ----------
    output_path : str
        The absolute or relative path where the final accepted mask PNG file will be stored.
    device : torch.device, optional
        Target hardware computing infrastructure context (defaults to CUDA if available, otherwise CPU).
    """

    WIN_ANNOTATION = "Mask Annotation"
    WIN_PREVIEW = "Mask Preview"

    def __init__(self, output_path: str, device: Optional[torch.device] = None):
        """
        Pre:
            - output_path must be a valid path string representing the destination directory and filename.
        Post:
            - Initializes the ManualMaskCreator state tracking configuration.
        """
        self.output_path = output_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self, image_path: str) -> Optional[np.ndarray]:
        """
        Pre:
            - image_path must point to an existing and readable image file.
        Post:
            - Launches the annotation and decision loops.
            - Serializes the selected mask as a uint8 image to disk.
            - Returns a float32 binary matrix (0.0 or 1.0) on selection confirmation, 
              or None if canceled.
            - Raises a FileNotFoundError if the target file cannot be loaded.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        while True:
            clicks, cancelled = self._collect_clicks(img)
            if cancelled:
                return None

            if not clicks:
                print("[MaskCreator] No points provided, please click at least once.")
                continue

            masks, scores = self._run_sam2(img, clicks)
            chosen = self._show_masks_and_pick(img, masks, scores)

            if chosen is not None:
                mask_uint8 = (chosen * 255).astype(np.uint8)
                import os
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                cv2.imwrite(self.output_path, mask_uint8)
                print(f"[MaskCreator] Mask saved → {self.output_path}")
                return chosen.astype(np.float32)

    def _collect_clicks(
        self, img: np.ndarray
    ) -> Tuple[dict, bool]:
        """
        Pre:
            - img must be a valid numpy matrix representing the reference canvas frame.
        Post:
            - Opens an interactive OpenCV window and monitors click placements.
            - Returns a dictionary mapping coordinates (x, y) to flags (1=positive, 0=negative) 
              and a boolean cancellation marker.
        """
        display = img.copy()
        clicks: dict = {}
        font = cv2.FONT_HERSHEY_TRIPLEX

        def mouse_callback(event, x, y, flags, param):
            nonlocal display
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_clicks[(x, y)] = "pending"

        pending_clicks: dict = {}  

        cv2.namedWindow(self.WIN_ANNOTATION, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN_ANNOTATION, mouse_callback)

        print(
            "[MaskCreator] Click on the image, then press:\n"
            "  P → mark last click as POSITIVE\n"
            "  N → mark last click as NEGATIVE\n"
            "  Any other key → run SAM2\n"
            "  ESC → cancel"
        )

        last_click: Optional[Tuple[int, int]] = None

        def mouse_cb(event, x, y, flags, param):
            nonlocal last_click
            if event == cv2.EVENT_LBUTTONDOWN:
                last_click = (x, y)

        cv2.setMouseCallback(self.WIN_ANNOTATION, mouse_cb)

        while True:
            cv2.imshow(self.WIN_ANNOTATION, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  
                cv2.destroyWindow(self.WIN_ANNOTATION)
                return {}, True

            if last_click is not None:
                lx, ly = last_click
                last_click = None

                self._draw_text(display, f"Click at ({lx},{ly}) → press P=pos / N=neg", (10, 30))
                cv2.imshow(self.WIN_ANNOTATION, display)
                label_key = cv2.waitKey(0) & 0xFF

                if label_key == ord("p"):
                    clicks[(lx, ly)] = 1
                    cv2.putText(display, "+", (lx - 7, ly + 5), font, 0.5, (0, 255, 0), 1)
                    print(f"  + positive ({lx},{ly})")
                elif label_key == ord("n"):
                    clicks[(lx, ly)] = 0
                    cv2.putText(display, "-", (lx - 7, ly + 5), font, 0.5, (0, 0, 255), 1)
                    print(f"  - negative ({lx},{ly})")
                continue

            if key != 255 and key != 0xFF and key > 0:
                cv2.destroyWindow(self.WIN_ANNOTATION)
                return clicks, False

        cv2.destroyWindow(self.WIN_ANNOTATION)
        return clicks, False

    def _run_sam2(
        self, img: np.ndarray, clicks: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre:
            - img must be a valid BGR image matrix.
            - clicks must be a dictionary filled with at least one coordinate point mapping token.
        Post:
            - Initializes SAM2ImagePredictor and computes structural inference sets.
            - Returns a tuple containing a multi-mask numpy array and their confidence values.
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print("[MaskCreator] Running SAM2...")
        predictor = SAM2ImagePredictor(
            build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.device)
        )
        predictor.set_image(img)

        input_points = np.array(list(clicks.keys()))
        input_labels = np.array(list(clicks.values()))

        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        del predictor
        return masks, scores

    def _show_masks_and_pick(
        self, img: np.ndarray, masks: np.ndarray, scores: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Pre:
            - img must be a valid source image array.
            - masks must contain three valid binary canvas layers generated by the predictor.
            - scores must contain three confidence float numbers.
        Post:
            - Renders a combined horizontal preview panel window detailing performance outputs.
            - Returns the chosen mask structure as a float32 array, or None if retrying.
        """
        cv2.namedWindow(self.WIN_PREVIEW, cv2.WINDOW_NORMAL)

        previews = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            overlay = self._make_overlay(img, mask)
            label = f"Mask {i+1}  score={score:.3f}"
            cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            previews.append(overlay)

        strip = np.hstack(previews)
        cv2.putText(
            strip,
            "Press 1 / 2 / 3 to accept    N to retry",
            (10, strip.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1,
        )
        cv2.imshow(self.WIN_PREVIEW, strip)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("1"):
                cv2.destroyWindow(self.WIN_PREVIEW)
                return masks[0].astype(np.float32)
            elif key == ord("2"):
                cv2.destroyWindow(self.WIN_PREVIEW)
                return masks[1].astype(np.float32)
            elif key == ord("3"):
                cv2.destroyWindow(self.WIN_PREVIEW)
                return masks[2].astype(np.float32)
            elif key in (ord("n"), ord("N")):
                cv2.destroyWindow(self.WIN_PREVIEW)
                return None

    @staticmethod
    def _make_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Pre:
            - img must be a valid BGR array image.
            - mask must be a valid single-channel binary matrix.
        Post:
            - Returns a blended image overlay matching non-zero location components.
        """
        from config import OVERLAY_COLOR_RGB, OVERLAY_ALPHA
        overlay_layer = np.zeros_like(img)
        overlay_layer[:] = OVERLAY_COLOR_RGB[::-1]  
        blended = cv2.addWeighted(img, 1 - OVERLAY_ALPHA, overlay_layer, OVERLAY_ALPHA, 0)
        result = img.copy()
        mask_bool = mask > 0
        result[mask_bool] = blended[mask_bool]
        return result

    @staticmethod
    def _draw_text(img: np.ndarray, text: str, pos: Tuple[int, int]):
        """
        Pre:
            - img must be a valid image matrix destination canvas.
            - text must be a string description line.
        Post:
            - Draws the text guidelines at the specified canvas position.
        """
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
