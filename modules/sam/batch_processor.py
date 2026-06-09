"""
Module: modules/sam/batch_processor.py
Description: Manages segmentation mask generation using the SAM2 video predictor over data frames 
             processed in sequential batches. Handles open/close tracking states, rolling update mask 
             matrices, temporary directories management, and GPU VRAM cache flushing operations.
"""

import gc
import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import BATCH_SIZE, BATCH_TEMP_DIR, OVERLAY_ALPHA, OVERLAY_COLOR_RGB
from config import SAM2_CHECKPOINT, SAM2_MODEL_CONFIG


class BatchProcessor:
    """
    Processes SAM2 model inference sequences over multiple segmented frames in localized batches.

    Parameters
    ----------
    frames_dir : str
        Source input directory where extraction image files are located.
    masks_dir : str
        Output destination directory for storing serialized binary mask png files.
    visuals_dir : str
        Output destination directory for saving mask-blended visualization images.
    classification : Dict[int, str]
        Pre-calculated structure mapping frame IDs to eye status tokens ('open'|'close').
    initial_mask : np.ndarray
        Float32 array mask indicating the starting seed region for prediction propagation.
    frame_list : List[Tuple[int, str]]
        Sorted subset collection of (frame_id, filename) elements within the requested range.
    last_mask_path : str
        Output absolute or relative path used to persist rolling tracking arrays between execution jumps.
    device : torch.device, optional
        Target hardware computing context (defaults to CUDA if available, otherwise CPU).
    """

    def __init__(
        self,
        frames_dir: str,
        masks_dir: str,
        visuals_dir: str,
        classification: Dict[int, str],
        initial_mask: np.ndarray,
        frame_list: List[Tuple[int, str]],
        last_mask_path: str,
        device: Optional[torch.device] = None,
    ):
        """
        Pre:
            - frames_dir, masks_dir, visuals_dir, and last_mask_path must be valid path strings.
            - classification must map valid integer IDs to eye status values.
            - initial_mask must be a valid numpy matrix representing the starting region.
            - frame_list must be a sorted collection structure of frames matching directory indices.
        Post:
            - Initializes attributes and guarantees structural directory pathways exist on disk, 
              including the clean temporary initialization folder.
        """
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.visuals_dir = visuals_dir
        self.classification = classification
        self.current_mask_np = initial_mask.copy()
        self.frame_list = frame_list  
        self.last_mask_path = last_mask_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(BATCH_TEMP_DIR, exist_ok=True)

    def run(self, progress_window: bool = True):
        """
        Pre:
            - The internal instance parameters must be correctly configured.
        Post:
            - Iterates sequentially over frame lists slicing them according to defined block sizes 
              and calling individual partition managers.
        """
        total = len(self.frame_list)
        print(f"[Batch] Processing {total} frames in batches of {BATCH_SIZE}.")

        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = self.frame_list[batch_start:batch_end]
            print(f"\n[Batch] Frames {batch[0][0]} → {batch[-1][0]}")
            self._process_batch(batch, progress_window)

        print("[Batch] All batches done.")

    def _process_batch(
        self,
        batch: List[Tuple[int, str]],
        progress_window: bool,
    ):
        """
        Pre:
            - batch must be a slice containing elements structured as (frame_id, filename).
        Post:
            - Separates active frames from closed intervals.
            - For open entries, constructs workspace environments and executes multi-frame tracking logic.
            - For closed entries, generates and exports standard blank empty masks immediately.
            - Serializes updated state arrays and flushes volatile memory blocks before returning.
        """
        self._clear_temp()

        active_indices: List[int] = []  

        for rel_idx, (frame_id, filename) in enumerate(
            tqdm(batch, desc="Filtering (ORB)")
        ):
            full_path = os.path.join(self.frames_dir, filename)
            img = cv2.imread(full_path)
            if img is None:
                continue

            is_open = self.classification.get(frame_id, "close") == "open"

            if is_open:
                new_name = f"{rel_idx:06d}.jpg"
                cv2.imwrite(os.path.join(BATCH_TEMP_DIR, new_name), img)
                active_indices.append(rel_idx)
            else:
                h, w = img.shape[:2]
                zero_mask = np.zeros((h, w), dtype=np.uint8)
                base = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(self.visuals_dir, f"final_{base}.png"), img)
                cv2.imwrite(os.path.join(self.masks_dir, f"mask_{base}.png"), zero_mask)

        if not active_indices:
            print("[Batch] No open frames in this batch, skipping SAM2.")
            return

        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.device
        )
        inference_state = predictor.init_state(
            video_path=BATCH_TEMP_DIR,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            mask=self.current_mask_np,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            rel_idx = active_indices[out_frame_idx]
            frame_id, filename = batch[rel_idx]

            mask_thresholded = out_mask_logits[0] > 0.0
            final_mask_uint8 = (
                mask_thresholded.cpu().numpy().astype(np.uint8) * 255
            ).squeeze()

            if mask_thresholded.any():
                self.current_mask_np = (
                    mask_thresholded.cpu().float().numpy().squeeze()
                )

            full_path = os.path.join(self.frames_dir, filename)
            orig = cv2.imread(full_path)
            base = os.path.splitext(filename)[0]

            if orig is not None:
                visual = self._apply_overlay(orig, final_mask_uint8)
                cv2.imwrite(os.path.join(self.visuals_dir, f"final_{base}.png"), visual)

            cv2.imwrite(os.path.join(self.masks_dir, f"mask_{base}.png"), final_mask_uint8)

            if progress_window and orig is not None:
                cv2.imshow("Inference Progress", visual)
                cv2.waitKey(1)

            del out_mask_logits

        predictor.reset_state(inference_state)
        del inference_state
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        np.save(self.last_mask_path, self.current_mask_np)

    def _clear_temp(self):
        """
        Pre:
            - BATCH_TEMP_DIR path must exist and be accessible.
        Post:
            - Iterates through the directory, removing all individual internal target file files.
        """
        for f in os.listdir(BATCH_TEMP_DIR):
            fp = os.path.join(BATCH_TEMP_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)

    @staticmethod
    def _apply_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Pre:
            - frame must be a valid BGR array structure.
            - mask must be a valid single-channel uint8 binary matrix representation.
        Post:
            - Blends a colored overlay indicator over target coordinates defined by non-zero elements.
            - Returns a new modified numpy array image instance containing the visual details.
        """
        overlay_layer = np.zeros_like(frame)
        overlay_layer[:] = OVERLAY_COLOR_RGB[::-1]  
        blended = cv2.addWeighted(frame, 1 - OVERLAY_ALPHA, overlay_layer, OVERLAY_ALPHA, 0)
        result = frame.copy()
        mask_bool = mask > 0
        result[mask_bool] = blended[mask_bool]
        return result
