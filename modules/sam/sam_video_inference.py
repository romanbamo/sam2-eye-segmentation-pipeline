"""
Module: modules/sam/sam_video_inference.py
Description: Orchestrates the entire SAM2 video tracking pipeline over a sequence of video frames.
             Acts as the intermediary coordination layer between main.py and BatchProcessor by
             resolving start boundaries, managing initial seed masks, delegating batch segments,
             and providing diagnostic performance tracking statistics.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import SAM2_CHECKPOINT, SAM2_MODEL_CONFIG
from modules.frame_manager import FrameManager
from modules.sam.batch_processor import BatchProcessor
from modules.sam.manual_mask_creator import ManualMaskCreator
from modules.sam.mask_validator import MaskValidator
from modules.storage.folder_manager import FolderManager
from modules.ui.mask_viewer import MaskViewer


class SAMVideoInference:
    """
    Coordinates and executes the full multi-frame SAM2 eye structure tracking loop.

    Parameters
    ----------
    video_name : str
        The workspace identifier or folder name of the target video.
    eye : str
        Target eye selector token string ('left' | 'right').
    obj : str
        The structural target element label to segment ('iris', 'pupil', etc.).
    start_frame : int
        The inclusive lower index limit to start or resume pipeline operations.
    end_frame : Optional[int]
        The inclusive upper limit frame index, or None to process until the last available frame.
    classification : Dict[int, str]
        Pre-calculated matching structure dictionary connecting frame IDs to states ('open'|'close').
    folder : FolderManager
        Project paths utility instance helper.
    device : torch.device, optional
        Target hardware compute accelerator reference.
    """

    def __init__(
        self,
        video_name: str,
        eye: str,
        obj: str,
        start_frame: int,
        end_frame: Optional[int],
        classification: Dict[int, str],
        folder: FolderManager,
        device: Optional[torch.device] = None,
    ):
        """
        Pre:
            - video_name, eye, and obj must be non-empty strings.
            - start_frame must be a non-negative integer.
            - classification must be a mapping of frame indices to state strings.
            - folder must be a valid instance of FolderManager.
        Post:
            - Initializes target metadata attributes and instantiates the internal FrameManager cache.
        """
        self.video_name = video_name
        self.eye = eye
        self.obj = obj
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.classification = classification
        self.folder = folder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._frame_mgr = FrameManager(folder.frames_dir(eye))

    def run(self, show_progress: bool = True) -> bool:
        """
        Pre:
            - The initialization structures must contain paths to valid frame collections.
        Post:
            - Verifies sequence limits, sets up a seed mask matrix, and transfers tracking loop execution 
              to BatchProcessor.
            - Destroys active visualization windows on completion.
            - Returns True if the batch execution run succeeds, False if tracking is aborted or empty.
        """
        all_ids = self._frame_mgr.all_frame_ids()
        if not all_ids:
            print(f"[SAMVideoInference] No frames found in {self.folder.frames_dir(self.eye)}")
            return False

        effective_end = self.end_frame if self.end_frame is not None else all_ids[-1]
        print(
            f"[SAMVideoInference] Range: frame {self.start_frame} → {effective_end} "
            f"| eye={self.eye} | obj={self.obj} | device={self.device}"
        )

        initial_mask = self._resolve_initial_mask()
        if initial_mask is None:
            print("[SAMVideoInference] Cannot obtain initial mask. Aborting inference.")
            return False

        frame_list = self._frame_mgr.frames_in_range(self.start_frame, effective_end)
        if not frame_list:
            print(
                f"[SAMVideoInference] No frames in range [{self.start_frame}, {effective_end}]."
            )
            return False

        print(f"[SAMVideoInference] {len(frame_list)} frames to process.")

        processor = BatchProcessor(
            frames_dir=self.folder.frames_dir(self.eye),
            masks_dir=self.folder.masks_dir(self.eye, self.obj),
            visuals_dir=self.folder.visualizations_dir(self.eye, self.obj),
            classification=self.classification,
            initial_mask=initial_mask,
            frame_list=frame_list,
            last_mask_path=self.folder.last_mask_path(self.eye, self.obj),
            device=self.device,
        )
        processor.run(progress_window=show_progress)

        if show_progress:
            cv2.destroyWindow("Inference Progress")

        return True

    def preview_results(self, n_frames: int = 20):
        """
        Pre:
            - n_frames must be a positive integer indicating the maximum display item sample limit.
        Post:
            - Searches output locations, builds synchronized path tuples connecting visualization overlays 
              to structural masks, and launches the interactive MaskViewer interface.
        """
        masks_dir = self.folder.masks_dir(self.eye, self.obj)
        visuals_dir = self.folder.visualizations_dir(self.eye, self.obj)

        mask_files = sorted(
            f for f in os.listdir(masks_dir) if f.lower().endswith(".png")
        )[:n_frames]

        if not mask_files:
            print("[SAMVideoInference] No masks to preview yet.")
            return

        pairs: List[Tuple[str, str]] = []
        for mf in mask_files:
            mask_path = os.path.join(masks_dir, mf)
            base = mf[len("mask_"):]  
            visual_path = os.path.join(visuals_dir, f"final_{base}")
            if os.path.isfile(visual_path):
                pairs.append((visual_path, mask_path))

        viewer = MaskViewer(pairs)
        viewer.run()

    def _resolve_initial_mask(self) -> Optional[np.ndarray]:
        """
        Pre:
            - Storage systems and classification properties must be operational.
        Post:
            - Selects a seed mask based on hierarchical matching priorities:
                1. Loads last_mask.npy rolling arrays to resume an interrupted pipeline.
                2. Loads existing static initial_mask.png files from disk.
                3. Triggers the interactive ManualMaskCreator tool on the first available open frame.
            - Returns a valid binary float32 numpy tracking canvas matrix, or None if creation is aborted.
        """
        last = MaskValidator.load_npy(self.folder.last_mask_path(self.eye, self.obj))
        if last is not None:
            print(
                f"[SAMVideoInference] Resuming with rolling mask: "
                f"{self.folder.last_mask_path(self.eye, self.obj)}"
            )
            return last

        if self.folder.initial_mask_exists(self.eye, self.obj):
            print(
                f"[SAMVideoInference] Loading existing initial mask: "
                f"{self.folder.initial_mask_path(self.eye, self.obj)}"
            )
            return MaskValidator.load_png(self.folder.initial_mask_path(self.eye, self.obj))

        annotation_fid = self._frame_mgr.first_open_from(
            self.start_frame, self.classification
        )
        if annotation_fid is None:
            print(
                f"[SAMVideoInference] No open frame found >= {self.start_frame}."
            )
            return None

        if self.classification.get(self.start_frame, "open") == "close":
            print(
                f"[SAMVideoInference] Frame {self.start_frame} is CLOSE → "
                f"annotating on first OPEN frame: {annotation_fid}"
            )

        image_path = os.path.join(
            self.folder.frames_dir(self.eye),
            self._frame_mgr.filename(annotation_fid),  
        )

        creator = ManualMaskCreator(
            output_path=self.folder.initial_mask_path(self.eye, self.obj),
            device=self.device,
        )
        print(f"[SAMVideoInference] Annotate initial mask on frame {annotation_fid}...")
        return creator.run(image_path)

    def reset_mask(self):
        """
        Pre:
            - Project directories must be accessible.
        Post:
            - Deletes rolling tracking .npy matrices and static baseline initial mask images from disk, 
              forcing a manual re-annotation step on the next run.
        """
        for path in [
            self.folder.last_mask_path(self.eye, self.obj),
            self.folder.initial_mask_path(self.eye, self.obj),
        ]:
            if os.path.isfile(path):
                os.remove(path)
                print(f"[SAMVideoInference] Removed: {path}")

    def get_statistics(self) -> dict:
        """
        Pre:
            - Classification tables and target output destination directories must be initialized.
        Post:
            - Tallies processed frames, group states within the range, and matching output files on disk.
            - Returns a summary dictionary containing counts for: masks_generated, visuals_generated, 
              open_frames, close_frames, and total_in_range.
        """
        masks_dir = self.folder.masks_dir(self.eye, self.obj)
        visuals_dir = self.folder.visualizations_dir(self.eye, self.obj)

        effective_end = self.end_frame if self.end_frame is not None else -1
        frame_list = self._frame_mgr.frames_in_range(self.start_frame, effective_end if effective_end != -1 else None)

        open_count = sum(
            1 for fid, _ in frame_list if self.classification.get(fid, "close") == "open"
        )
        close_count = len(frame_list) - open_count

        masks_count = len([
            f for f in os.listdir(masks_dir) if f.endswith(".png")
        ]) if os.path.isdir(masks_dir) else 0

        visuals_count = len([
            f for f in os.listdir(visuals_dir) if f.endswith(".png")
        ]) if os.path.isdir(visuals_dir) else 0

        return {
            "masks_generated": masks_count,
            "visuals_generated": visuals_count,
            "open_frames": open_count,
            "close_frames": close_count,
            "total_in_range": len(frame_list),
        }
