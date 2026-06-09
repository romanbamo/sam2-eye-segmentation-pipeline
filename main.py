"""
Module: main.py
Description: Main entry point for the SAM2 Eye Segmentation Pipeline. 
             Handles frame extraction, interactive open/close eye classification, 
             and SAM2-based mask generation and propagation for specific eye structures.
"""

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch

import config
from modules.frame_extractor import FrameExtractor
from modules.frame_manager import FrameManager
from modules.classifier.classification_storage import ClassificationStorage
from modules.classifier.classifier_ui import ClassifierUI
from modules.classifier.orb_classifier import ORBClassifier
from modules.sam.manual_mask_creator import ManualMaskCreator
from modules.sam.mask_validator import MaskValidator
from modules.sam.batch_processor import BatchProcessor
from modules.sam.sam_video_inference import SAMVideoInference
from modules.storage.folder_manager import FolderManager
from modules.storage.csv_manager import CSVManager
from modules.storage.session_manager import SessionManager
from modules.ui.video_player import VideoPlayer
from modules.ui.mask_viewer import MaskViewer


def parse_args() -> argparse.Namespace:
    """
    Pre:
        - None.
    Post:
        - Returns an argparse.Namespace object containing the parsed command-line arguments:
          video_name (str), object (str), start_frame (int), end_frame (int), and eye (str).
    """
    p = argparse.ArgumentParser(
        description="SAM2 Eye Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video_name", help="Video name / directory name (no extension)")
    p.add_argument("object", help="Object to segment: iris | pupil | conjunctiva | ...")
    p.add_argument("start_frame", type=int, help="Start frame (0 = beginning)")
    p.add_argument("end_frame", type=int, help="End frame (-1 = end of video)")
    p.add_argument(
        "eye",
        choices=["left", "right", "both"],
        help="Eye side to process",
    )
    return p.parse_args()


def find_video_file(video_name: str) -> Optional[str]:
    """
    Pre:
        - video_name must be a valid string representing the name or path of the video without extension.
    Post:
        - Returns the path to the valid video file as a string if found with common extensions 
          in the local or current working directory.
        - Returns None if no matching video file is found.
    """
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]
    for ext in extensions:
        candidate = video_name + ext
        if os.path.isfile(candidate):
            return candidate
        candidate2 = os.path.join(os.getcwd(), os.path.basename(video_name) + ext)
        if os.path.isfile(candidate2):
            return candidate2
    return None


def get_video_fps(video_path: str) -> float:
    """
    Pre:
        - video_path must be a string pointing to an existing and accessible video file.
    Post:
        - Returns a float representing the frames per second (FPS) of the video.
        - Returns 25.0 if the FPS cannot be determined or is less than or equal to 0.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def run_pipeline_for_eye(
    eye: str,
    args: argparse.Namespace,
    folder: FolderManager,
    session: SessionManager,
    video_path: Optional[str],
):
    """
    Pre:
        - eye must be a string representing the side ('left' or 'right').
        - args must be an argparse.Namespace object containing valid pipeline arguments.
        - folder must be an instance of FolderManager.
        - session must be an instance of SessionManager.
        - video_path can be a string pointing to the video file or None if frames already exist.
    Post:
        - Executes the full pipeline for the specified eye: manages folder structures, extracts 
          frames if missing, performs open/close state classification, runs SAM2 video inference, 
          and handles results and optional preview visualization.
    """

    print(f"\n{'='*60}")
    print(f"  EYE: {eye.upper()}  |  OBJECT: {args.object}")
    print(f"{'='*60}")

    folder.ensure_structure(eye, args.object)

    if folder.frames_exist(eye):
        print(f"[Step 1] Frames already exist in {folder.frames_dir(eye)} — skipping extraction.")
    else:
        if video_path is None:
            sys.exit(
                f"[Error] No video file found for '{args.video_name}'. "
                "Please provide the video in the current directory."
            )
        print(f"[Step 1] Extracting frames from {video_path} ...")

        extractor = FrameExtractor(
            video_path=video_path,
            video_name=args.video_name,
            dir_left=folder.frames_dir("left"),
            dir_right=folder.frames_dir("right"),
            start_sec=0,
            end_sec=None,
            samples_per_second=-1,
        )
        extractor.extract()

    frame_mgr = FrameManager(folder.frames_dir(eye))
    all_ids = frame_mgr.all_frame_ids()

    if not all_ids:
        print(f"[Error] No frames found in {folder.frames_dir(eye)}")
        return

    end_frame = args.end_frame if args.end_frame != -1 else all_ids[-1]
    print(f"[Info] Processing frames {args.start_frame} → {end_frame} ({eye})")

    csv_path = folder.classification_csv(eye)

    if folder.classification_exists(eye):
        print(f"[Step 3] Classification CSV found — loading {csv_path}")
        classification = CSVManager.read_as_dict(csv_path)
    else:
        print(f"[Step 3] Starting interactive classification for {eye} eye...")
        classification = run_classification(eye, frame_mgr, folder, video_path)
        CSVManager.write(csv_path, _build_csv_rows(frame_mgr, classification))
        print(f"[Step 3] Classification saved → {csv_path}")
        session.classification_done = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step 4+5] Launching SAM2 inference on {device} ...")

    inference = SAMVideoInference(
        video_name=args.video_name,
        eye=eye,
        obj=args.object,
        start_frame=args.start_frame,
        end_frame=end_frame if end_frame != all_ids[-1] or args.end_frame != -1 else None,
        classification=classification,
        folder=folder,
        device=device,
    )

    success = inference.run(show_progress=True)

    if not success:
        print("[Error] SAM2 inference failed or was cancelled.")
        return

    session.mark_object_processed(args.object)
    session.set_last_frame(args.object, end_frame)

    stats = inference.get_statistics()
    print(
        f"\n[Summary] eye={eye} | obj={args.object}\n"
        f"  Masks generated : {stats['masks_generated']}\n"
        f"  Visuals         : {stats['visuals_generated']}\n"
        f"  Open frames     : {stats['open_frames']}\n"
        f"  Close frames    : {stats['close_frames']}\n"
        f"  Total in range  : {stats['total_in_range']}\n"
        f"  Output dir      : {folder.masks_dir(eye, args.object)}"
    )

    print("\n[Preview] Press P to review results in MaskViewer, any other key to skip.")
    key = input("  > ").strip().lower()
    if key == "p":
        try:
            viewer = MaskViewer.from_dirs(
                visuals_dir=folder.visualizations_dir(eye, args.object),
                masks_dir=folder.masks_dir(eye, args.object),
                fps=0.0,
                export_dir=folder.visualizations_dir(eye, args.object),
            )
            viewer.run()
        except (FileNotFoundError, ValueError) as e:
            print(f"[Preview] Could not open viewer: {e}")
    print("[Info] Removing initial masks")
    inference.reset_mask()


def run_classification(
    eye: str,
    frame_mgr: FrameManager,
    folder: FolderManager,
    video_path: Optional[str],
) -> dict:
    """
    Pre:
        - eye must be a valid string ('left' or 'right').
        - frame_mgr must be an instance of FrameManager containing valid eye frame data.
        - folder must be an instance of FolderManager.
        - video_path can be a string pointing to the video file or None.
    Post:
        - Launches an interactive classifier UI for manual template selection.
        - Runs ORB feature matching to classify the remaining unlabelled frames.
        - Persists template images and returns a dictionary mapping frame IDs to their calculated 
          state ('open' or 'close'). Returns an empty dictionary if cancelled.
    """
    fps = get_video_fps(video_path) if video_path else 25.0

    sampled = frame_mgr.sampled_frames(config.CLASSIFIER_DISPLAY_FPS, fps)

    frames_for_ui = [
        (fid, fn, os.path.join(folder.frames_dir(eye), fn))
        for fid, fn in sampled
    ]

    templates_dir = os.path.join(folder.eye_root(eye), "templates")
    storage = ClassificationStorage(templates_dir)

    ui = ClassifierUI(storage=storage, frames=frames_for_ui)
    confirmed = ui.run()

    if not confirmed:
        print("[Classification] Cancelled by user.")
        return {}

    orb = ORBClassifier(
        open_images=storage.open_templates(),
        close_images=storage.close_templates(),
    )

    classification: dict = {}

    for fid, _, state in storage.all_items():
        classification[fid] = state

    all_ids = frame_mgr.all_frame_ids()
    print(f"[Classification] Running ORB on {len(all_ids)} frames...")
    for fid in all_ids:
        if fid in classification:
            continue
        fn = frame_mgr.filename(fid)
        if fn is None:
            continue
        fp = os.path.join(folder.frames_dir(eye), fn)
        img = cv2.imread(fp)
        if img is None:
            classification[fid] = "close"
            continue
        classification[fid] = orb.classify(img)

    storage.persist_templates()

    return classification


def _build_csv_rows(frame_mgr: FrameManager, classification: dict):
    """
    Pre:
        - frame_mgr must be an instance of FrameManager.
        - classification must be a dictionary mapping frame IDs to eye states ('open' | 'close').
    Post:
        - Returns a list of tuples in the format (filename, frame_id, state) for all valid frames 
          managed by frame_mgr. Defaults unclassified frames to 'close'.
    """
    rows = []
    for fid in frame_mgr.all_frame_ids():
        fn = frame_mgr.filename(fid)
        if fn is None:
            continue
        state = classification.get(fid, "close")
        rows.append((fn, fid, state))
    return rows


def main():
    """
    Pre:
        - Command line arguments must match the expected pipeline structure.
    Post:
        - Checks for the video file availability, initializes managers, and executes 
          the segmentation pipeline for each requested eye side. Destroys all open CV2 windows upon exit.
    """
    args = parse_args()

    video_path = find_video_file(args.video_name)
    if video_path:
        print(f"[Info] Video found: {video_path}")
    else:
        print(f"[Info] Video file not found for '{args.video_name}' — will use existing frames if available.")

    folder = FolderManager(args.video_name)
    session = SessionManager(folder.sessions_dir(), args.video_name)

    if args.eye == "both":
        eyes = ["left", "right"]
    else:
        eyes = [args.eye]

    for eye in eyes:
        run_pipeline_for_eye(
            eye=eye,
            args=args,
            folder=folder,
            session=session,
            video_path=video_path,
        )

    cv2.destroyAllWindows()
    print("\n[Done] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
