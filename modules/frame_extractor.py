"""
Module: modules/frame_extractor.py
Description: Opens a video file, reads its frames, downsamples them according to parameters, 
             splits each frame vertically down the center into 50% left and right segments, 
             and exports the sequence into corresponding left and right output directories.
"""

import os
import cv2
from typing import Optional


class FrameExtractor:
    """
    Extracts and splits frames from a video file into left and right sequences.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    video_name : str
        Base name of the video (without extension), used for naming output files.
    dir_left : str
        Output directory for the left eye frames.
    dir_right : str
        Output directory for the right eye frames.
    start_sec : int, optional
        Starting time in seconds (0 = from the beginning).
    end_sec : Optional[int], optional
        Ending time in seconds (None = until the end of the video).
    samples_per_second : int, optional
        Target number of frames to extract per second (-1 = extract all frames).
    """

    def __init__(
        self,
        video_path: str,
        video_name: str,
        dir_left: str,
        dir_right: str,
        start_sec: int = 0,
        end_sec: Optional[int] = None,
        samples_per_second: int = -1,
    ):
        """
        Pre:
            - video_path must be a valid path string to an existing video file.
            - video_name, dir_left, and dir_right must be valid non-empty strings representing names or paths.
            - start_sec must be a non-negative integer.
            - end_sec must be a positive integer greater than start_sec, or None.
            - samples_per_second must be an integer (either -1 or greater than 0).
        Post:
            - Initializes the FrameExtractor attributes and ensures that both left and right target directories 
              exist on disk, creating them if necessary.
        """
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_name))[0]
        self.dir_left = dir_left
        self.dir_right = dir_right
        self.start_sec = start_sec
        self.end_sec = end_sec  
        self.samples_per_second = samples_per_second

        os.makedirs(dir_left, exist_ok=True)
        os.makedirs(dir_right, exist_ok=True)

    def extract(self) -> int:
        """
        Pre:
            - The video file located at self.video_path must be accessible and readable by OpenCV.
        Post:
            - Processes the video based on start_sec, end_sec, and samples_per_second parameters.
            - Extracts, splits vertically, and writes left and right sub-images to disk.
            - Returns an integer indicating the total number of frame pairs successfully saved.
            - Raises an IOError if the video file cannot be opened.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(self.start_sec * fps)
        end_time_sec = float("inf") if self.end_sec is None else self.end_sec

        if start_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        sample_step = (
            1
            if self.samples_per_second == -1
            else max(1, int(fps / self.samples_per_second))
        )

        current_frame = start_frame
        saved = 0

        print(
            f"[Extractor] Extracting frames {self.start_sec}s → "
            f"{'end' if self.end_sec is None else str(self.end_sec) + 's'} "
            f"(step={sample_step})"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_second = current_frame / fps
            if current_second > end_time_sec:
                break

            if current_frame % sample_step == 0:
                height, width = frame.shape[:2]
                middle = width // 2

                left_eye = frame[:, 0:middle]
                right_eye = frame[:, middle:width]

                name_l = os.path.join(
                    self.dir_left,
                    f"{self.video_name}_frame_{current_frame:07d}_L.png",
                )
                name_r = os.path.join(
                    self.dir_right,
                    f"{self.video_name}_frame_{current_frame:07d}_R.png",
                )

                cv2.imwrite(name_l, left_eye)
                cv2.imwrite(name_r, right_eye)
                saved += 1

                if current_frame % 500 == 0:
                    print(f"  [Extractor] frame {current_frame}")

            current_frame += 1

        cap.release()
        print(f"[Extractor] Done. {saved} frame pairs saved.")
        return saved
