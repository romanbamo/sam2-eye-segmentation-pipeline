"""
Module: config.py
Description: Global configuration file for the application. Contains all adjustable parameters 
             and path templates to ensure zero hardcoded values across other system modules.
"""

import os

SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/Path/to/weight/sam2.1_b.pt",
)
SAM2_MODEL_CONFIG = os.environ.get(
    "SAM2_MODEL_CONFIG",
    "configs/sam2.1/sam2.1_hiera_b+.yaml",
)

ORB_MAX_FEATURES: int = 1000
MATCH_DISTANCE_THRESHOLD: int = 40
MIN_OPEN_MATCHES: int = 10

BATCH_SIZE: int = 3000
BATCH_TEMP_DIR: str = "./temp_batch_frames"

CLASSIFIER_DISPLAY_FPS: int = 10

BASE_DATA_DIR: str = "."

FRAMES_SUBDIR: str = "frames"
MASKS_SUBDIR: str = "masks"
VISUALIZATIONS_SUBDIR: str = "visualizations"
SESSIONS_SUBDIR: str = "sessions"

KEY_OPEN: int = ord("o")
KEY_CLOSE: int = ord("c")
KEY_BATCH: int = ord("b")
KEY_NEXT: int = ord("n")
KEY_REPEAT: int = ord("r")
KEY_ESC: int = 27

SESSION_FILENAME: str = "session.json"
INITIAL_MASK_FILENAME: str = "initial_mask.png"
LAST_MASK_FILENAME: str = "last_mask.npy"

OVERLAY_COLOR_RGB: tuple = (0, 255, 0)
OVERLAY_ALPHA: float = 0.3
