"""
Module: modules/ui/keyboard_controller.py
Description: Centralizes the mapping of keyboard key integer codes captured via OpenCV 
             to semantic action string tokens. Decouples interface rendering loops 
             from explicit keycode comparison checks.
"""

from config import (
    KEY_BATCH,
    KEY_CLOSE,
    KEY_ESC,
    KEY_NEXT,
    KEY_OPEN,
    KEY_REPEAT,
)


class KeyboardController:
    """Maps keyboard inputs to semantic system action tokens."""

    @staticmethod
    def action(key: int) -> str:
        """
        Pre:
            - key must be an integer, typically obtained from cv2.waitKey() & 0xFF.
        Post:
            - Returns a matching lowercase action string token representing the recognized command:
              'open', 'close', 'batch', 'next', 'repeat', 'esc', or 'unknown'.
        """
        mapping = {
            KEY_OPEN: "open",
            KEY_CLOSE: "close",
            KEY_BATCH: "batch",
            KEY_NEXT: "next",
            KEY_REPEAT: "repeat",
            KEY_ESC: "esc",
        }
        return mapping.get(key, "unknown")
