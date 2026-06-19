from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import RECORDING_DIR, SCREENSHOT_DIR


def timestamp_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"


def save_screenshot(frame: np.ndarray) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / timestamp_name("rubberface", ".png")
    cv2.imwrite(str(path), frame)
    return path


class VideoRecorder:
    def __init__(self) -> None:
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None

    @property
    def active(self) -> bool:
        return self.writer is not None

    def toggle(self, frame: np.ndarray, fps: float) -> Path | None:
        if self.writer is not None:
            return self.stop()
        return self.start(frame, fps)

    def start(self, frame: np.ndarray, fps: float) -> Path:
        RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RECORDING_DIR / timestamp_name("rubberface_recording", ".mp4")
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        safe_fps = max(12.0, min(60.0, fps if fps > 0 else 30.0))
        self.writer = cv2.VideoWriter(str(self.path), fourcc, safe_fps, (width, height))
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError("Could not start video recorder. Try running as administrator or use a different folder.")
        return self.path

    def write(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self) -> Path | None:
        path = self.path
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.path = None
        return path
