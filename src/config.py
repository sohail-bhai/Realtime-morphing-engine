from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SCREENSHOT_DIR = OUTPUTS_DIR / "screenshots"
RECORDING_DIR = OUTPUTS_DIR / "recordings"


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0

    # 0 means: do not force camera resolution. Use the camera's native/default frame.
    # This avoids stretched/cropped-looking output on webcams that do not support 16:9 properly.
    width: int = 0
    height: int = 0

    # Only the preview window is resized to fit screen. Processing still uses the real camera frame.
    max_display_width: int = 1280
    max_display_height: int = 720

    mirror: bool = True
    strength: float = 1.10

    # Morph tuning
    max_pull_ratio: float = 1.85
    warp_radius_ratio: float = 1.28
    feather: int = 31
    target_smoothing: float = 0.45
    release_decay: float = 0.72

    debug: bool = True
    draw_landmarks: bool = False
    show_help: bool = True

    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_face_tracking_confidence: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5

    face_model_path: Path = MODELS_DIR / "face_landmarker.task"
    hand_model_path: Path = MODELS_DIR / "hand_landmarker.task"