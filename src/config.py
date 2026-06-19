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
    width: int = 960
    height: int = 540
    mirror: bool = True
    strength: float = 1.15
    max_pull_ratio: float = 2.35
    warp_radius_ratio: float = 1.45
    feather: int = 41
    target_smoothing: float = 0.42  # higher = follows finger faster
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
