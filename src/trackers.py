from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AppConfig


@dataclass(slots=True)
class TrackingResult:
    face_landmarks: list[tuple[float, float, float]] | None
    hand_landmarks: list[tuple[float, float, float]] | None
    handedness: str | None
    face_count: int
    hand_count: int


class MediaPipeTasksTracker:
    """Face + hand tracking using the modern MediaPipe Tasks API.

    This class deliberately avoids old `mp.solutions.*` imports.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.mp = self._import_mediapipe()
        self.BaseOptions, self.vision = self._load_tasks_api(self.mp)
        self.face_landmarker = self._create_face_landmarker(cfg.face_model_path)
        self.hand_landmarker = self._create_hand_landmarker(cfg.hand_model_path)

    @staticmethod
    def _import_mediapipe() -> Any:
        try:
            import mediapipe as mp  # type: ignore

            return mp
        except Exception as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "MediaPipe could not be imported.\n"
                "Install dependencies first:\n"
                "  python -m pip install -r requirements.txt\n"
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _load_tasks_api(mp: Any) -> tuple[Any, Any]:
        """Return (BaseOptions, vision module) across MediaPipe packaging variants."""
        try:
            return mp.tasks.BaseOptions, mp.tasks.vision
        except Exception:
            pass

        try:
            from mediapipe.tasks import python as mp_tasks_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore

            return mp_tasks_python.BaseOptions, vision
        except Exception as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "Your MediaPipe install does not expose the Tasks Vision API.\n"
                "This v2 project needs FaceLandmarker and HandLandmarker, not legacy mp.solutions.\n\n"
                "Try inside your virtual environment:\n"
                "  python -m pip install --upgrade mediapipe\n"
                "  python check_environment.py\n\n"
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _require_model(path: Path, friendly_name: str) -> None:
        if not path.exists() or path.stat().st_size < 1024:
            raise FileNotFoundError(
                f"Missing {friendly_name} model file:\n"
                f"  {path}\n\n"
                "Download the required model files with:\n"
                "  python download_models.py\n"
            )

    def _create_face_landmarker(self, model_path: Path) -> Any:
        self._require_model(model_path, "Face Landmarker")
        options = self.vision.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(model_path)),
            running_mode=self.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.cfg.min_face_detection_confidence,
            min_face_presence_confidence=self.cfg.min_face_presence_confidence,
            min_tracking_confidence=self.cfg.min_face_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return self.vision.FaceLandmarker.create_from_options(options)

    def _create_hand_landmarker(self, model_path: Path) -> Any:
        self._require_model(model_path, "Hand Landmarker")
        options = self.vision.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(model_path)),
            running_mode=self.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_hand_presence_confidence,
            min_tracking_confidence=self.cfg.min_hand_tracking_confidence,
        )
        return self.vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> TrackingResult:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_landmarks = None
        hand_landmarks = None
        handedness = None

        if getattr(face_result, "face_landmarks", None):
            face_landmarks = self._normalized_to_pixels(face_result.face_landmarks[0], width, height)

        if getattr(hand_result, "hand_landmarks", None):
            hand_landmarks = self._normalized_to_pixels(hand_result.hand_landmarks[0], width, height)
            handedness = self._extract_handedness(hand_result)

        return TrackingResult(
            face_landmarks=face_landmarks,
            hand_landmarks=hand_landmarks,
            handedness=handedness,
            face_count=len(getattr(face_result, "face_landmarks", []) or []),
            hand_count=len(getattr(hand_result, "hand_landmarks", []) or []),
        )

    @staticmethod
    def _normalized_to_pixels(landmarks: list[Any], width: int, height: int) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for lm in landmarks:
            points.append((float(lm.x) * width, float(lm.y) * height, float(getattr(lm, "z", 0.0))))
        return points

    @staticmethod
    def _extract_handedness(hand_result: Any) -> str | None:
        handedness_list = getattr(hand_result, "handedness", None)
        if not handedness_list:
            return None
        try:
            first = handedness_list[0][0]
            return str(first.category_name)
        except Exception:
            return None

    def close(self) -> None:
        for landmarker in (self.face_landmarker, self.hand_landmarker):
            try:
                landmarker.close()
            except Exception:
                pass
