from __future__ import annotations

import cv2
import numpy as np

from .gesture import PinchState
from .mouth import MouthRegion
from .trackers import TrackingResult


_FACE_MOUTH_INDICES = [61, 291, 13, 14, 0, 17, 37, 267, 39, 269, 40, 270, 185, 409, 50, 207, 280, 427]
_HAND_IMPORTANT_INDICES = [0, 4, 5, 8, 9, 12, 17, 20]


def put_text(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int] = (245, 245, 245)) -> None:
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


def _draw_cross(frame: np.ndarray, point: tuple[float, float], color: tuple[int, int, int]) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.line(frame, (x - 8, y), (x + 8, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 8), (x, y + 8), color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    tracking: TrackingResult,
    pinch: PinchState,
    mouth: MouthRegion | None,
    active_anchor: tuple[float, float] | None,
    active_anchor_name: str | None,
    smoothed_target: tuple[float, float] | None,
    fps: float,
    strength: float,
    pull_length: float,
    release_decay: float,
    recording: bool,
    draw_landmarks: bool,
    show_help: bool,
) -> None:
    status_color = (0, 230, 0) if pinch.is_pinching else (80, 80, 255)
    if active_anchor and pinch.is_pinching:
        state = "LOCKED+STRETCH"
    elif active_anchor and release_decay > 0.05:
        state = "RELEASE_SMOOTH"
    elif pinch.is_pinching:
        state = "PINCH_NEAR_MOUTH"
    else:
        state = "idle"

    anchor_label = active_anchor_name or "none"
    put_text(frame, f"RubberFace AR v2.1 | Tasks API | FPS {fps:04.1f}", 28)
    put_text(
        frame,
        f"face={tracking.face_count} hand={tracking.hand_count} pinch={'ON' if pinch.is_pinching else 'OFF'} "
        f"ratio={pinch.raw_ratio:.2f} state={state}",
        56,
        status_color,
    )
    put_text(
        frame,
        f"anchor={anchor_label} pull={pull_length:.0f}px strength={strength:.2f} recording={'ON' if recording else 'OFF'}",
        84,
        (0, 0, 255) if recording else (245, 245, 245),
    )

    if mouth:
        x1, y1, x2, y2 = mouth.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 255), 1)
        anchor_colors = {
            "left_corner": (0, 255, 255),
            "right_corner": (0, 255, 255),
            "upper_lip": (255, 255, 0),
            "lower_lip": (255, 255, 0),
            "center": (255, 255, 255),
        }
        for name, point in mouth.anchors().items():
            color = anchor_colors.get(name, (255, 255, 255))
            radius = 6 if name == active_anchor_name else 4
            cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)
        cv2.circle(frame, (int(mouth.left_cheek[0]), int(mouth.left_cheek[1])), 3, (160, 255, 160), -1)
        cv2.circle(frame, (int(mouth.right_cheek[0]), int(mouth.right_cheek[1])), 3, (160, 255, 160), -1)

    if pinch.thumb_tip:
        cv2.circle(frame, (int(pinch.thumb_tip[0]), int(pinch.thumb_tip[1])), 6, (0, 180, 255), -1)
    if pinch.index_tip:
        cv2.circle(frame, (int(pinch.index_tip[0]), int(pinch.index_tip[1])), 9, status_color, -1)

    if smoothed_target:
        _draw_cross(frame, smoothed_target, (255, 0, 255))

    if active_anchor and smoothed_target:
        cv2.circle(frame, (int(active_anchor[0]), int(active_anchor[1])), 8, (255, 0, 255), -1)
        cv2.line(
            frame,
            (int(active_anchor[0]), int(active_anchor[1])),
            (int(smoothed_target[0]), int(smoothed_target[1])),
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if draw_landmarks:
        if tracking.face_landmarks:
            for idx in _FACE_MOUTH_INDICES:
                if idx < len(tracking.face_landmarks):
                    x, y, _ = tracking.face_landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)
        if tracking.hand_landmarks:
            for idx in _HAND_IMPORTANT_INDICES:
                if idx < len(tracking.hand_landmarks):
                    x, y, _ = tracking.hand_landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    if show_help:
        help_lines = [
            "Q/Esc quit | D debug | L landmarks | R reset | S screenshot | V record",
            "+/- strength | M mirror | H help | tip: pinch FIRST, then drag from mouth",
        ]
        base_y = frame.shape[0] - 44
        for i, line in enumerate(help_lines):
            put_text(frame, line, base_y + i * 22, (220, 220, 220))
