from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(slots=True)
class PinchState:
    is_pinching: bool
    point: tuple[float, float] | None
    distance: float
    scale: float
    raw_ratio: float
    thumb_tip: tuple[float, float] | None = None
    index_tip: tuple[float, float] | None = None
    started: bool = False
    released: bool = False


class PinchDetector:
    """Stable thumb-index pinch detector with hysteresis.

    v2.1 change:
    - The control point is now the INDEX FINGER TIP, not the midpoint.
      This feels much more natural when pulling the mouth.
    - started/released flags are exposed so the app can lock one mouth anchor
      for the whole drag instead of constantly changing anchors.
    """

    THUMB_TIP = 4
    INDEX_TIP = 8
    WRIST = 0
    INDEX_MCP = 5
    MIDDLE_MCP = 9

    def __init__(self, on_threshold: float = 0.32, off_threshold: float = 0.44, stable_frames: int = 2):
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.stable_frames = max(1, stable_frames)
        self._active = False
        self._candidate_count = 0
        self._release_count = 0

    def update(self, hand_landmarks: list[tuple[float, float, float]] | None) -> PinchState:
        previous_active = self._active

        if not hand_landmarks or len(hand_landmarks) <= self.INDEX_TIP:
            self._active = False
            self._candidate_count = 0
            self._release_count = 0
            return PinchState(False, None, 0.0, 1.0, 999.0, None, None, started=False, released=previous_active)

        thumb = hand_landmarks[self.THUMB_TIP]
        index = hand_landmarks[self.INDEX_TIP]
        wrist = hand_landmarks[self.WRIST]
        middle_mcp = hand_landmarks[self.MIDDLE_MCP] if len(hand_landmarks) > self.MIDDLE_MCP else hand_landmarks[self.INDEX_MCP]

        thumb_xy = (float(thumb[0]), float(thumb[1]))
        index_xy = (float(index[0]), float(index[1]))

        distance = hypot(index_xy[0] - thumb_xy[0], index_xy[1] - thumb_xy[1])
        scale = max(25.0, hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]))
        ratio = distance / scale

        if self._active:
            if ratio > self.off_threshold:
                self._release_count += 1
                if self._release_count >= self.stable_frames:
                    self._active = False
                    self._candidate_count = 0
            else:
                self._release_count = 0
        else:
            if ratio < self.on_threshold:
                self._candidate_count += 1
                if self._candidate_count >= self.stable_frames:
                    self._active = True
                    self._release_count = 0
            else:
                self._candidate_count = 0

        started = self._active and not previous_active
        released = (not self._active) and previous_active

        # Use the index fingertip as the drag point. It matches what the user sees.
        return PinchState(self._active, index_xy, distance, scale, ratio, thumb_xy, index_xy, started=started, released=released)
