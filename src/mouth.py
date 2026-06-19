from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(slots=True)
class MouthRegion:
    center: tuple[float, float]
    left_corner: tuple[float, float]
    right_corner: tuple[float, float]
    upper_lip: tuple[float, float]
    lower_lip: tuple[float, float]
    left_cheek: tuple[float, float]
    right_cheek: tuple[float, float]
    bbox: tuple[int, int, int, int]
    width: float
    height: float
    lip_gap: float
    open_ratio: float
    near_radius: float

    def anchors(self) -> dict[str, tuple[float, float]]:
        return {
            "left_corner": self.left_corner,
            "right_corner": self.right_corner,
            "upper_lip": self.upper_lip,
            "lower_lip": self.lower_lip,
            "center": self.center,
        }

    def is_open_for_teeth_stretch(self) -> bool:
        """True when the mouth is open enough to sample visible inner-mouth/teeth texture."""
        return self.open_ratio >= 0.115 and self.lip_gap >= max(6.0, self.width * 0.085)

    def choose_anchor(self, point: tuple[float, float]) -> tuple[str, tuple[float, float]]:
        """For Luffy-style pulling, lock only left/right corner.

        The x/y position of this anchor is refreshed every frame by app.py, so the
        effect follows the moving face instead of using an old frozen coordinate.
        """
        if point[0] < self.center[0]:
            return "left_corner", self.left_corner
        return "right_corner", self.right_corner

    def is_near(self, point: tuple[float, float]) -> bool:
        return hypot(point[0] - self.center[0], point[1] - self.center[1]) <= self.near_radius


def _xy(points: list[tuple[float, float, float]], index: int) -> tuple[float, float]:
    x, y, _ = points[index]
    return x, y


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def build_mouth_region(face_landmarks: list[tuple[float, float, float]] | None, frame_width: int, frame_height: int) -> MouthRegion | None:
    """Extract mouth landmarks and mouth-open measurements from MediaPipe Face Landmarker."""
    if not face_landmarks or len(face_landmarks) < 292:
        return None

    left = _xy(face_landmarks, 61)
    right = _xy(face_landmarks, 291)
    upper = _xy(face_landmarks, 13)
    lower = _xy(face_landmarks, 14)

    left_cheek = _mid(_xy(face_landmarks, 50), _xy(face_landmarks, 207)) if len(face_landmarks) > 207 else left
    right_cheek = _mid(_xy(face_landmarks, 280), _xy(face_landmarks, 427)) if len(face_landmarks) > 427 else right

    center = (
        (left[0] + right[0] + upper[0] + lower[0]) * 0.25,
        (left[1] + right[1] + upper[1] + lower[1]) * 0.25,
    )
    width = max(30.0, hypot(right[0] - left[0], right[1] - left[1]))
    lip_gap = max(1.0, hypot(lower[0] - upper[0], lower[1] - upper[1]))
    open_ratio = lip_gap / max(width, 1.0)

    # height is only a broad mouth-area measure. The actual stretch strip uses lip_gap.
    height = max(width * 0.34, lip_gap * 2.15, 18.0)

    xs = [left[0], right[0], upper[0], lower[0], left_cheek[0], right_cheek[0]]
    ys = [left[1], right[1], upper[1], lower[1], left_cheek[1], right_cheek[1]]

    pad_x = width * 0.78
    pad_y = height * 1.05

    x1 = max(0, int(min(xs) - pad_x))
    y1 = max(0, int(min(ys) - pad_y))
    x2 = min(frame_width - 1, int(max(xs) + pad_x))
    y2 = min(frame_height - 1, int(max(ys) + pad_y))

    return MouthRegion(
        center=center,
        left_corner=left,
        right_corner=right,
        upper_lip=upper,
        lower_lip=lower,
        left_cheek=left_cheek,
        right_cheek=right_cheek,
        bbox=(x1, y1, x2, y2),
        width=width,
        height=height,
        lip_gap=lip_gap,
        open_ratio=open_ratio,
        near_radius=width * 3.20,
    )
