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
    near_radius: float

    def anchors(self) -> dict[str, tuple[float, float]]:
        return {
            "left_corner": self.left_corner,
            "right_corner": self.right_corner,
            "upper_lip": self.upper_lip,
            "lower_lip": self.lower_lip,
            "center": self.center,
        }

    def choose_anchor(self, point: tuple[float, float]) -> tuple[str, tuple[float, float]]:
        """Pick the nearest useful mouth anchor.

        v2.1 improvement: older version only chose left/right side based on x.
        This version supports upper/lower lip and center pulls too.
        """
        best_name = "center"
        best_point = self.center
        best_score = float("inf")
        for name, anchor in self.anchors().items():
            dx = point[0] - anchor[0]
            dy = point[1] - anchor[1]
            score = dx * dx + dy * dy

            # Small bias toward corners because side pulling is the main effect.
            if name in {"left_corner", "right_corner"}:
                score *= 0.78
            elif name == "center":
                score *= 1.25

            if score < best_score:
                best_score = score
                best_name = name
                best_point = anchor
        return best_name, best_point

    def is_near(self, point: tuple[float, float]) -> bool:
        return hypot(point[0] - self.center[0], point[1] - self.center[1]) <= self.near_radius


def _xy(points: list[tuple[float, float, float]], index: int) -> tuple[float, float]:
    x, y, _ = points[index]
    return x, y


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def build_mouth_region(face_landmarks: list[tuple[float, float, float]] | None, frame_width: int, frame_height: int) -> MouthRegion | None:
    """Extract a robust mouth/cheek region from Face Landmarker points."""
    if not face_landmarks or len(face_landmarks) < 292:
        return None

    left = _xy(face_landmarks, 61)
    right = _xy(face_landmarks, 291)
    upper = _xy(face_landmarks, 13)
    lower = _xy(face_landmarks, 14)

    # Cheek-side helper anchors make the deformation area bigger and more natural.
    # These are not used directly as latch anchors yet, but they expand the ROI.
    left_cheek = _mid(_xy(face_landmarks, 50), _xy(face_landmarks, 207)) if len(face_landmarks) > 207 else left
    right_cheek = _mid(_xy(face_landmarks, 280), _xy(face_landmarks, 427)) if len(face_landmarks) > 427 else right

    center = (
        (left[0] + right[0] + upper[0] + lower[0]) * 0.25,
        (left[1] + right[1] + upper[1] + lower[1]) * 0.25,
    )
    width = max(30.0, hypot(right[0] - left[0], right[1] - left[1]))
    lip_gap = max(8.0, hypot(lower[0] - upper[0], lower[1] - upper[1]))
    height = max(width * 0.50, lip_gap * 5.0)

    # v2.1: slightly larger base ROI. The warp function expands it further toward the finger.
    pad_x = width * 1.45
    pad_y_top = height * 1.00
    pad_y_bottom = height * 1.15

    x1 = max(0, int(center[0] - pad_x))
    x2 = min(frame_width - 1, int(center[0] + pad_x))
    y1 = max(0, int(center[1] - pad_y_top))
    y2 = min(frame_height - 1, int(center[1] + pad_y_bottom))

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
        near_radius=width * 3.15,
    )
