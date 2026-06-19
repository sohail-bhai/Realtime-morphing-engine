from __future__ import annotations

from math import hypot

import cv2
import numpy as np

from .mouth import MouthRegion


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _clamp_roi(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(frame_w - 1, int(x2)),
        min(frame_h - 1, int(y2)),
    )


def apply_elastic_mouth_pull(
    frame_bgr: np.ndarray,
    mouth: MouthRegion,
    anchor: tuple[float, float],
    target: tuple[float, float],
    strength: float = 1.0,
    max_pull_ratio: float = 2.35,
    radius_ratio: float = 1.45,
    feather: int = 41,
) -> np.ndarray:
    """Apply a stronger local rubber deformation around the selected mouth anchor.

    v2.1 fixes the main visual weakness from v2:
    the old warp only edited the mouth rectangle, so the mouth could not stretch far
    toward the finger. This function builds a dynamic ROI that includes both the
    mouth and the fingertip, then inverse-maps pixels along the pull line.
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    raw_dx = float(target[0] - anchor[0])
    raw_dy = float(target[1] - anchor[1])
    raw_len = hypot(raw_dx, raw_dy)
    if raw_len < 3.0:
        return frame_bgr

    max_pull = max(24.0, mouth.width * max_pull_ratio)
    limit_scale = min(1.0, max_pull / max(raw_len, 1e-6))

    dx = raw_dx * limit_scale * strength
    dy = raw_dy * limit_scale * strength

    # Keep vertical deformation possible but reduce it so the face does not melt downward.
    dy *= 0.62

    pull_len = hypot(dx, dy)
    if pull_len < 3.0:
        return frame_bgr

    effective_target = (anchor[0] + dx, anchor[1] + dy)

    base_x1, base_y1, base_x2, base_y2 = mouth.bbox
    pad = max(mouth.width * 0.80, mouth.height * 1.25, 34.0)
    x1, y1, x2, y2 = _clamp_roi(
        min(base_x1, anchor[0], effective_target[0]) - pad,
        min(base_y1, anchor[1], effective_target[1]) - pad,
        max(base_x2, anchor[0], effective_target[0]) + pad,
        max(base_y2, anchor[1], effective_target[1]) + pad,
        frame_w,
        frame_h,
    )

    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return frame_bgr

    roi = frame_bgr[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h < 4 or w < 4:
        return frame_bgr

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    abs_x = grid_x + x1
    abs_y = grid_y + y1

    # Projection of each pixel onto the anchor->target line segment.
    length_sq = max(dx * dx + dy * dy, 1e-6)
    rel_x = abs_x - anchor[0]
    rel_y = abs_y - anchor[1]
    t = (rel_x * dx + rel_y * dy) / length_sq
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    nearest_x = anchor[0] + t * dx
    nearest_y = anchor[1] + t * dy
    perp_sq = (abs_x - nearest_x) ** 2 + (abs_y - nearest_y) ** 2

    # Wider ribbon = more cheek/skin follows the pull.
    ribbon_sigma = max(mouth.height * radius_ratio * 0.36, mouth.width * 0.20, 18.0)
    line_weight = np.exp(-perp_sq / (2.0 * ribbon_sigma * ribbon_sigma)).astype(np.float32)

    # Local mouth/anchor attachment keeps the stretch connected to the lips.
    anchor_radius = max(mouth.width * 0.38, mouth.height * 0.55, 24.0)
    target_radius = max(mouth.width * 0.28, 20.0)
    anchor_dist_sq = (abs_x - anchor[0]) ** 2 + (abs_y - anchor[1]) ** 2
    target_dist_sq = (abs_x - effective_target[0]) ** 2 + (abs_y - effective_target[1]) ** 2
    anchor_weight = np.exp(-anchor_dist_sq / (2.0 * anchor_radius * anchor_radius)).astype(np.float32)
    target_weight = np.exp(-target_dist_sq / (2.0 * target_radius * target_radius)).astype(np.float32)

    # Coeff = how far backward each output pixel samples from.
    # Strong near the pulled endpoint, smooth along the line, smaller near the original mouth.
    segment_coeff = np.power(t, 0.52) * line_weight
    coeff = np.clip(segment_coeff + (0.14 * anchor_weight) + (0.26 * target_weight), 0.0, 1.0).astype(np.float32)

    # Add a gentle side cheek pull around the original anchor.
    # This prevents the stretch from looking like only a pasted streak.
    local_coeff = np.clip(anchor_weight * 0.18, 0.0, 0.32).astype(np.float32)
    coeff = np.maximum(coeff, local_coeff)

    map_x = abs_x - (dx * coeff)
    map_y = abs_y - (dy * coeff)

    warped = cv2.remap(
        frame_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    alpha = np.clip(
        (line_weight * (0.14 + 0.86 * np.power(t, 0.50)))
        + (anchor_weight * 0.30)
        + (target_weight * 0.50),
        0.0,
        1.0,
    ).astype(np.float32)

    feather_kernel = _odd(feather)
    alpha = cv2.GaussianBlur(alpha, (feather_kernel, feather_kernel), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    blended = (warped.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out = frame_bgr.copy()
    out[y1:y2, x1:x2] = blended
    return out


def run_offline_warp_test(output_path: str) -> None:
    """Create a synthetic face-like test image to verify the warp engine without webcam."""
    frame = np.full((500, 900, 3), 235, dtype=np.uint8)
    cv2.circle(frame, (430, 250), 150, (210, 195, 180), -1)
    cv2.circle(frame, (370, 220), 18, (30, 30, 30), -1)
    cv2.circle(frame, (490, 220), 18, (30, 30, 30), -1)
    cv2.ellipse(frame, (430, 310), (75, 28), 0, 0, 180, (30, 30, 30), 8)
    cv2.circle(frame, (360, 310), 8, (20, 20, 20), -1)
    cv2.circle(frame, (500, 310), 8, (20, 20, 20), -1)
    cv2.putText(frame, "original", (320, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    from .mouth import MouthRegion

    mouth = MouthRegion(
        center=(430, 310),
        left_corner=(360, 310),
        right_corner=(500, 310),
        upper_lip=(430, 290),
        lower_lip=(430, 330),
        left_cheek=(315, 305),
        right_cheek=(545, 305),
        bbox=(230, 210, 630, 405),
        width=140,
        height=85,
        near_radius=400,
    )
    warped = apply_elastic_mouth_pull(frame, mouth, mouth.left_corner, (155, 275), strength=1.35)
    cv2.circle(warped, (155, 275), 10, (0, 255, 0), -1)
    cv2.line(warped, (360, 310), (155, 275), (255, 0, 255), 2)
    cv2.putText(warped, "v2.1 stronger dynamic ROI", (215, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    combined = np.hstack([frame, warped])
    cv2.imwrite(output_path, combined)
