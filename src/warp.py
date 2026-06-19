from __future__ import annotations

from math import atan2, cos, hypot, sin

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
        min(frame_w, int(x2)),
        min(frame_h, int(y2)),
    )


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_elastic_mouth_pull(
    frame_bgr: np.ndarray,
    mouth: MouthRegion,
    anchor: tuple[float, float],
    target: tuple[float, float],
    strength: float = 1.0,
    max_pull_ratio: float = 1.85,
    radius_ratio: float = 1.28,
    feather: int = 31,
) -> np.ndarray:
    """Apply a cleaner local face morph controlled by the fingertip.

    v2.1 had a dynamic ROI that stretched all the way from the mouth to the finger.
    That made the hand/background smear into the face and look transparent.

    This version uses the finger only as a CONTROL VECTOR. The warp is restricted to
    the face/mouth/cheek area, so the result looks more solid and less ghost-like.
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    raw_dx = float(target[0] - anchor[0])
    raw_dy = float(target[1] - anchor[1])
    raw_len = hypot(raw_dx, raw_dy)
    if raw_len < 3.0:
        return frame_bgr

    # Limit the visual pull. The finger can move far away, but the face should not
    # liquify across the whole frame.
    max_pull = max(18.0, mouth.width * max_pull_ratio)
    limited_len = min(raw_len, max_pull)
    scale = (limited_len / max(raw_len, 1e-6)) * strength

    dx = raw_dx * scale
    dy = raw_dy * scale * 0.58  # reduce vertical melting

    pull_len = hypot(dx, dy)
    if pull_len < 3.0:
        return frame_bgr

    # Direction unit vector and perpendicular vector.
    angle = atan2(dy, dx)
    ux, uy = cos(angle), sin(angle)
    px, py = -uy, ux

    # ROI stays around mouth/cheek only. It expands in pull direction, but never to the finger.
    base_x1, base_y1, base_x2, base_y2 = mouth.bbox
    side_pad = max(mouth.width * 1.05, mouth.height * 1.25, 42.0)
    forward_pad = max(abs(dx) * 1.15, mouth.width * 0.85, 40.0)
    back_pad = max(mouth.width * 0.75, 34.0)
    vertical_pad = max(mouth.height * 1.05, 38.0) + abs(dy) * 0.35

    # Four directional extents around the anchor.
    pts = np.array(
        [
            [anchor[0] - ux * back_pad - px * side_pad, anchor[1] - uy * back_pad - py * side_pad],
            [anchor[0] - ux * back_pad + px * side_pad, anchor[1] - uy * back_pad + py * side_pad],
            [anchor[0] + ux * forward_pad - px * side_pad, anchor[1] + uy * forward_pad - py * side_pad],
            [anchor[0] + ux * forward_pad + px * side_pad, anchor[1] + uy * forward_pad + py * side_pad],
        ],
        dtype=np.float32,
    )

    x1, y1, x2, y2 = _clamp_roi(
        min(base_x1, float(np.min(pts[:, 0]))) - 18,
        min(base_y1, float(np.min(pts[:, 1]))) - vertical_pad,
        max(base_x2, float(np.max(pts[:, 0]))) + 18,
        max(base_y2, float(np.max(pts[:, 1]))) + vertical_pad,
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

    rel_x = abs_x - anchor[0]
    rel_y = abs_y - anchor[1]

    # Coordinates in pull-aligned space.
    along = rel_x * ux + rel_y * uy
    across = rel_x * px + rel_y * py

    # Elliptical influence: strong near the selected lip/corner, fades inside cheek area.
    sigma_forward = max(mouth.width * radius_ratio * 0.95, 38.0)
    sigma_back = max(mouth.width * radius_ratio * 0.45, 22.0)
    sigma_side = max(mouth.height * radius_ratio * 0.92, 26.0)

    forward_part = np.maximum(along, 0.0) / sigma_forward
    back_part = np.maximum(-along, 0.0) / sigma_back
    side_part = across / sigma_side
    dist_field = forward_part * forward_part + back_part * back_part + side_part * side_part
    core_weight = np.exp(-0.5 * dist_field).astype(np.float32)

    # Stronger pull around the selected anchor/lip, weaker at edges.
    anchor_radius = max(mouth.width * 0.34, mouth.height * 0.62, 22.0)
    anchor_dist_sq = rel_x * rel_x + rel_y * rel_y
    anchor_weight = np.exp(-anchor_dist_sq / (2.0 * anchor_radius * anchor_radius)).astype(np.float32)

    # Forward gradient makes pixels in front of the anchor follow the drag more.
    forward_gain = _smoothstep(-mouth.width * 0.25, sigma_forward * 0.90, along).astype(np.float32)
    coeff = np.clip(core_weight * (0.28 + 0.72 * forward_gain) + anchor_weight * 0.26, 0.0, 1.0)

    # Inverse map: output pixel samples from behind it. This creates a liquify/pull effect.
    map_x = abs_x - dx * coeff
    map_y = abs_y - dy * coeff

    warped_full = cv2.remap(
        frame_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Edge-only blending. The center is almost fully solid, not transparent.
    alpha = np.clip(core_weight * 1.45 + anchor_weight * 0.35, 0.0, 1.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (_odd(feather), _odd(feather)), 0)
    alpha = np.clip(alpha * 1.20, 0.0, 0.98)[..., None]

    # For very tiny alpha values, keep original pixels exactly. This avoids cloudy ghosting.
    blended = (warped_full.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    keep_original = alpha[..., 0] < 0.035
    blended[keep_original] = roi[keep_original]

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
        bbox=(260, 230, 600, 390),
        width=140,
        height=85,
        near_radius=400,
    )
    warped = apply_elastic_mouth_pull(frame, mouth, mouth.left_corner, (155, 275), strength=1.25)
    cv2.circle(warped, (155, 275), 10, (0, 255, 0), -1)
    cv2.line(warped, (360, 310), (155, 275), (255, 0, 255), 2)
    cv2.putText(warped, "v2.2 cleaner local morph", (235, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    combined = np.hstack([frame, warped])
    cv2.imwrite(output_path, combined)