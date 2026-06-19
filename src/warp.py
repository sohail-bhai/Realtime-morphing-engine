from __future__ import annotations

from math import hypot

import cv2
import numpy as np

from .mouth import MouthRegion


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    length = hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _clamp_roi(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(frame_w, int(np.ceil(x2))),
        min(frame_h, int(np.ceil(y2))),
    )


def _sample_oriented_patch(
    frame_bgr: np.ndarray,
    root: tuple[float, float],
    x_axis: tuple[float, float],
    y_axis: tuple[float, float],
    length: float,
    half_width: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Sample a rectangular patch from the frame in custom local coordinates.

    x=0 is the mouth corner root. x=length goes inside the mouth.
    y is the vertical lip/teeth direction.
    """
    out_w = max(4, int(out_w))
    out_h = max(4, int(out_h))
    xs = np.linspace(0.0, float(length), out_w, dtype=np.float32)
    ys = np.linspace(-float(half_width), float(half_width), out_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    map_x = root[0] + x_axis[0] * grid_x + y_axis[0] * grid_y
    map_y = root[1] + x_axis[1] * grid_x + y_axis[1] * grid_y

    return cv2.remap(
        frame_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _polygon_alpha(height: int, width: int, polygon: np.ndarray, feather_px: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255, lineType=cv2.LINE_AA)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)
    alpha = np.clip(dist / max(float(feather_px), 1.0), 0.0, 1.0)
    alpha[mask == 0] = 0.0
    return alpha[..., None]


def _local_root_blend(
    frame_bgr: np.ndarray,
    root: tuple[float, float],
    dx: float,
    dy: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    """A subtle deformation at the mouth corner so the ribbon does not look detached."""
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_roi(
        root[0] - radius_x * 1.4,
        root[1] - radius_y * 1.4,
        root[0] + radius_x * 1.4,
        root[1] + radius_y * 1.4,
        frame_w,
        frame_h,
    )
    if x2 <= x1 + 3 or y2 <= y1 + 3:
        return frame_bgr

    grid_x, grid_y = np.meshgrid(np.arange(x1, x2, dtype=np.float32), np.arange(y1, y2, dtype=np.float32))
    rx = (grid_x - root[0]) / max(radius_x, 1e-6)
    ry = (grid_y - root[1]) / max(radius_y, 1e-6)
    weight = np.exp(-(rx * rx + ry * ry) * 3.0).astype(np.float32)

    # Keep this weak. Its job is seam glue, not the main effect.
    coeff = np.clip(weight * 0.075, 0.0, 0.075)
    map_x = grid_x - dx * coeff
    map_y = grid_y - dy * coeff

    pulled = cv2.remap(frame_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha = np.clip(weight * 0.30, 0.0, 0.30)[..., None]
    roi = frame_bgr[y1:y2, x1:x2]
    blended = (pulled.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out = frame_bgr.copy()
    out[y1:y2, x1:x2] = blended
    return out


def _build_rubber_ribbon(
    frame_bgr: np.ndarray,
    mouth: MouthRegion,
    anchor: tuple[float, float],
    target: tuple[float, float],
    strength: float,
    max_pull_ratio: float,
) -> np.ndarray:
    """Texture-stretch ribbon from mouth corner to fingertip.

    This is intentionally different from the old liquify/copy-strip:
    1. Take a small oriented source patch from the mouth corner into the mouth.
    2. Resize that patch into a long ribbon, so teeth/lip texture stretches naturally.
    3. Warp that ribbon to the pull direction.
    4. Feather only the outer edges; center stays solid.
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    raw_dx = float(target[0] - anchor[0])
    raw_dy = float(target[1] - anchor[1])
    raw_len = hypot(raw_dx, raw_dy)
    if raw_len < 6.0:
        return frame_bgr

    pull_ux, pull_uy = _normalize(raw_dx, raw_dy)
    inward_ux, inward_uy = _normalize(mouth.center[0] - anchor[0], mouth.center[1] - anchor[1])
    if abs(inward_ux) + abs(inward_uy) < 1e-6:
        inward_ux, inward_uy = -pull_ux, -pull_uy

    # Only create a strong ribbon when pulling outward from that corner.
    outward_score = raw_dx * (-inward_ux) + raw_dy * (-inward_uy)
    if outward_score < mouth.width * 0.04:
        # Wrong direction = very small safe tug.
        return _local_root_blend(frame_bgr, anchor, raw_dx * 0.30, raw_dy * 0.30, mouth.width * 0.22, mouth.height * 0.40)

    max_pull = max(mouth.width * max_pull_ratio, 34.0)
    visual_pull = min(raw_len, max_pull) * strength
    visual_pull = min(visual_pull, max_pull * 1.08)
    dx = pull_ux * visual_pull
    dy = pull_uy * visual_pull
    pull_len = hypot(dx, dy)
    if pull_len < 8.0:
        return frame_bgr

    # Width axis follows upper lip -> lower lip. This keeps teeth/lip vertical orientation correct.
    side_ux, side_uy = _normalize(mouth.lower_lip[0] - mouth.upper_lip[0], mouth.lower_lip[1] - mouth.upper_lip[1])
    if abs(side_ux) + abs(side_uy) < 1e-6:
        side_ux, side_uy = -pull_uy, pull_ux

    is_open = mouth.is_open_for_teeth_stretch()

    # Closed mouth still works, but becomes a narrow lip ribbon instead of a full mouth/teeth strip.
    if is_open:
        source_len = max(mouth.width * 0.82, mouth.lip_gap * 3.2, 28.0)
        source_half = max(min(mouth.lip_gap * 0.95, mouth.width * 0.24), 8.0)
        root_half = max(min(mouth.lip_gap * 0.92, mouth.width * 0.24), 8.0)
        tip_half = max(root_half * 0.46, 4.5)
    else:
        source_len = max(mouth.width * 0.48, 18.0)
        source_half = max(min(mouth.width * 0.075, 9.0), 4.5)
        root_half = max(min(mouth.width * 0.070, 8.0), 4.2)
        tip_half = max(root_half * 0.48, 3.0)
        # Closed-mouth pulls should be smaller and less aggressive.
        pull_len *= 0.72
        dx = pull_ux * pull_len
        dy = pull_uy * pull_len

    # Root starts slightly inside the mouth corner for seam continuity.
    root = (
        anchor[0] + inward_ux * max(mouth.width * 0.020, 1.0),
        anchor[1] + inward_uy * max(mouth.lip_gap * 0.040, 0.5),
    )
    tip = (root[0] + dx, root[1] + dy)

    # Oriented source patch from corner into mouth interior.
    src_w = max(18, int(round(source_len)))
    src_h = max(8, int(round(source_half * 2.0)))
    source_patch = _sample_oriented_patch(
        frame_bgr=frame_bgr,
        root=root,
        x_axis=(inward_ux, inward_uy),
        y_axis=(side_ux, side_uy),
        length=source_len,
        half_width=source_half,
        out_w=src_w,
        out_h=src_h,
    )

    # Stretch source patch into the desired ribbon length.
    ribbon_w = max(12, int(round(pull_len)))
    ribbon_h = max(6, int(round(root_half * 2.0)))
    ribbon = cv2.resize(source_patch, (ribbon_w, ribbon_h), interpolation=cv2.INTER_LINEAR)

    # Darken/soften closed-mouth lip ribbon a tiny bit to avoid harsh pasted rectangles.
    if not is_open:
        ribbon = cv2.GaussianBlur(ribbon, (3, 3), 0)

    src_rect = np.array(
        [[0, 0], [ribbon_w - 1, 0], [ribbon_w - 1, ribbon_h - 1], [0, ribbon_h - 1]],
        dtype=np.float32,
    )
    dst_poly = np.array(
        [
            [root[0] - side_ux * root_half, root[1] - side_uy * root_half],
            [tip[0] - side_ux * tip_half, tip[1] - side_uy * tip_half],
            [tip[0] + side_ux * tip_half, tip[1] + side_uy * tip_half],
            [root[0] + side_ux * root_half, root[1] + side_uy * root_half],
        ],
        dtype=np.float32,
    )

    margin = max(root_half * 1.0, 8.0)
    x1, y1, x2, y2 = _clamp_roi(
        float(dst_poly[:, 0].min()) - margin,
        float(dst_poly[:, 1].min()) - margin,
        float(dst_poly[:, 0].max()) + margin,
        float(dst_poly[:, 1].max()) + margin,
        frame_w,
        frame_h,
    )
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return frame_bgr

    base = _local_root_blend(frame_bgr, root, dx, dy, max(root_half * 1.1, 10.0), max(root_half * 1.05, 9.0))

    # Warp ribbon to full-frame canvas then blend only the small ROI.
    transform = cv2.getPerspectiveTransform(src_rect, dst_poly)
    canvas = cv2.warpPerspective(
        ribbon,
        transform,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    alpha_full = _polygon_alpha(frame_h, frame_w, dst_poly, feather_px=max(2, int(root_half * 0.24)))

    # Fade the very tip a little so it feels held by the finger rather than squared off.
    # This is still mostly opaque in the center.
    yy, xx = np.mgrid[0:frame_h, 0:frame_w].astype(np.float32)
    rel_x = xx - root[0]
    rel_y = yy - root[1]
    along = (rel_x * pull_ux + rel_y * pull_uy) / max(pull_len, 1e-6)
    tip_fade = 1.0 - 0.10 * _smoothstep(0.90, 1.0, along)
    root_boost = 0.82 + 0.18 * _smoothstep(0.02, 0.09, along)
    alpha_full[..., 0] *= np.clip(tip_fade * root_boost, 0.0, 1.0)

    roi = base[y1:y2, x1:x2]
    canvas_roi = canvas[y1:y2, x1:x2]
    alpha = alpha_full[y1:y2, x1:x2]

    out_roi = (canvas_roi.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out = base.copy()
    out[y1:y2, x1:x2] = out_roi
    return out


def apply_elastic_mouth_pull(
    frame_bgr: np.ndarray,
    mouth: MouthRegion,
    anchor: tuple[float, float],
    target: tuple[float, float],
    strength: float = 1.0,
    max_pull_ratio: float = 1.85,
    radius_ratio: float = 1.10,
    feather: int = 17,
    anchor_name: str | None = None,
) -> np.ndarray:
    if anchor_name in {"left_corner", "right_corner", "left", "right"}:
        return _build_rubber_ribbon(frame_bgr, mouth, anchor, target, strength, max_pull_ratio)

    # Non-corner anchors are not part of the current Luffy-style effect.
    raw_dx = float(target[0] - anchor[0])
    raw_dy = float(target[1] - anchor[1])
    return _local_root_blend(frame_bgr, anchor, raw_dx * 0.20, raw_dy * 0.20, mouth.width * 0.18, mouth.height * 0.30)


def _draw_synthetic_face(open_mouth: bool) -> tuple[np.ndarray, MouthRegion]:
    frame = np.full((430, 760, 3), 232, dtype=np.uint8)
    cv2.circle(frame, (230, 210), 132, (194, 174, 157), -1)
    cv2.circle(frame, (190, 178), 14, (30, 30, 30), -1)
    cv2.circle(frame, (270, 178), 14, (30, 30, 30), -1)

    if open_mouth:
        upper = (230, 235)
        lower = (230, 270)
        lip_gap = 35.0
        cv2.ellipse(frame, (230, 253), (62, 30), 0, 0, 180, (35, 35, 35), 8)
        cv2.ellipse(frame, (230, 255), (44, 14), 0, 0, 180, (240, 240, 230), -1)
        # tooth divisions
        for x in [205, 220, 235, 250]:
            cv2.line(frame, (x, 244), (x, 257), (200, 200, 190), 1, cv2.LINE_AA)
        cv2.line(frame, (176, 253), (284, 253), (65, 35, 42), 5, cv2.LINE_AA)
    else:
        upper = (230, 249)
        lower = (230, 254)
        lip_gap = 5.0
        cv2.line(frame, (176, 252), (284, 252), (60, 28, 36), 6, cv2.LINE_AA)

    cv2.circle(frame, (176, 252), 7, (52, 25, 30), -1)
    cv2.circle(frame, (284, 252), 7, (52, 25, 30), -1)

    mouth = MouthRegion(
        center=(230, 252),
        left_corner=(176, 252),
        right_corner=(284, 252),
        upper_lip=upper,
        lower_lip=lower,
        left_cheek=(150, 250),
        right_cheek=(310, 250),
        bbox=(110, 205, 350, 296),
        width=108,
        height=max(108 * 0.34, lip_gap * 2.15, 18.0),
        lip_gap=lip_gap,
        open_ratio=lip_gap / 108,
        near_radius=280,
    )
    return frame, mouth


def run_offline_warp_test(output_path: str) -> None:
    """Offline test: open mouth and closed mouth should both be stable."""
    open_frame, open_mouth = _draw_synthetic_face(open_mouth=True)
    closed_frame, closed_mouth = _draw_synthetic_face(open_mouth=False)

    open_out = apply_elastic_mouth_pull(
        open_frame,
        open_mouth,
        open_mouth.left_corner,
        (50, 246),
        strength=1.08,
        max_pull_ratio=1.75,
        anchor_name="left_corner",
    )
    cv2.circle(open_out, (50, 246), 8, (0, 255, 0), -1)
    cv2.line(open_out, (176, 252), (50, 246), (255, 0, 255), 2)
    cv2.putText(open_out, "OPEN: teeth/lip ribbon", (35, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    closed_out = apply_elastic_mouth_pull(
        closed_frame,
        closed_mouth,
        closed_mouth.left_corner,
        (50, 246),
        strength=1.08,
        max_pull_ratio=1.75,
        anchor_name="left_corner",
    )
    cv2.circle(closed_out, (50, 246), 8, (0, 255, 0), -1)
    cv2.line(closed_out, (176, 252), (50, 246), (255, 0, 255), 2)
    cv2.putText(closed_out, "CLOSED: lip ribbon", (55, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    combined = np.hstack([open_out, closed_out])
    cv2.imwrite(output_path, combined)
