from __future__ import annotations

import argparse
import sys
import time
from math import hypot

import cv2

from .config import AppConfig
from .debug_draw import draw_overlay, put_text
from .gesture import PinchDetector, PinchState
from .io_utils import VideoRecorder, save_screenshot
from .mouth import MouthRegion, build_mouth_region
from .trackers import MediaPipeTasksTracker
from .warp import apply_elastic_mouth_pull


class FpsMeter:
    def __init__(self) -> None:
        self.last = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        dt = max(1e-6, now - self.last)
        instant = 1.0 / dt
        self.fps = instant if self.fps <= 0 else (self.fps * 0.85 + instant * 0.15)
        self.last = now
        return self.fps


class RubberFaceApp:
    """Main realtime app loop.

    v2.9:
    - moving anchor name stays locked but x/y follows the current face
    - closed mouth now uses a narrow lip ribbon instead of disabling the effect
    - open mouth uses a teeth/lip texture ribbon sampled from the actual mouth
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.tracker = MediaPipeTasksTracker(cfg)
        self.pinch_detector = PinchDetector()
        self.fps_meter = FpsMeter()
        self.recorder = VideoRecorder()

        self.active_anchor: tuple[float, float] | None = None
        self.active_anchor_name: str | None = None
        self.smoothed_target: tuple[float, float] | None = None

        self.release_vector: tuple[float, float] | None = None
        self.release_decay_value = 0.0

        self.pull_length = 0.0

        self.debug = cfg.debug
        self.draw_landmarks = cfg.draw_landmarks
        self.show_help = cfg.show_help
        self.mirror = cfg.mirror
        self.strength = cfg.strength
        self.last_message = ""
        self.last_message_until = 0.0

        self.previous_anchor_position: tuple[float, float] | None = None

    def run(self) -> int:
        cap = self._open_camera()
        window_name = "Realtime Morphing Engine - MediaPipe Tasks"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera frame not received. Try a different --camera value.")
                    return 1

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int(time.monotonic() * 1000)
                fps = self.fps_meter.update()

                try:
                    output = self._process_frame(frame, timestamp_ms, fps)
                except Exception as exc:
                    output = frame.copy()
                    put_text(output, f"Runtime error: {exc}", 36, (0, 0, 255))
                    print(f"Runtime error: {exc}", file=sys.stderr)

                self.recorder.write(output)
                cv2.imshow(window_name, output)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, output, fps):
                    break
        finally:
            self.recorder.stop()
            cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()
        return 0

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.cfg.camera_index}.\n"
                "Try: python main.py --camera 1\n"
                "Also check Windows camera privacy permission."
            )
        if self.cfg.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        if self.cfg.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        return cap

    def _process_frame(self, frame, timestamp_ms: int, fps: float):
        height, width = frame.shape[:2]
        tracking = self.tracker.process(frame, timestamp_ms)
        mouth = build_mouth_region(tracking.face_landmarks, width, height)
        pinch = self.pinch_detector.update(tracking.hand_landmarks)

        output = frame.copy()

        if mouth and pinch.is_pinching and pinch.point:
            self._update_drag_state(mouth, pinch)

            if self.active_anchor_name:
                self.active_anchor = self._current_anchor_from_name(mouth, self.active_anchor_name)

                if self._anchor_jump_too_large(mouth):
                    self._clear_drag_if_dead(force=True)
                    self._flash("Tracking jump reset")
                elif self.active_anchor and self.smoothed_target:
                    self.pull_length = hypot(
                        self.smoothed_target[0] - self.active_anchor[0],
                        self.smoothed_target[1] - self.active_anchor[1],
                    )
                    output = apply_elastic_mouth_pull(
                        output,
                        mouth=mouth,
                        anchor=self.active_anchor,
                        target=self.smoothed_target,
                        strength=self.strength,
                        max_pull_ratio=self.cfg.max_pull_ratio,
                        radius_ratio=self.cfg.warp_radius_ratio,
                        feather=self.cfg.feather,
                        anchor_name=self.active_anchor_name,
                    )
        else:
            output = self._apply_release_if_needed(output, mouth)

        if self.debug:
            draw_overlay(
                output,
                tracking=tracking,
                pinch=pinch,
                mouth=mouth,
                active_anchor=self.active_anchor,
                active_anchor_name=self.active_anchor_name,
                smoothed_target=self.smoothed_target,
                fps=fps,
                strength=self.strength,
                pull_length=self.pull_length,
                release_decay=self.release_decay_value,
                recording=self.recorder.active,
                draw_landmarks=self.draw_landmarks,
                show_help=self.show_help,
            )

            if mouth:
                mode = "teeth" if mouth.is_open_for_teeth_stretch() else "lip"
                put_text(output, f"mouth={mode} open={mouth.open_ratio:.2f} gap={mouth.lip_gap:.0f}px", 104, (0, 255, 0))

            self._draw_flash_message(output)

        return output

    def _current_anchor_from_name(self, mouth: MouthRegion, name: str) -> tuple[float, float]:
        if name == "left_corner":
            return mouth.left_corner
        if name == "right_corner":
            return mouth.right_corner
        if name == "upper_lip":
            return mouth.upper_lip
        if name == "lower_lip":
            return mouth.lower_lip
        return mouth.center

    def _anchor_jump_too_large(self, mouth: MouthRegion) -> bool:
        if self.active_anchor is None:
            return False

        if self.previous_anchor_position is None:
            self.previous_anchor_position = self.active_anchor
            return False

        jump = hypot(
            self.active_anchor[0] - self.previous_anchor_position[0],
            self.active_anchor[1] - self.previous_anchor_position[1],
        )
        self.previous_anchor_position = self.active_anchor
        return jump > max(42.0, mouth.width * 1.15)

    def _update_drag_state(self, mouth: MouthRegion, pinch: PinchState) -> None:
        point = pinch.point
        if point is None:
            return

        self.release_decay_value = 0.0
        self.release_vector = None

        if self.active_anchor_name is None:
            if not mouth.is_near(point):
                self.smoothed_target = point
                self.pull_length = 0.0
                return

            self.active_anchor_name, self.active_anchor = mouth.choose_anchor(point)
            self.previous_anchor_position = self.active_anchor
            self.smoothed_target = point
            self.pull_length = 0.0
            return

        self.active_anchor = self._current_anchor_from_name(mouth, self.active_anchor_name)

        if self.smoothed_target is None:
            self.smoothed_target = point
            return

        a = max(0.05, min(0.95, self.cfg.target_smoothing))
        self.smoothed_target = (
            self.smoothed_target[0] * (1.0 - a) + point[0] * a,
            self.smoothed_target[1] * (1.0 - a) + point[1] * a,
        )

    def _apply_release_if_needed(self, output, mouth: MouthRegion | None):
        if not mouth or not self.active_anchor_name or not self.smoothed_target:
            self._clear_drag_if_dead()
            return output

        self.active_anchor = self._current_anchor_from_name(mouth, self.active_anchor_name)

        if self.release_vector is None:
            self.release_vector = (
                self.smoothed_target[0] - self.active_anchor[0],
                self.smoothed_target[1] - self.active_anchor[1],
            )
            self.release_decay_value = 1.0

        if self.release_decay_value <= 0.06:
            self._clear_drag_if_dead(force=True)
            return output

        target = (
            self.active_anchor[0] + self.release_vector[0] * self.release_decay_value,
            self.active_anchor[1] + self.release_vector[1] * self.release_decay_value,
        )
        self.smoothed_target = target
        self.pull_length = hypot(target[0] - self.active_anchor[0], target[1] - self.active_anchor[1])

        output = apply_elastic_mouth_pull(
            output,
            mouth=mouth,
            anchor=self.active_anchor,
            target=target,
            strength=self.strength,
            max_pull_ratio=self.cfg.max_pull_ratio,
            radius_ratio=self.cfg.warp_radius_ratio,
            feather=self.cfg.feather,
            anchor_name=self.active_anchor_name,
        )
        self.release_decay_value *= self.cfg.release_decay
        return output

    def _clear_drag_if_dead(self, force: bool = False) -> None:
        if force or self.release_decay_value <= 0.06:
            self.active_anchor = None
            self.active_anchor_name = None
            self.smoothed_target = None
            self.release_vector = None
            self.release_decay_value = 0.0
            self.pull_length = 0.0
            self.previous_anchor_position = None

    def _handle_key(self, key: int, frame, fps: float) -> bool:
        if key in (27, ord("q"), ord("Q")):
            return False
        if key in (ord("d"), ord("D")):
            self.debug = not self.debug
        elif key in (ord("l"), ord("L")):
            self.draw_landmarks = not self.draw_landmarks
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        elif key in (ord("m"), ord("M")):
            self.mirror = not self.mirror
        elif key in (ord("r"), ord("R")):
            self._clear_drag_if_dead(force=True)
            self._flash("Stretch reset")
        elif key in (ord("s"), ord("S")):
            path = save_screenshot(frame)
            self._flash(f"Screenshot saved: {path.name}")
            print(f"Screenshot saved: {path}")
        elif key in (ord("v"), ord("V")):
            path = self.recorder.toggle(frame, fps)
            if self.recorder.active:
                self._flash("Recording started")
                print(f"Recording started: {path}")
            else:
                self._flash("Recording stopped")
                print(f"Recording stopped: {path}")
        elif key in (ord("+"), ord("=")):
            self.strength = min(2.50, self.strength + 0.05)
        elif key in (ord("-"), ord("_")):
            self.strength = max(0.10, self.strength - 0.05)
        return True

    def _flash(self, message: str) -> None:
        self.last_message = message
        self.last_message_until = time.perf_counter() + 1.8

    def _draw_flash_message(self, frame) -> None:
        if self.last_message and time.perf_counter() < self.last_message_until:
            put_text(frame, self.last_message, 130, (50, 255, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Morphing Engine - MediaPipe Tasks")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Try 1 if 0 does not work.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width. Default is 640 for lower lag.")
    parser.add_argument("--height", type=int, default=360, help="Requested camera height. Default is 360 for lower lag.")
    parser.add_argument("--strength", type=float, default=1.00, help="Morph strength, e.g. 0.8 to 1.3.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror view.")
    parser.add_argument("--no-debug", action="store_true", help="Start without debug overlay.")
    parser.add_argument("--landmarks", action="store_true", help="Draw selected face/hand landmarks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = AppConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        strength=args.strength,
        debug=not args.no_debug,
        draw_landmarks=args.landmarks,
    )
    try:
        app = RubberFaceApp(cfg)
        return app.run()
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
