"""Check whether RubberFace AR v2 can run in the current environment.

Run:
    python check_environment.py
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def check_python() -> bool:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    if sys.version_info < (3, 10):
        print("  ❌ Python 3.10+ is recommended.")
        return False
    print("  ✅ Python version OK")
    return True


def check_imports() -> bool:
    ok = True
    try:
        import cv2  # type: ignore

        print(f"OpenCV: {cv2.__version__} ✅")
    except Exception as exc:
        print(f"OpenCV import failed ❌: {exc}")
        ok = False

    try:
        import numpy as np  # type: ignore

        print(f"NumPy: {np.__version__} ✅")
    except Exception as exc:
        print(f"NumPy import failed ❌: {exc}")
        ok = False

    try:
        import mediapipe as mp  # type: ignore

        print(f"MediaPipe: {getattr(mp, '__version__', 'unknown')} from {getattr(mp, '__file__', 'unknown')}")
        has_tasks = hasattr(mp, "tasks")
        has_image = hasattr(mp, "Image") and hasattr(mp, "ImageFormat")
        if not has_tasks:
            print("  ❌ mediapipe.tasks not found")
            ok = False
        else:
            print("  ✅ mediapipe.tasks found")
        if not has_image:
            print("  ❌ mp.Image/mp.ImageFormat not found")
            ok = False
        else:
            print("  ✅ mp.Image found")
        try:
            _ = mp.tasks.vision.FaceLandmarker
            _ = mp.tasks.vision.HandLandmarker
            print("  ✅ FaceLandmarker and HandLandmarker found")
        except Exception as exc:
            print(f"  ❌ MediaPipe vision tasks not available: {exc}")
            ok = False
    except Exception as exc:
        print(f"MediaPipe import failed ❌: {exc}")
        ok = False

    return ok


def check_models() -> bool:
    ok = True
    required = ["face_landmarker.task", "hand_landmarker.task"]
    for filename in required:
        path = MODEL_DIR / filename
        if path.exists() and path.stat().st_size > 1024:
            print(f"Model {filename}: ✅")
        else:
            print(f"Model {filename}: ❌ missing")
            ok = False
    if not ok:
        print("\nRun this to download missing models:")
        print("  python download_models.py")
    return ok


def main() -> int:
    print("RubberFace AR v2 environment check\n")
    checks = [check_python(), check_imports(), check_models()]
    if all(checks):
        print("\n✅ Environment looks ready. Run: python main.py")
        return 0
    print("\n❌ Environment is not ready yet. Fix the items above and run this check again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
