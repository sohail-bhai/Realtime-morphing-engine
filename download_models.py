"""Download MediaPipe Tasks model files required by RubberFace AR v2.

Run:
    python download_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0") or "0")
            chunk_size = 1024 * 512
            downloaded = 0
            with target.open("wb") as file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded * 100 / total
                        print(f"  {percent:5.1f}%", end="\r")
        print(f"  saved: {target}")
    except Exception as exc:  # pragma: no cover - this is user environment dependent
        if target.exists():
            target.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}\nReason: {exc}") from exc


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        target = MODEL_DIR / filename
        if target.exists() and target.stat().st_size > 1024:
            print(f"Already exists: {target}")
            continue
        download(url, target)
    print("\nAll models are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
