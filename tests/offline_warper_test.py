from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SCREENSHOT_DIR
from src.warp import run_offline_warp_test


def main() -> int:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SCREENSHOT_DIR / "v2_1_offline_warper_test.png"
    run_offline_warp_test(str(output_path))
    print(f"Saved offline warp test: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
