#!/usr/bin/env python3
"""Prepare independent official 100-sample data for the 4B transition smoke."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/prepare_sft_smoke.py"),
        "--count", "100", "--output", str(ROOT / "data/sft_4b_smoke_100.json"),
        "--raw-dir", str(ROOT / "data/raw"),
        "--media-dir", str(ROOT / "data/sft_4b_media"),
    ], check=True)


if __name__ == "__main__":
    main()
