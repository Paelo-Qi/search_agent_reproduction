#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.dev_data import DEV_SEED, write_dev_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fully local Phase 0 Dev dataset.")
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "sft_dev_4.json")
    parser.add_argument("--media-dir", type=Path, default=PROJECT_ROOT / "data" / "dev_media")
    args = parser.parse_args()
    output, metadata = write_dev_dataset(args.output, args.media_dir, args.seed)
    print(json.dumps(json.loads(metadata.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    print(f"wrote {metadata}")


if __name__ == "__main__":
    main()

