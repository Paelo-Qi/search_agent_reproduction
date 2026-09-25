#!/usr/bin/env python3
"""Display 4B smoke A/B measurements without choosing a formal micro-batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()
    output = {}
    for name, path in (("A", args.a), ("B", args.b)):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("stage") != "smoke" or not report.get("passed"):
            raise ValueError(f"{name} is not a completed 4B smoke report")
        output[name] = {
            "micro_batch": report["micro_batch"],
            "gradient_accumulation": report["gradient_accumulation"],
            "effective_global_batch": report["effective_global_batch"],
            "optimizer_steps": report["global_step"],
            "all_losses_finite": report["all_losses_finite"],
            "peak_allocated_bytes_overall": report["peak_allocated_bytes_overall"],
            "peak_reserved_bytes_overall": report["peak_reserved_bytes_overall"],
            "seconds_per_optimizer_step": report["seconds_per_optimizer_step"],
            "mean_step_time_seconds": report["mean_step_time_seconds"],
            "samples_per_second": report["samples_per_second"],
            "checkpoint": report["checkpoint"],
        }
    if {row["effective_global_batch"] for row in output.values()} != {8}:
        raise ValueError("A/B effective global batch must both be 8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
