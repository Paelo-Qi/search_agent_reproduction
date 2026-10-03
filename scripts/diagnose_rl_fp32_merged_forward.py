#!/usr/bin/env python3
"""Offline A0/A2/B0/B1/B2 forensic measurement; never a Gate run/update."""
from __future__ import annotations

import os
from diagnose_rl_merge_precision import ROOT, build_parser


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = build_parser()
    parser.description = __doc__
    parser._option_string_actions["--keep-diagnostic-model"].help = (
        "Retain this diagnostic's BF16/FP32 checkpoints (not for training/rollout)")
    parser.add_argument("--allow-sdpa-fp32-secondary", action="store_true",
                        help="Explicit exploratory backend-changed control; excluded from primary comparisons")
    args = parser.parse_args()
    if args.top_n < 1: raise ValueError("--top-n must be positive")
    from opensearch_vl_repro.rl.merge_precision_diagnostic import run_fp32_forward_diagnostic
    return run_fp32_forward_diagnostic(args, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
