"""Read-only Eval/SFT source audit; writes independent RL overlap hash lists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.eval_subset import sha256_file
from opensearch_vl_repro.evaluation.eval300 import build_eval300_plan
from opensearch_vl_repro.inference.eval_reader import read_eval_samples_by_ids
from opensearch_vl_repro.rl.data import make_overlap_manifest, question_sha256
from opensearch_vl_repro.sft_main_data import SHARD_SIZES, load_sft_manifest
from opensearch_vl_repro.sft_preflight import sample_questions


def build_eval(path: Path) -> dict:
    plan = build_eval300_plan(path)
    samples = read_eval_samples_by_ids(path, list(plan.entries))
    return make_overlap_manifest(kind="eval", source_sha256=plan.dataset_sha256,
        question_hashes={question_sha256(item.question) for item in samples},
        image_hashes={image_sha256(image) for item in samples for image in item.images},
        image_audit_complete=True)


def build_sft(directory: Path, *, allow_missing_images: bool = False) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = load_sft_manifest(manifest_path)
    question_hashes, image_hashes = set(), set()
    missing = 0
    for name in SHARD_SIZES:
        shard = directory / manifest["shards"][name]["path"]
        if sha256_file(shard) != manifest["shards"][name]["sha256"]:
            raise ValueError(f"SFT shard checksum mismatch: {name}")
        records = json.loads(shard.read_text(encoding="utf-8"))
        for record in records:
            question_hashes.update(question_sha256(question) for question in sample_questions(record))
            for relative in record["images"]:
                image_path = directory / relative
                if not image_path.is_file():
                    missing += 1
                else:
                    image_hashes.add(image_sha256(image_path))
    if missing and not allow_missing_images:
        raise FileNotFoundError(f"SFT image overlap audit incomplete: {missing} missing images")
    return make_overlap_manifest(kind="sft", source_sha256=sha256_file(manifest_path),
        question_hashes=question_hashes, image_hashes=image_hashes,
        image_audit_complete=missing == 0, missing_image_count=missing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["eval", "sft"], required=True)
    parser.add_argument("--eval-parquet", type=Path)
    parser.add_argument("--sft-dir", type=Path)
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="development fixture only; result cannot pass formal RL preflight")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite overlap manifest: {args.output}")
        if args.kind == "eval":
            if args.eval_parquet is None or args.sft_dir is not None:
                raise ValueError("eval requires only --eval-parquet")
            result = build_eval(args.eval_parquet)
        else:
            if args.sft_dir is None or args.eval_parquet is not None:
                raise ValueError("sft requires only --sft-dir")
            result = build_sft(args.sft_dir, allow_missing_images=args.allow_missing_images)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, KeyError, FileNotFoundError, TypeError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"passed": True, "kind": args.kind,
                      "question_hash_count": len(result["question_hashes"]),
                      "image_hash_count": len(result["image_hashes"]),
                      "image_audit_complete": result["image_audit_complete"],
                      "manifest_sha256": result["manifest_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
