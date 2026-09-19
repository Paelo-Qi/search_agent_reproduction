#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import ijson
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opensearch_vl_repro.data import validate_raw_sample  # noqa: E402


DATASET_ID = "OpenSearch-VL/Search-VL-SFT-36K"
DATASET_REVISION = "2c1c460af4fa15bd63210cbf426a96664b959944"
SOURCE_FILES = {
    "fvqa": "fvqa/fvqa_llama_factory_clean.json",
    "livevqa": "livevqa/livevqa_llama_factory_filtered.json",
    "palace": "palace/palace_llama_factory_filtered.json",
    "webqa": "webqa/webqa_llama_factory_filtered.json",
    "wiki_art": "wiki_art/wikiart_llama_factory_filtered.json",
    "wiki_en": "wiki_en/wiki_en_llama_factory_filtered.json",
    "wiki_zh": "wiki_zh/wiki_zh_llama_factory_filtered.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic, stratified Search-VL-SFT smoke set."
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "sft_smoke_100.json"
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--media-dir", type=Path, default=PROJECT_ROOT / "data" / "media")
    parser.add_argument(
        "--cleanup-downloads",
        action="store_true",
        help="Remove the exact downloaded JSON/ZIP files after successful extraction.",
    )
    return parser.parse_args()


def allocation(total: int, sources: list[str]) -> dict[str, int]:
    if total < len(sources):
        raise ValueError(f"count must be at least {len(sources)} to cover every source")
    base, remainder = divmod(total, len(sources))
    return {name: base + (index < remainder) for index, name in enumerate(sources)}


def download(filename: str, raw_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=raw_dir,
        )
    )


def reservoir_sample(path: Path, size: int, rng: random.Random) -> tuple[list[dict[str, Any]], int, int]:
    chosen: list[dict[str, Any]] = []
    valid_count = 0
    invalid_count = 0
    with path.open("rb") as handle:
        for source_index, sample in enumerate(ijson.items(handle, "item")):
            try:
                validate_raw_sample(sample, require_images=True)
            except (TypeError, ValueError, json.JSONDecodeError):
                invalid_count += 1
                continue
            sample["_source_index"] = source_index
            if valid_count < size:
                chosen.append(sample)
            else:
                replacement = rng.randrange(valid_count + 1)
                if replacement < size:
                    chosen[replacement] = sample
            valid_count += 1
    if len(chosen) != size:
        raise RuntimeError(f"{path} yielded only {len(chosen)} valid records; need {size}")
    return chosen, valid_count, invalid_count


def resolve_zip_member(reference: str, names: set[str], basename_map: dict[str, list[str]]) -> str:
    normalized = str(PurePosixPath(reference.replace("\\", "/"))).lstrip("./")
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.split("/", 1)[1])
    candidates.append(f"images/{PurePosixPath(normalized).name}")
    for candidate in candidates:
        if candidate in names:
            return candidate
    matches = basename_map.get(PurePosixPath(normalized).name, [])
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot uniquely resolve {reference!r} inside image archive")


def extract_selected_images(
    archive_path: Path,
    source: str,
    samples: list[dict[str, Any]],
    media_dir: Path,
    output_dir: Path,
) -> int:
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [item.filename for item in archive.infolist() if not item.is_dir()]
        names = set(members)
        basename_map: dict[str, list[str]] = {}
        for name in members:
            basename_map.setdefault(PurePosixPath(name).name, []).append(name)

        for sample in samples:
            local_images: list[str] = []
            original_images = list(sample["images"])
            for reference in original_images:
                member = resolve_zip_member(reference, names, basename_map)
                suffix = Path(member).suffix.lower() or ".img"
                digest = hashlib.sha256(f"{source}:{member}".encode("utf-8")).hexdigest()[:16]
                target = media_dir / source / f"{digest}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with archive.open(member) as source_stream, target.open("wb") as target_stream:
                        shutil.copyfileobj(source_stream, target_stream)
                    extracted += 1
                local_images.append(target.relative_to(output_dir).as_posix())
            sample["_source_images"] = original_images
            sample["images"] = local_images
    return extracted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    raw_dir = args.raw_dir.resolve()
    media_dir = args.media_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    sources = list(SOURCE_FILES)
    quotas = allocation(args.count, sources)
    selected_by_source: dict[str, list[dict[str, Any]]] = {}
    downloaded: list[Path] = []
    source_stats: dict[str, dict[str, int]] = {}

    for source, filename in SOURCE_FILES.items():
        print(f"[{source}] downloading metadata: {filename}", flush=True)
        json_path = download(filename, raw_dir)
        downloaded.append(json_path)
        records, valid_count, invalid_count = reservoir_sample(
            json_path, quotas[source], random.Random(f"{args.seed}:{source}")
        )
        for sample in records:
            sample["_source"] = source
            sample["_source_file"] = filename
        selected_by_source[source] = records
        source_stats[source] = {
            "selected": len(records),
            "valid_records_seen": valid_count,
            "invalid_records_skipped": invalid_count,
        }

    total_extracted = 0
    for source, samples in selected_by_source.items():
        archive_name = f"{source}/images.zip"
        print(f"[{source}] downloading published image archive: {archive_name}", flush=True)
        archive_path = download(archive_name, raw_dir)
        downloaded.append(archive_path)
        total_extracted += extract_selected_images(
            archive_path, source, samples, media_dir, output.parent
        )

    records = [record for source in sources for record in selected_by_source[source]]
    random.Random(args.seed).shuffle(records)
    if len(records) != args.count:
        raise AssertionError(f"prepared {len(records)} records, expected {args.count}")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    meta_path = output.with_suffix(".meta.json")
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "seed": args.seed,
        "sample_count": len(records),
        "source_counts": dict(Counter(record["_source"] for record in records)),
        "source_stats": source_stats,
        "extracted_file_count": total_extracted,
        "output_sha256": sha256_file(output),
        "published_packaging_note": (
            "Images are distributed as one ZIP per source; only referenced members were extracted."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if args.cleanup_downloads:
        for path in downloaded:
            resolved = path.resolve()
            if raw_dir == resolved or raw_dir not in resolved.parents:
                raise RuntimeError(f"refusing to remove path outside raw dir: {resolved}")
            resolved.unlink(missing_ok=True)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

