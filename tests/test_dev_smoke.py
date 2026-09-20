from __future__ import annotations

import json
from pathlib import Path

from opensearch_vl_repro.config import load_config
from opensearch_vl_repro.data import IMAGE_MARKER, validate_raw_sample
from opensearch_vl_repro.dev_data import DEV_SEED, write_dev_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_dataset_is_local_valid_and_covers_observation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output, metadata_path = write_dev_dataset(
        data_dir / "sft_dev_4.json", data_dir / "dev_media", DEV_SEED
    )
    records = json.loads(output.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(records) == 4
    assert metadata["downloads_required"] is False
    assert metadata["sample_count"] == 4
    assert any(turn["from"] == "observation" for row in records for turn in row["conversations"])
    for sample in records:
        validate_raw_sample(sample)
        marker_count = sum(turn["value"].count(IMAGE_MARKER) for turn in sample["conversations"])
        assert marker_count == len(sample["images"]) == 1
        assert (output.parent / sample["images"][0]).is_file()


def test_synthetic_dataset_is_seed_deterministic(tmp_path: Path) -> None:
    first, _ = write_dev_dataset(
        tmp_path / "first" / "sft_dev_4.json",
        tmp_path / "first" / "dev_media",
        DEV_SEED,
    )
    second, _ = write_dev_dataset(
        tmp_path / "second" / "sft_dev_4.json",
        tmp_path / "second" / "dev_media",
        DEV_SEED,
    )
    assert first.read_bytes() == second.read_bytes()
    for index in range(4):
        assert (first.parent / "dev_media" / f"dev_{index}.png").read_bytes() == (
            second.parent / "dev_media" / f"dev_{index}.png"
        ).read_bytes()


def test_dev_and_formal_configs_keep_distinct_gates() -> None:
    dev = load_config(PROJECT_ROOT / "configs" / "sft_dev.yaml")
    formal = load_config(PROJECT_ROOT / "configs" / "sft_smoke.yaml")

    assert dev["model"]["name_or_path"] == formal["model"]["name_or_path"]
    assert dev["model"]["revision"] == formal["model"]["revision"]
    assert dev["data"]["expected_samples"] == 4
    assert dev["data"]["max_length"] == 4096
    assert dev["training"]["max_steps"] == 2
    assert dev["project"]["output_dir"].startswith("outputs/phase0_dev/")

    assert formal["data"]["expected_samples"] == 100
    assert formal["data"]["max_length"] == 32000
    assert formal["training"]["max_steps"] == 20
    assert formal["project"]["output_dir"].startswith("outputs/phase0/")
    assert "save_steps" not in formal["training"]
