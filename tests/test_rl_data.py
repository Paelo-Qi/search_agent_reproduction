from PIL import Image
import pytest

from opensearch_vl_repro.rl.data import overlap_audit, prepare_prompts


def records(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "red").save(image)
    return [{"source_sample_id": str(i), "question": f"What is object {i}?",
             "image_paths": [str(image)]} for i in range(3)]


def test_deterministic_membership_and_provenance(tmp_path):
    source = records(tmp_path)
    first, manifest = prepare_prompts(source, dataset_id="local", dataset_revision="pinned",
                                       seed=12, limit=2, validation_count=1)
    second, repeated = prepare_prompts(list(reversed(source)), dataset_id="local",
                                       dataset_revision="pinned", seed=12, limit=2,
                                       validation_count=1)
    assert first == second and manifest == repeated
    assert {item.split for item in first} == {"train", "validation"}
    assert all(len(item.question_hash) == 64 and len(item.image_hashes[0]) == 64 for item in first)
    assert len(manifest["manifest_sha256"]) == 64


def test_overlap_audit_interface(tmp_path):
    samples, _ = prepare_prompts(records(tmp_path), dataset_id="local",
                                 dataset_revision="pinned", seed=1, limit=2)
    result = overlap_audit(samples, known_question_hashes={samples[0].question_hash},
                           known_image_hashes={samples[1].image_hashes[0]})
    assert samples[0].sample_id in result["question_overlap_ids"]
    assert set(result["image_overlap_ids"]) == {item.sample_id for item in samples}


def test_duplicate_or_missing_source_fails(tmp_path):
    source = records(tmp_path)
    with pytest.raises(ValueError):
        prepare_prompts([source[0], source[0]], dataset_id="local",
                        dataset_revision="pinned", seed=1, limit=2)
    with pytest.raises(FileNotFoundError):
        prepare_prompts([{"source_sample_id": "bad", "question": "q",
                          "image_paths": [str(tmp_path / "missing.png")]}],
                        dataset_id="local", dataset_revision="pinned", seed=1, limit=1)
