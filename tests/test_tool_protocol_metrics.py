from __future__ import annotations

import pytest
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.evaluation.eval300 import FROZEN_EVAL300_SHA256
from opensearch_vl_repro.sft_protocol_diagnostics import CORRECTED_POOL_MANIFEST_SHA256
from opensearch_vl_repro.sft_tool_audit import DATASET_ID, DATASET_REVISION
from opensearch_vl_repro.tool_protocol_metrics import protocol_metrics


def _manifest():
    ids = ["fvqa:1", "fvqa:2"]
    samples = [{"sample_id": ids[0], "source": "fvqa", "protocol_tags": ["image_search_img_1"],
                "expected_tool_label": "image_search"},
               {"sample_id": ids[1], "source": "fvqa", "protocol_tags": ["multi_tool"],
                "expected_tool_label": None}]
    return ids, {"version": 2,
                 "runtime_tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
                 "count": 2, "samples": samples, "ids_sha256": canonical_json_sha256(ids),
                 "sample_metadata_sha256": canonical_json_sha256(samples),
                 "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                 "corrected_pool_manifest_sha256": CORRECTED_POOL_MANIFEST_SHA256,
                 "eval300_dataset_sha256": FROZEN_EVAL300_SHA256,
                 "dev30_subset_of_eval300": True,
                 **{key: 0 for key in ("eval300_question_overlap_count", "eval300_image_overlap_count",
                                        "sft_membership_overlap_count", "frozen_exclusion_overlap_count",
                                        "image_contract_bad_count")}}


def _turn(name, argument, error=None, provider=None, derived=()):
    key = "image_id" if name == "image_search" else "image"
    return {"tool_call": {"name": name, "arguments": {key: argument}}, "error": error,
            "metadata": {"provider_called": provider},
            "derived_images": [{"image_id": value} for value in derived]}


def _record(sample_id, turns):
    return {"sample_id": sample_id, "trajectory": {"status": "success", "turns": turns,
            "images": [{"image_id": "img_1", "kind": "initial"},
                       {"image_id": "img_2", "kind": "derived"}]}}


def test_protocol_metrics_valid_unknown_http_path_duplicate_and_order_independent():
    ids, manifest = _manifest()
    first = _record(ids[0], [_turn("image_search", "img_1")])
    second = _record(ids[1], [
        _turn("crop", "img_1", derived=["img_2"]),
        _turn("layout_parsing", "img_2"),
        _turn("image_search", "https://example.com/a.png", "unknown_image_id", False),
        _turn("crop", "file.png", "unknown_image_id", False),
        _turn("image_search", "img_1", "duplicate_tool_call", False),
    ])
    a = protocol_metrics(ids, manifest, [first, second])
    b = protocol_metrics(ids, manifest, [second, first])
    assert a == b
    metric = a["metrics"]
    assert metric["image_tool_argument_count"] == 6
    assert metric["registered_image_id_count"] == 4
    assert metric["registered_image_id_rate"] == pytest.approx(4/6)
    assert metric["http_image_argument_hallucination_count"] == 1
    assert metric["image_search_http_image_id_hallucination_count"] == 1
    assert metric["image_search_image_id_argument_count"] == 3
    assert metric["image_search_legacy_url_argument_count"] == 0
    assert metric["unknown_image_id_count"] == 2
    assert metric["provider_not_called_due_to_bad_image_id"] == 2
    assert metric["duplicate_tool_call_count"] == 1
    assert metric["image_search_valid_argument_rate"] == pytest.approx(2/3)
    assert metric["layout_parsing_valid_argument_rate"] == 1
    assert metric["crop_valid_argument_rate"] == 0.5
    assert metric["tool_selection_agreement"] == 1
    with pytest.raises(ValueError, match="outside tool-protocol dev set"):
        protocol_metrics(ids, manifest, [first, _record("eval300:1", [])])


def test_protocol_metrics_reports_direct_answer_without_forcing_tool_label():
    ids, manifest = _manifest()
    manifest["samples"][0]["expected_tool_label"] = None
    manifest["sample_metadata_sha256"] = canonical_json_sha256(manifest["samples"])
    result = protocol_metrics(ids, manifest, [_record(ids[0], []), _record(ids[1], [])])
    assert result["metrics"]["no_tool_behavior_count"] == 2
    assert result["metrics"]["tool_selection_agreement"] is None
    assert result["metrics"]["tool_call_count_distribution"] == {0: 2}


def test_protocol_metrics_audits_legacy_key_and_http_in_image_id():
    ids, manifest = _manifest()
    legacy = _turn("image_search", "img_1")
    legacy["tool_call"]["arguments"] = {"url": "img_1"}
    http = _turn("image_search", "https://example.com/a.jpg", "unknown_image_id", False)
    result = protocol_metrics(ids, manifest, [
        _record(ids[0], [legacy]), _record(ids[1], [http])])
    metrics = result["metrics"]
    assert metrics["image_search_legacy_url_argument_count"] == 1
    assert metrics["image_search_image_id_argument_count"] == 1
    assert metrics["http_image_argument_hallucination_count"] == 1
    assert metrics["image_search_http_image_id_hallucination_count"] == 1
