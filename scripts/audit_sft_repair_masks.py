#!/usr/bin/env python3
"""CPU-only, real-processor repair mask audit; fail closed before GPU training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.agent.reliability import image_sha256  # noqa: E402
from opensearch_vl_repro.data import build_messages, load_json_records  # noqa: E402
from opensearch_vl_repro.evaluation.eval300 import build_eval300_plan  # noqa: E402
from opensearch_vl_repro.inference.eval_reader import read_eval_samples_by_ids  # noqa: E402
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.sft_main_data import (load_data_quality_exclusions,  # noqa: E402
                                                load_sft_manifest)
from opensearch_vl_repro.sft_preflight import normalized_question, sample_questions  # noqa: E402
from opensearch_vl_repro.sft_repair_data import validate_repair_manifest  # noqa: E402
from opensearch_vl_repro.sft_repair_mask import (RepairCollator, target_character_spans,  # noqa: E402
                                                  target_token_spans)
from opensearch_vl_repro.sft_repair_training import (load_repair_config,  # noqa: E402
                                                      require_legacy_repair_runtime)
from opensearch_vl_repro.sft_tool_audit import (sha256_file, write_json_atomic,  # noqa: E402
                                                 write_text_atomic)
from opensearch_vl_repro.sft_train_plan import load_main_config  # noqa: E402
from opensearch_vl_repro.tool_protocol_dev import (candidate_metadata,  # noqa: E402
                                                    source_zip_image_hasher)


def audit(config_path: Path, report_dir: Path, *, examples: int = 30) -> dict:
    require_legacy_repair_runtime()
    if examples < 20:
        raise ValueError("mask audit requires at least 20 representative samples")
    repair = load_repair_config(config_path)
    base = load_main_config(ROOT / repair["base_config"],
                            base_eval_config=ROOT / "configs/eval_base_300.yaml")
    data_path = ROOT / repair["dataset"]
    manifest = validate_repair_manifest(data_path, ROOT / repair["manifest"],
                                        mode=repair["repair_mode"])
    if (sha256_file(ROOT / "data/sft_main/manifest.json")
            != manifest["corrected_pool_manifest_sha256"]
            or sha256_file(ROOT / "data/eval/tool_protocol_dev50/"
                           "tool_protocol_dev50_manifest.json")
            != manifest["dev50_manifest_sha256"]):
        raise ValueError("repair audit provenance no longer matches corrected pool/Dev50")
    records = load_json_records(data_path)
    is_r3 = repair.get("repair_experiment") == "r3"
    if is_r3:
        if (manifest.get("repair_experiment") != "r3" or len(records) != 600
                or manifest.get("repair_mode") != "argument_only"):
            raise ValueError("R3 audit requires 600 corrected-pool argument-only samples")
        eval_path = ROOT / "data/eval/combined_eval_300.parquet"
        eval_plan = build_eval300_plan(eval_path)
        if eval_plan.dataset_sha256 != manifest.get("eval300_dataset_sha256"):
            raise ValueError("R3 Eval300 provenance changed")
        eval_samples = read_eval_samples_by_ids(eval_path, list(eval_plan.entries))
        eval_ids = {str(sample.sample_id) for sample in eval_samples}
        eval_questions = {normalized_question(sample.question) for sample in eval_samples}
        eval_hashes = {image_sha256(image) for sample in eval_samples
                       for image in sample.images}
        dev_dir = ROOT / "data/eval/tool_protocol_dev50"
        dev_ids = json.loads((dev_dir / "ids.json").read_text(encoding="utf-8"))
        external, _ = candidate_metadata(ROOT / "data/raw",
                                         load_sft_manifest(ROOT / "data/sft_main/manifest.json"),
                                         eval_questions, load_data_quality_exclusions())
        by_id = {item["sample_id"]: item for item in external}
        if not set(dev_ids).issubset(by_id):
            raise ValueError("R3 audit cannot locate all pinned Dev50 source records")
        dev_questions = set().union(*(sample_questions(by_id[identity]["record"])
                                      for identity in dev_ids))
        hasher = source_zip_image_hasher(ROOT / "data/raw")
        try:
            dev_hashes = {value for identity in dev_ids
                          for value in hasher(by_id[identity])}
        finally:
            hasher.close()  # type: ignore[attr-defined]
        for record, member in zip(records, manifest["membership"], strict=True):
            if (record["_sample_id"] in eval_ids or record["_sample_id"] in dev_ids
                    or sample_questions(record) & (eval_questions | dev_questions)):
                raise ValueError("R3 sample overlaps Eval300/Dev50 identity/question")
            actual_hashes = [image_sha256(data_path.parent / path)
                             for path in record["images"]]
            if (actual_hashes != member["image_sha256"]
                    or set(actual_hashes) & (eval_hashes | dev_hashes)):
                raise ValueError("R3 media hash changed or overlaps Eval300/Dev50")
    by_category = {}
    for index, record in enumerate(records):
        by_category.setdefault(record["_repair_category"], []).append(index)
    example_indices = set()
    for category in sorted(by_category):
        example_indices.update(by_category[category][:min(5, len(by_category[category]))])
    for index in range(len(records)):
        if len(example_indices) >= examples:
            break
        example_indices.add(index)
    processor = load_processor(base, local_files_only=True)
    collator = RepairCollator(processor, data_path, int(base["data"]["max_length"]),
                              repair["repair_mode"])
    rows = []
    supervised_total = 0
    tool_supervised_total = 0
    token_counts = []
    per_sample_tokens = []
    multi_call = 0
    exact_boundary_samples = 0
    token_expanded_samples = 0
    max_left_expansion_chars = 0
    max_right_expansion_chars = 0
    representative_expanded_spans = []
    for index, record in enumerate(records):
        batch = collator([record])
        full, spans = target_token_spans(processor, record, data_path, repair["repair_mode"])
        target_positions = {position for span in spans for position in
                            range(span["start"], span["end_exclusive"])}
        valid = batch.get("attention_mask")
        valid_indices = (valid[0].nonzero(as_tuple=True)[0].tolist() if valid is not None
                         else list(range(batch["input_ids"].shape[1])))
        actual_positions = {position for position, padded in enumerate(valid_indices)
                            if int(batch["labels"][0, padded]) != -100}
        if actual_positions != target_positions:
            raise ValueError(f"unexpected assistant tokens entered repair loss: {record['_sample_id']}")
        supervised_total += len(actual_positions)
        token_counts.append(len(actual_positions))
        sample_exact = all(span["exact_character_alignment"] for span in spans)
        if repair["repair_mode"] == "argument_only":
            exact_boundary_samples += int(sample_exact)
            token_expanded_samples += int(not sample_exact)
            for span in spans:
                left = span["left_expansion_text"]
                right = span["right_expansion_text"]
                max_left_expansion_chars = max(max_left_expansion_chars, len(left))
                max_right_expansion_chars = max(max_right_expansion_chars, len(right))
                if (left or right) and len(representative_expanded_spans) < 20:
                    representative_expanded_spans.append({
                        "sample_id": record["_sample_id"], "source": record["_source"],
                        "turn_index": span["turn_index"],
                        "supervised_decoded_span": span["decoded"],
                        "exact_character_alignment": False,
                        "left_expansion_text": left, "right_expansion_text": right})
        per_sample_tokens.append({"sample_id": record["_sample_id"],
                                  "category": record["_repair_category"],
                                  "supervised_token_count": len(actual_positions),
                                  "supervised_span_count": len(spans),
                                  "exact_character_alignment": sample_exact})
        tool_supervised_total += len(actual_positions)
        multi_call += int(len(spans) > 1)
        if index not in example_indices:
            continue
        assistant_text = "\n".join(turn["value"] for turn in record["conversations"]
                                   if turn["from"] == "gpt")
        messages, _, _ = build_messages(record, data_path)
        masked_spans = ([{"turn_index": None, "role": "system",
                          "excerpt": messages[0]["content"][:300]}]
                        if messages[0]["role"] == "system" else [])
        targets_by_turn = {}
        for (turn_index, start, end, _), span in zip(
                target_character_spans(record, repair["repair_mode"]), spans, strict=True):
            targets_by_turn.setdefault(turn_index, []).append((
                start - len(span["left_expansion_text"]),
                end + len(span["right_expansion_text"])))
        for turn_index, turn in enumerate(record["conversations"]):
            text = turn["value"]
            if turn["from"] != "gpt":
                masked_spans.append({"turn_index": turn_index, "role": turn["from"],
                                     "excerpt": text[:300]})
                continue
            cursor = 0
            for start, end in sorted(targets_by_turn.get(turn_index, [])):
                if cursor < start:
                    masked_spans.append({"turn_index": turn_index, "role": "gpt",
                                         "excerpt": text[cursor:start][:300]})
                cursor = end
            if cursor < len(text):
                masked_spans.append({"turn_index": turn_index, "role": "gpt",
                                     "excerpt": text[cursor:][:300]})
        first_target_turn = spans[0]["turn_index"]
        registered = [f"img_{i}" for i in range(
            1, record["conversations"][0]["value"].count("<image>") + 1)]
        for turn in record["conversations"][:first_target_turn]:
            if turn["from"] == "observation":
                registered.extend(re.findall(r"New image ID:\s*(img_[1-9][0-9]*)\b",
                                             turn["value"]))
        rows.append({"sample_id": record["_sample_id"], "source": record["_source"],
                     "repair_category": record["_repair_category"],
                     "raw_assistant_target": assistant_text,
                     "rendered_text": processor.tokenizer.decode(
                         full, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False),
                     "supervised_token_count": len(actual_positions),
                     "decoded_supervised_spans": spans,
                     "masked_spans": masked_spans,
                     "image_search_url_target": [row.get("target_image_ids") for row in
                                                 manifest["membership"] if row["sample_id"] ==
                                                 record["_sample_id"]][0],
                     "registered_image_ids": registered,
                     "span_alignment_exact": sample_exact,
                     "unexpected_assistant_tokens_in_loss": False})
    stats = {
        "passed": True, "repair_mode": repair["repair_mode"],
        "repair_experiment": repair.get("repair_experiment"),
        "repair_config_sha256": sha256_file(config_path),
        "repair_dataset_sha256": sha256_file(data_path),
        "repair_manifest_sha256": sha256_file(ROOT / repair["manifest"]),
        "sample_count": len(records), "source_counts": dict(sorted(Counter(
            record["_source"] for record in records).items())),
        "category_counts": dict(sorted(Counter(record["_repair_category"]
                                            for record in records).items())),
        "target_tool_counts": manifest["target_tool_counts"],
        "img_n_target_counts": manifest["image_target_counts"],
        "supervised_token_total": supervised_total,
        "mean_supervised_tokens_per_sample": supervised_total / len(records),
        "min_supervised_tokens_per_sample": min(token_counts),
        "max_supervised_tokens_per_sample": max(token_counts),
        "multi_image_search_call_samples": multi_call if repair["repair_mode"] ==
        "argument_only" else 0,
        "exact_boundary_sample_count": exact_boundary_samples,
        "token_expanded_sample_count": token_expanded_samples,
        "max_left_expansion_chars": max_left_expansion_chars,
        "max_right_expansion_chars": max_right_expansion_chars,
        "representative_expanded_spans": representative_expanded_spans,
        "tool_call_supervised_token_total": tool_supervised_total,
        "direct_answer_supervised_token_total": 0,
        "direct_answer_sample_count": 0,
        "per_sample_supervision": per_sample_tokens,
        "examples": rows,
    }
    if is_r3:
        stats["sample_image_target_coverage_counts"] = manifest[
            "sample_image_target_coverage_counts"]
        stats["image_target_occurrence_category_counts"] = manifest[
            "image_target_occurrence_category_counts"]
        stats["total_supervised_image_search_argument_occurrences"] = manifest[
            "total_supervised_image_search_argument_occurrences"]
        stats["candidate_rejections"] = manifest["candidate_rejections"]
        stats["selection_rejections"] = manifest["selection"]["selection_rejections"]
        stats["overlap_counts"] = {key: manifest[key] for key in (
            "eval300_id_overlap_count", "eval300_question_overlap_count",
            "eval300_image_overlap_count", "dev50_id_overlap_count",
            "dev50_question_overlap_count", "dev50_image_overlap_count",
            "frozen_exclusion_overlap_count", "image_contract_bad_count")}
    prefix = "r3" if is_r3 else "r1" if repair["repair_mode"] == "argument_only" else "r2"
    write_json_atomic(report_dir / f"{prefix}_mask_audit.json", stats)
    lines = [f"# {prefix.upper()} targeted repair mask audit", "",
             f"Samples: {len(records)}; supervised tokens: {supervised_total}; "
             f"exact-boundary samples: {exact_boundary_samples}; "
             f"token-expanded samples: {token_expanded_samples}", "",
             "## Representative supervised spans", ""]
    for row in rows:
        lines.extend([f"### {row['sample_id']} ({row['repair_category']})", "",
                      f"Tokens: {row['supervised_token_count']}", "",
                      "```text", *(span["decoded"] for span in row["decoded_supervised_spans"]),
                      "```", ""])
    write_text_atomic(report_dir / f"{prefix}_mask_audit.md", "\n".join(lines) + "\n")
    return stats


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--report-dir", type=Path, default=ROOT / "reports/sft_repair")
    cli.add_argument("--examples", type=int, default=30)
    args = cli.parse_args()
    stats = audit(args.config, args.report_dir, examples=args.examples)
    print(json.dumps({key: value for key, value in stats.items()
                      if key not in ("examples", "per_sample_supervision")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
