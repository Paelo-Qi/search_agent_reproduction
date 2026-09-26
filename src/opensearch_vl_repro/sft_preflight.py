"""Read-only SFT leakage, sequence, and runtime tool-contract audits."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .agent.question_normalization import normalize_model_question
from .agent.reliability import image_sha256
from .agent.tool_contracts import TOOL_DECLARATIONS, TOOL_DECLARATIONS_BY_NAME
from .agent.tool_parser import ToolCallParser
from .data import (SFT_INPUT_MESSAGE_VERSION, SFT_MASK_VERSION, RoleTokenSpan, build_messages, message_role_spans, parse_tools,
                   render_prompt)
from .sft_image_grounding import audit_sample_image_grounding, summarize_image_grounding


RESERVED_CHAT_LITERALS = ("<|im_start|>", "<|im_end|>")


def reserved_literal_audit(records_by_shard: dict[str, list[dict[str, Any]]],
                           max_examples: int = 20) -> dict[str, Any]:
    """Read-only role-wise count of reserved spellings in source message bodies."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    role_names = {"human": "user", "gpt": "assistant", "observation": "tool"}
    for shard, records in records_by_shard.items():
        for record in records:
            turns = ([(-1, "system", record.get("system", ""))]
                     + [(index, role_names[turn["from"]], turn["value"])
                        for index, turn in enumerate(record["conversations"])])
            for turn_index, role, text in turns:
                for literal in RESERVED_CHAT_LITERALS:
                    count = text.count(literal)
                    if not count:
                        continue
                    counts[role][literal] += count
                    if len(examples) < max_examples:
                        examples.append({"sample_id": record["_sample_id"], "shard": shard,
                                         "turn_index": turn_index, "role": role,
                                         "literal": literal, "count": count})
    return {"literals": list(RESERVED_CHAT_LITERALS),
            "counts_by_role": {role: dict(counts[role])
                               for role in ("system", "user", "assistant", "tool")},
            "example_count": len(examples), "examples": examples}


def normalized_question(value: str) -> str:
    return " ".join(normalize_model_question(value).replace("<image>", " ").split()).casefold()


def sample_questions(record: dict[str, Any]) -> set[str]:
    questions = {normalized_question(turn["value"]) for turn in record["conversations"]
                 if turn["from"] == "human"}
    if not questions:
        raise ValueError("SFT sample has no human question")
    return questions


def leakage_audit(records: Iterable[dict[str, Any]], eval_samples: Iterable[Any],
                  image_root: str | Path) -> dict[str, Any]:
    eval_questions: dict[str, list[dict[str, str]]] = defaultdict(list)
    eval_images: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in eval_samples:
        reference = {"eval_sample_id": sample.sample_id, "benchmark": sample.benchmark}
        eval_questions[normalized_question(sample.question)].append(reference)
        for image in sample.images:
            eval_images[image_sha256(image)].append(reference)
    question_overlaps, image_overlaps, missing_images = [], [], []
    root = Path(image_root)
    for record in records:
        identity = {"training_sample_id": record["_sample_id"], "source": record["_source"]}
        for question in sample_questions(record):
            for reference in eval_questions.get(question, []):
                question_overlaps.append({**identity, **reference})
        for image_path in record["images"]:
            path = root / image_path
            if not path.is_file():
                missing_images.append({**identity, "image_path": str(path)})
                continue
            image_hash = image_sha256(path)
            for reference in eval_images.get(image_hash, []):
                image_overlaps.append({**identity, **reference, "image_sha256": image_hash})
    return {
        "question_overlap_count": len(question_overlaps), "question_overlaps": question_overlaps,
        "image_overlap_count": len(image_overlaps), "image_overlaps": image_overlaps,
        "missing_image_count": len(missing_images), "missing_images": missing_images,
        "complete": not missing_images,
    }


def _declaration_mismatches(value: Any, identity: dict[str, str]) -> list[dict[str, Any]]:
    mismatches = []
    parsed = parse_tools(value)
    expected_tools = [tool.as_chat_template_tool() for tool in TOOL_DECLARATIONS]
    if parsed == expected_tools:
        return []
    for declared_tool in parsed:
            function = declared_tool.get("function", {}) if isinstance(declared_tool, dict) else {}
            name = function.get("name")
            if not isinstance(name, str):
                mismatches.append({**identity, "kind": "invalid_declaration", "tool": repr(name)})
                continue
            runtime = TOOL_DECLARATIONS_BY_NAME.get(name)
            if runtime is None:
                mismatches.append({**identity, "kind": "undeclared_runtime_tool", "tool": name})
                continue
            parameters = function.get("parameters", {})
            if not isinstance(parameters, dict):
                mismatches.append({**identity, "kind": "invalid_declaration", "tool": name})
                continue
            expected = runtime.parameters
            actual_properties = parameters.get("properties", {})
            if not isinstance(actual_properties, dict):
                mismatches.append({**identity, "kind": "invalid_declaration", "tool": name})
                continue
            actual_types = {key: value.get("type") if isinstance(value, dict) else None
                            for key, value in actual_properties.items()}
            expected_types = {key: value["type"] for key, value in expected["properties"].items()}
            # Dataset integer coordinates are accepted by runtime's number schema.
            incompatible_types = {
                key: {"dataset": value, "runtime": expected_types.get(key)}
                for key, value in actual_types.items()
                if key in expected_types and value != expected_types[key]
                and not (value == "integer" and expected_types[key] == "number")
            }
            missing_runtime = sorted(set(expected_types) - set(actual_types))
            extra_dataset = sorted(set(actual_types) - set(expected_types))
            actual_required = set(parameters.get("required", []))
            runtime_required = set(expected.get("required", []))
            if incompatible_types or missing_runtime or extra_dataset or actual_required != runtime_required:
                mismatches.append({
                    **identity, "kind": "declaration_schema_drift", "tool": name,
                    "missing_runtime_properties": missing_runtime,
                    "extra_dataset_properties": extra_dataset,
                    "incompatible_types": incompatible_types,
                    "dataset_required": sorted(actual_required),
                    "runtime_required": sorted(runtime_required),
                })
    # The full chat-template declaration includes descriptions, order, and
    # metadata, not merely parameter types. Detect any otherwise unreported drift.
    if not mismatches:
        mismatches.append({**identity, "kind": "declaration_template_drift"})
    return mismatches


def tool_contract_audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    parser = ToolCallParser(TOOL_DECLARATIONS_BY_NAME)
    calls = Counter()
    raw_mismatches, effective_mismatches, call_mismatches = [], [], []
    declared = Counter()
    for record in records:
        identity = {"training_sample_id": record["_sample_id"], "source": record["_source"]}
        raw_mismatches.extend(_declaration_mismatches(
            record.get("_source_tools", record.get("tools")), identity))
        effective_mismatches.extend(_declaration_mismatches(record.get("tools"), identity))
        for declared_tool in parse_tools(record.get("tools")):
            name = declared_tool.get("function", {}).get("name") if isinstance(declared_tool, dict) else None
            if isinstance(name, str):
                declared[name] += 1
        for turn_index, turn in enumerate(record["conversations"]):
            if turn["from"] != "gpt":
                continue
            parsed = parser.parse(turn["value"])
            if parsed.kind == "final_answer":
                continue
            if parsed.kind != "valid_tool_call":
                call_mismatches.append({**identity, "kind": parsed.kind, "turn_index": turn_index,
                                   "detail": parsed.error})
                continue
            for call in parsed.tool_calls:
                calls[call.name] += 1
                runtime = TOOL_DECLARATIONS_BY_NAME[call.name]
                try:
                    runtime.validate_arguments(call.arguments)
                except ValueError as exc:
                    call_mismatches.append({**identity, "kind": "call_schema_drift",
                                       "turn_index": turn_index, "tool": call.name,
                                       "detail": str(exc), "arguments": call.arguments})
    mismatches = effective_mismatches + call_mismatches
    return {"passed": not mismatches, "tool_call_counts": dict(sorted(calls.items())),
            "declared_tool_counts": dict(sorted(declared.items())),
            "raw_declaration_drift_count": len(raw_mismatches),
            "effective_declaration_drift_count": len(effective_mismatches),
            "actual_call_drift_count": len(call_mismatches),
            "raw_declaration_mismatches": raw_mismatches,
            "effective_declaration_mismatches": effective_mismatches,
            "actual_call_mismatches": call_mismatches,
            "mismatch_count": len(mismatches),
            "mismatch_kind_counts": dict(Counter(item["kind"] for item in mismatches)),
            "mismatch_tool_counts": dict(Counter(item.get("tool", "unknown") for item in mismatches)),
            "mismatches": mismatches}


def token_audit(input_ids: list[int], truncated_ids: list[int],
                spans: list[RoleTokenSpan], messages: list[dict[str, Any]],
                tokenizer: Any) -> dict[str, Any]:
    """Audit truncation against the same role-derived spans used by the collator."""
    if truncated_ids != input_ids[:len(truncated_ids)]:
        raise ValueError("multimodal processor truncation is not a right-hand prefix")
    cutoff = len(truncated_ids)
    supervised = sum(max(0, min(span.body_end, cutoff) - span.body_start)
                     for span in spans)
    partial = any(span.body_start < cutoff < span.body_end for span in spans)
    dropped = sum(span.body_start >= cutoff for span in spans)
    partial_tool = False
    dropped_tool = 0
    tool_spans = 0
    for span in spans:
        message = messages[span.message_index]
        content = message["content"]
        raw_text = (content if isinstance(content, str) else "".join(
            part.get("text", "") for part in content if part.get("type") == "text"))
        full_text = tokenizer.decode(input_ids[span.body_start:span.body_end],
                                     skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        calls = list(re.finditer(r"<tool_call>.*?</tool_call>", full_text, re.DOTALL))
        expected = raw_text.count("<tool_call>")
        if (len(calls) != expected or full_text.count("<tool_call>") != expected
                or full_text.count("</tool_call>") != expected):
            raise RuntimeError(f"assistant tool-call text is not preserved at message {span.message_index}")
        tool_spans += len(calls)
        if span.body_start >= cutoff:
            dropped_tool += len(calls)
        elif span.body_start < cutoff < span.body_end and calls:
            visible = tokenizer.decode(input_ids[span.body_start:cutoff],
                                       skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            partial_tool |= any(call.start() < len(visible) < call.end() for call in calls)
    return {
        "token_length": len(input_ids), "truncated_length": len(truncated_ids),
        "supervised_tokens_after_truncation": supervised,
        "zero_supervised_tokens": supervised == 0,
        "partial_assistant_span_cut": partial,
        "partial_tool_call_cut": partial_tool,
        "complete_assistant_span_dropped": dropped,
        "complete_tool_call_turn_dropped": dropped_tool,
        "tool_call_span_count": tool_spans,
        "last_complete_assistant_end_before_cutoff": max(
            (span.body_end for span in spans if span.body_end <= cutoff), default=None),
    }


def _percentile(values: list[int], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def sequence_summary(rows: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("sequence audit needs at least one sample")
    lengths = sorted(int(row["token_length"]) for row in rows)
    over = sum(value > max_length for value in lengths)
    return {
        "count": len(rows), "max_length": max_length,
        "min": lengths[0], "mean": statistics.mean(lengths),
        "median": statistics.median(lengths), "p50": _percentile(lengths, .5),
        "p90": _percentile(lengths, .9), "p95": _percentile(lengths, .95),
        "p99": _percentile(lengths, .99), "max": lengths[-1],
        "count_over_max_length": over, "ratio_over_max_length": over / len(rows),
        "zero_supervised_count": sum(bool(row["zero_supervised_tokens"]) for row in rows),
        "partial_assistant_span_cut_count": sum(bool(row["partial_assistant_span_cut"]) for row in rows),
        "partial_tool_call_cut_count": sum(bool(row["partial_tool_call_cut"]) for row in rows),
        "complete_assistant_span_dropped_count": sum(
            int(row["complete_assistant_span_dropped"]) for row in rows),
        "complete_tool_call_turn_dropped_count": sum(
            int(row["complete_tool_call_turn_dropped"]) for row in rows),
    }


def sequence_audit(records_by_shard: dict[str, list[dict[str, Any]]],
                   processor: Any, data_dir: str | Path,
                   max_length: int = 32000) -> dict[str, Any]:
    """Use the actual multimodal processor/template, never text-only estimates."""
    rows_by_shard: dict[str, list[dict[str, Any]]] = {}
    grounding_rows: list[dict[str, Any]] = []
    for shard, records in records_by_shard.items():
        rows = []
        dataset_path = Path(data_dir) / f"{shard}.json"
        for record in records:
            messages, images, tools = build_messages(record, dataset_path)
            grounding_rows.append(audit_sample_image_grounding(record, messages))
            prompt = render_prompt(processor, messages, tools)
            full = processor(text=[prompt], images=[images], padding=False,
                             truncation=False, return_tensors="pt")
            truncated = processor(text=[prompt], images=[images], padding=False,
                                  truncation=True, max_length=max_length, return_tensors="pt")
            full_ids = full["input_ids"][0].tolist()
            spans = message_role_spans(processor, messages, images, tools, full_ids)
            row = token_audit(full_ids, truncated["input_ids"][0].tolist(),
                              spans, messages, processor.tokenizer)
            row.update(sample_id=record["_sample_id"], source=record["_source"], shard=shard)
            rows.append(row)
        rows_by_shard[shard] = rows
    all_rows = [row for rows in rows_by_shard.values() for row in rows]
    return {"mask_version": SFT_MASK_VERSION,
            "sft_input_message_version": SFT_INPUT_MESSAGE_VERSION,
            "image_grounding": summarize_image_grounding(grounding_rows),
            "shards": {shard: sequence_summary(rows, max_length)
                       for shard, rows in rows_by_shard.items()},
            "full_8k": sequence_summary(all_rows, max_length),
            "problem_samples": [row for row in all_rows if row["zero_supervised_tokens"]
                                or row["partial_assistant_span_cut"] or row["partial_tool_call_cut"]],
            "report_only_truncation": [row for row in all_rows
                                       if row["complete_assistant_span_dropped"]
                                       and not (row["zero_supervised_tokens"]
                                                or row["partial_assistant_span_cut"]
                                                or row["partial_tool_call_cut"])]}


def formal_preflight_checks(tool: dict[str, Any], leakage: dict[str, Any],
                            sequence: dict[str, Any] | None) -> dict[str, bool]:
    full = sequence.get("full_8k", {}) if sequence else {}
    return {
        "effective_tool_contract": tool.get("effective_declaration_drift_count") == 0,
        "actual_tool_calls": tool.get("actual_call_drift_count") == 0,
        "leakage_complete": leakage.get("complete") is True,
        "zero_question_overlap": leakage.get("question_overlap_count") == 0,
        "zero_image_overlap": leakage.get("image_overlap_count") == 0,
        "sequence_complete": sequence is not None,
        "supervised_targets": full.get("zero_supervised_count") == 0,
        "assistant_spans_intact": full.get("partial_assistant_span_cut_count") == 0,
        "tool_calls_intact": full.get("partial_tool_call_cut_count") == 0,
        "image_id_grounding": bool(sequence and sequence.get("image_grounding", {}).get("passed")),
    }
