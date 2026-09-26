"""Read-only diagnostics for the corrected SFT tool protocol."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .agent.question_normalization import normalize_model_question
from .agent.runtime import AGENT_SYSTEM_GUIDANCE, AgentRuntime
from .agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from .agent.tool_parser import TOOL_CALL_BLOCK, ToolCallParser
from .data import (IMAGE_MARKER, OpenSearchVLCollator, build_messages, load_json_records,
                   messages_to_json_safe, render_prompt)
from .sft_main_data import (load_sft_manifest, require_data_quality_exclusions,
                            runtime_tools)
from .sft_tool_audit import sha256_file


CORRECTED_POOL_MANIFEST_SHA256 = "0faf210483978435e808e4ba8ce4fb2556fb27ecfe30267e948f5cc2f1c9c637"
PARSER = ToolCallParser(TOOL_DECLARATIONS_BY_NAME)
IMAGE_TOOLS = frozenset(("image_search", "layout_parsing", "crop", "sharpen",
                         "super_resolution", "perspective_correct"))
URL_PATTERNS = {
    "http": re.compile(r"http://", re.I), "https": re.compile(r"https://", re.I),
    "i.imgur.com": re.compile(r"i\.imgur\.com", re.I),
    "imgur": re.compile(r"imgur", re.I), "jpg": re.compile(r"\.jpg\b", re.I),
    "jpeg": re.compile(r"\.jpeg\b", re.I), "png": re.compile(r"\.png\b", re.I),
    "webp": re.compile(r"\.webp\b", re.I),
    "upload.wikimedia": re.compile(r"upload\.wikimedia", re.I),
    "images.unsplash": re.compile(r"images\.unsplash", re.I),
    "domain": re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|cn|io|edu)\b", re.I),
    '"url":': re.compile(r'"url"\s*:', re.I),
}


def load_corrected_shards(data_dir: Path, shard_names: Iterable[str],
                          expected_sha256: str = CORRECTED_POOL_MANIFEST_SHA256,
                          ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    path = data_dir / "manifest.json"
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"corrected SFT manifest SHA256 mismatch: expected {expected_sha256}, got {actual}")
    manifest = load_sft_manifest(path)
    require_data_quality_exclusions(manifest)
    names = tuple(shard_names)
    records = {name: load_json_records(data_dir / manifest["shards"][name]["path"])
               for name in names}
    return manifest, records


def parsed_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the same parser as AgentRuntime, preserving conversation order."""
    return [{"turn_index": index, "name": call.name, "arguments": call.arguments}
            for index, turn in enumerate(record["conversations"])
            if turn["from"] == "gpt"
            for call in PARSER.parse(turn["value"]).tool_calls]


def protocol_tags(record: dict[str, Any]) -> set[str]:
    calls = parsed_calls(record)
    names = [call["name"] for call in calls]
    tags = set()
    if not calls:
        tags.add("no_tool")
    if len(calls) > 1:
        tags.add("multi_tool")
    if any(name != "image_search" and name in IMAGE_TOOLS for name in names):
        tags.add("non_image_search_visual")
    for call in calls:
        if call["name"] == "image_search":
            tags.add("image_search_img_1" if call["arguments"].get("url") == "img_1"
                     else "image_search_derived")
        elif call["name"] in ("layout_parsing", "crop"):
            tags.add(call["name"])
        elif call["name"] in IMAGE_TOOLS:
            tags.add("other_image_tool")
    return tags


def tool_distribution(records_by_shard: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for shard, records in records_by_shard.items():
        calls = Counter()
        samples = Counter()
        targets = Counter()
        by_source: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: {"calls": Counter(), "first_tool": Counter(), "samples": Counter()})
        no_tool = single = multi = follows = image_search_samples = 0
        for record in records:
            source = record.get("_source", "unknown")
            parsed = parsed_calls(record)
            names = [call["name"] for call in parsed]
            by_source[source]["samples"]["total"] += 1
            by_source[source]["first_tool"][names[0] if names else "no_tool"] += 1
            if not names:
                no_tool += 1
            elif len(names) == 1:
                single += 1
            else:
                multi += 1
            if "image_search" in names:
                image_search_samples += 1
                if any(name == "text_search" for name in names[names.index("image_search") + 1:]):
                    follows += 1
            for name in set(names):
                samples[name] += 1
            for call in parsed:
                name = call["name"]
                calls[name] += 1
                by_source[source]["calls"][name] += 1
                if name == "image_search":
                    targets[str(call["arguments"].get("url"))] += 1
        total = len(records)
        visual_three = sum(calls[name] for name in ("image_search", "layout_parsing", "crop"))
        result[shard] = {
            "total_samples": total, "tool_containing_sample_count": total - no_tool,
            "tool_containing_sample_ratio": (total - no_tool) / total if total else 0,
            "total_tool_calls": sum(calls.values()), "tool_call_counts": dict(sorted(calls.items())),
            "tool_sample_counts": dict(sorted(samples.items())),
            "mean_tool_calls_per_trajectory": sum(calls.values()) / total if total else 0,
            "no_tool_sample_count": no_tool, "single_tool_sample_count": single,
            "multi_tool_sample_count": multi,
            "image_search_followed_by_text_search_count": follows,
            "image_search_followed_by_text_search_ratio":
                follows / image_search_samples if image_search_samples else None,
            "visual_tool_relative_call_ratio": {
                name: calls[name] / visual_three if visual_three else None
                for name in ("image_search", "layout_parsing", "crop")},
            "image_search_target_counts": dict(sorted(targets.items())),
            "by_source": {source: {key: dict(sorted(value.items())) for key, value in counters.items()}
                          for source, counters in sorted(by_source.items())},
        }
    return result


def representative_records(records_by_shard: dict[str, list[dict[str, Any]]],
                           per_shard: int = 15) -> dict[str, list[dict[str, Any]]]:
    """Deterministically cover available protocol tags, then fill in shard order."""
    order = ("image_search_img_1", "image_search_derived", "layout_parsing",
             "crop", "other_image_tool", "no_tool", "multi_tool")
    selected = {}
    for shard, records in records_by_shard.items():
        chosen: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tag in order:
            for record in records:
                if tag in protocol_tags(record) and record["_sample_id"] not in seen:
                    chosen.append(record)
                    seen.add(record["_sample_id"])
                    break
        for record in records:
            if len(chosen) >= per_shard:
                break
            if record["_sample_id"] not in seen:
                chosen.append(record)
                seen.add(record["_sample_id"])
        selected[shard] = chosen[:per_shard]
    return selected


def system_instruction_features(system: str) -> dict[str, bool]:
    """Compare concrete behavior constraints, not mere wording equality."""
    lowered = system.casefold()
    return {
        "explicit_registered_image_ids": "registered input images:" in lowered,
        "tool_first_mindset": "tool-first mindset" in lowered or "proactively using" in lowered,
        "must_use_text_search": "must use `text_search`" in lowered or "must use text_search" in lowered,
        "clear_image_may_answer_without_tool":
            "clear image with a directly answerable question needs no tool call" in lowered,
        "image_search_registered_url": ("image_search" in lowered and
                                        "registered" in lowered and "url" in lowered),
        "no_http_image_id": "never use" in lowered and "http url" in lowered,
        "duplicate_call_guard": "do not repeat an identical tool call" in lowered,
        "verify_dont_guess_policy": "verify, don't guess" in lowered,
        "visual_to_text_retrieval_guidance": ("image_search" in lowered and
                                              "text_search" in lowered and
                                              "after image_search" in lowered),
    }


def prompt_alignment_record(record: dict[str, Any], dataset_path: Path,
                            processor: Any, max_length: int,
                            runtime_declarations: list[dict[str, Any]] | None = None,
                            ) -> dict[str, Any]:
    """Use production builders, processor, and collator; never approximate the mask."""
    messages, images, tools = build_messages(record, dataset_path)
    training_prompt = render_prompt(processor, messages, tools)
    first = record["conversations"][0]["value"]
    initial_count = first.count(IMAGE_MARKER)
    question = normalize_model_question(first.replace(IMAGE_MARKER, " ").strip())
    runtime_messages = AgentRuntime._initial_messages(
        question, images[:initial_count], [f"img_{index}" for index in range(1, initial_count + 1)])
    declarations = runtime_declarations if runtime_declarations is not None else runtime_tools()
    runtime_prompt = processor.apply_chat_template(
        runtime_messages, tools=declarations, tokenize=False, add_generation_prompt=True)
    runtime_inputs = processor.apply_chat_template(
        runtime_messages, tools=declarations, tokenize=True,
        return_dict=True, return_tensors="pt", add_generation_prompt=True)
    batch = OpenSearchVLCollator(processor, dataset_path, max_length)([record])
    ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    supervised = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    spans = []
    for position in supervised:
        if not spans or position != spans[-1][-1] + 1:
            spans.append([position])
        else:
            spans[-1].append(position)
    token_spans = [{"start": span[0], "end_exclusive": span[-1] + 1,
                    "text": processor.tokenizer.decode(ids[span].tolist(),
                                                       skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)}
                   for span in spans]
    training_system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    runtime_system = runtime_messages[0]["content"]
    training_registration = training_system.split("Registered input images:", 1)[-1].split("\n\n", 1)[0]
    runtime_registration = runtime_system.split("Registered input images:", 1)[-1].split("\n\n", 1)[0]
    parsed_first = next((PARSER.parse(turn["value"]) for turn in record["conversations"]
                         if turn["from"] == "gpt" and PARSER.parse(turn["value"]).tool_calls), None)
    runtime_assistant = (AgentRuntime._structured_assistant_message(parsed_first)
                         if parsed_first is not None else None)
    first_assistant_index = next((index for index, message in enumerate(messages)
                                  if message["role"] == "assistant"), len(messages))
    training_prefix = processor.apply_chat_template(
        messages[:first_assistant_index], tools=tools, tokenize=False,
        add_generation_prompt=True)
    role_headers = re.compile(r"<\|im_start\|>([a-z_]+)")
    training_roles = role_headers.findall(training_prefix)
    runtime_roles = role_headers.findall(runtime_prompt)
    train_user = next((message["content"] for message in messages if message["role"] == "user"), None)
    runtime_user = runtime_messages[1]["content"]
    train_user_shape = [part.get("type") for part in train_user] if isinstance(train_user, list) else ["text"]
    runtime_user_shape = [part.get("type") for part in runtime_user]
    observation_messages = [message for message in messages if message["role"] == "tool"]
    train_url_rule = ("registered runtime image ID" in training_system
                      and "HTTP URL" in training_system)
    runtime_url_rule = ("registered img_n in its url argument" in runtime_system
                        and "HTTP URL" in runtime_system)
    training_features = system_instruction_features(training_system)
    runtime_features = system_instruction_features(runtime_system)
    system_severity = ("exact_match" if training_system == runtime_system else
                       "cosmetic_only_drift" if training_features == runtime_features else
                       "semantic_drift")
    differences = [
        {"field": "system_guidance", "severity": system_severity,
         "detail": "SFT source system plus appended rules vs Agent runtime guidance; compare feature map"},
        {"field": "tool_declarations", "severity": "exact_match" if tools == declarations else "semantic_drift",
         "detail": "ordered, full model-facing declarations"},
        {"field": "image_tool_argument_schemas",
         "severity": "exact_match" if [tool["function"]["parameters"] for tool in tools]
         == [tool["function"]["parameters"] for tool in declarations] else "semantic_drift",
         "detail": "ordered parameters, including image_search.url"},
        {"field": "image_search_url_constraint", "severity": "exact_match" if train_url_rule == runtime_url_rule else "semantic_drift",
         "detail": "both effective systems prohibit HTTP image IDs"},
        {"field": "registered_input_images", "severity": "exact_match" if training_registration == runtime_registration else "semantic_drift",
         "detail": "registered ID, width, height lines"},
        {"field": "registration_position", "severity": "semantic_drift" if
         training_system.index("Registered input images:") != runtime_system.index("Registered input images:")
         else "exact_match", "detail": "position inside the respective system prompt"},
        {"field": "initial_user_multimodal_parts",
         "severity": "exact_match" if train_user_shape == runtime_user_shape else "semantic_drift",
         "detail": "ordered image/text content part types"},
        {"field": "chat_template_role_headers", "severity": "exact_match" if training_roles == runtime_roles
         else "semantic_drift", "detail": "processor-rendered pre-first-assistant role headers"},
        {"field": "assistant_tool_call_representation",
         "severity": "semantic_drift" if runtime_assistant is not None else "not_applicable",
         "detail": "SFT assistant uses literal tool_call text; runtime history uses structured tool_calls"},
        {"field": "generation_prompt", "severity": "cosmetic_only_drift",
         "detail": "Eval appends assistant generation header; training renders completed assistant targets"},
        {"field": "tool_observation_payloads",
         "severity": "semantic_drift" if observation_messages else "not_applicable",
         "detail": "SFT uses source observation bodies; runtime uses live ToolResult observations and may append derived image parts"},
    ]
    return {"sample_id": record["_sample_id"], "source": record["_source"],
            "raw_conversation": record["conversations"],
            "effective_messages": messages_to_json_safe(messages),
            "training_prompt": training_prompt, "training_prefix_before_first_assistant": training_prefix,
            "runtime_prompt": runtime_prompt,
            "training_system": training_system, "runtime_system": runtime_system,
            "training_system_features": training_features,
            "runtime_system_features": runtime_features,
            "user_prompt": first, "tool_declarations_training": tools,
            "tool_declarations_runtime": declarations,
            "runtime_initial_messages": messages_to_json_safe(runtime_messages),
            "training_observation_messages": messages_to_json_safe(observation_messages),
            "training_role_headers": training_roles, "runtime_role_headers": runtime_roles,
            "training_user_part_types": train_user_shape, "runtime_user_part_types": runtime_user_shape,
            "runtime_input_ids_shape": list(runtime_inputs["input_ids"].shape),
            "training_input_ids_shape": list(batch["input_ids"].shape),
            "supervised_token_count": len(supervised), "supervised_token_spans": token_spans,
            "assistant_targets": [turn["value"] for turn in record["conversations"]
                                  if turn["from"] == "gpt"],
            "runtime_structured_first_tool_call": runtime_assistant,
            "differences": differences}


def split_supervised_assistant_text(text: str) -> list[tuple[str, str, str | None]]:
    """Classify decoded supervised assistant text without reading masked roles."""
    output = []
    cursor = 0
    for match in TOOL_CALL_BLOCK.finditer(text):
        if text[cursor:match.start()].strip():
            output.append(("assistant_natural_language", text[cursor:match.start()], None))
        block = match.group(0)
        parsed = PARSER.parse(block)
        name = parsed.tool_calls[0].name if len(parsed.tool_calls) == 1 else None
        category = ("image_search_arguments" if name == "image_search"
                    else "other_tool_arguments")
        output.append((category, block, name))
        cursor = match.end()
    if text[cursor:].strip():
        output.append(("assistant_natural_language", text[cursor:], None))
    return output


def url_pattern_rows(sample_id: str, shard: str, source: str,
                     category: str, text: str, tool: str | None = None) -> list[dict[str, Any]]:
    return [{"sample_id": sample_id, "shard": shard, "source": source,
             "category": category, "tool": tool, "pattern": pattern,
             "count": len(matches), "example": text[max(0, matches[0].start()-80):matches[0].end()+100]}
            for pattern, regex in URL_PATTERNS.items()
            if (matches := list(regex.finditer(text)))]


def supervised_url_rows(record: dict[str, Any], shard: str, dataset_path: Path,
                        processor: Any, max_length: int) -> list[dict[str, Any]]:
    """Only labels != -100 are scanned as supervised text."""
    batch = OpenSearchVLCollator(processor, dataset_path, max_length)([record])
    labels = batch["labels"][0]
    ids = batch["input_ids"][0]
    indices = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    contiguous: list[list[int]] = []
    for index in indices:
        if not contiguous or index != contiguous[-1][-1] + 1:
            contiguous.append([index])
        else:
            contiguous[-1].append(index)
    rows = []
    for span in contiguous:
        text = processor.tokenizer.decode(ids[span].tolist(), skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
        for category, section, tool in split_supervised_assistant_text(text):
            rows.extend(url_pattern_rows(record["_sample_id"], shard, record["_source"],
                                         category, section, tool))
            if category == "image_search_arguments":
                rows.append({"sample_id": record["_sample_id"], "shard": shard,
                             "source": record["_source"], "category": category,
                             "tool": tool, "pattern": "image_search_call", "count": 1,
                             "example": section[:180]})
    messages, _, _ = build_messages(record, dataset_path)
    for message in messages:
        role = message["role"]
        if role == "assistant":
            continue
        content = message["content"]
        text = content if isinstance(content, str) else "".join(
            part.get("text", "") for part in content if part.get("type") == "text")
        rows.extend(url_pattern_rows(record["_sample_id"], shard, record["_source"],
                                     {"tool": "masked_tool_observation", "system": "masked_system",
                                      "user": "masked_user"}[role], text))
    return rows


def summarize_url_rows(rows: Iterable[dict[str, Any]], max_examples: int = 12) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (row["shard"], row["source"], row["category"], row["tool"], row["pattern"])
        item = grouped.setdefault(str(key), {"shard": row["shard"], "source": row["source"],
                                            "category": row["category"], "tool": row["tool"],
                                            "pattern": row["pattern"], "count": 0,
                                            "sample_ids": set(), "examples": []})
        item["count"] += row["count"]
        item["sample_ids"].add(row["sample_id"])
        if len(item["examples"]) < max_examples:
            item["examples"].append({"sample_id": row["sample_id"], "text": row["example"]})
    values = []
    for item in grouped.values():
        ids = sorted(item.pop("sample_ids"))
        item["sample_count"] = len(ids)
        values.append(item)
    def aggregate(field: str) -> dict[str, dict[str, Any]]:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        samples: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for row in rows:
            key = str(row.get(field) or "none")
            counts[key][row["pattern"]] += row["count"]
            samples[key][row["pattern"]].add(row["sample_id"])
        return {key: {pattern: {"count": count,
                                "sample_count": len(samples[key][pattern])}
                      for pattern, count in sorted(patterns.items())}
                for key, patterns in sorted(counts.items())}
    supervised = [row for row in rows if not row["category"].startswith("masked_")]
    supervised_by_shard: dict[str, Counter[str]] = defaultdict(Counter)
    for row in supervised:
        supervised_by_shard[row["shard"]][row["pattern"]] += row["count"]
    return {"groups": sorted(values, key=lambda x: (x["shard"], x["source"], x["category"],
                                                    str(x["tool"]), x["pattern"])),
            "by_shard": aggregate("shard"), "by_source": aggregate("source"),
            "by_category": aggregate("category"), "by_tool": aggregate("tool"),
            "supervised_by_shard": {shard: dict(sorted(patterns.items()))
                                    for shard, patterns in sorted(supervised_by_shard.items())}}
