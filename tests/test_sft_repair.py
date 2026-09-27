from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.data import build_messages
from opensearch_vl_repro.sft_protocol_diagnostics import CORRECTED_POOL_MANIFEST_SHA256
from opensearch_vl_repro.sft_repair_data import (REPAIR_VERSION, SELECTION_VERSION,
                                                filter_repair_candidates,
                                                repair_category, select_repair_records,
                                                validate_repair_manifest)
from opensearch_vl_repro.sft_repair_mask import (MASK_VERSIONS, target_character_spans,
                                                RepairCollator, target_token_spans)
from opensearch_vl_repro.sft_repair_training import (REPAIR_KIND, load_repair_config,
                                                    repair_checkpoint_metadata,
                                                    validate_repair_checkpoint,
                                                    validate_parent_metadata)
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_long_training import PROJECT_ROOT


class _Row:
    def __init__(self, ids):
        self.ids = ids

    def tolist(self):
        return list(self.ids)


class _Tokenizer:
    specials = {"<|im_start|>": 1, "<|im_end|>": 2, "<|image_pad|>": 3}
    pattern = re.compile("|".join(re.escape(value) for value in specials))
    unk_token_id = 0

    def convert_tokens_to_ids(self, value):
        return self.specials.get(value, 0)

    def encode(self, text):
        ids = []
        offset = 0
        for match in self.pattern.finditer(text):
            ids.extend(ord(char) + 10 for char in text[offset:match.start()])
            ids.append(self.specials[match.group()])
            offset = match.end()
        ids.extend(ord(char) + 10 for char in text[offset:])
        return ids

    def decode(self, values, **kwargs):
        reverse = {value: key for key, value in self.specials.items()}
        return "".join(reverse[value] if value in reverse else chr(value - 10)
                       for value in values)


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.render_calls = 0
        self.token_calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.render_calls += 1
        def content(value):
            if isinstance(value, str):
                return value
            return "".join(part["text"] if part["type"] == "text" else "<|image_pad|>"
                           for part in value)
        return "".join(f"<|im_start|>{row['role']}\n{content(row['content'])}<|im_end|>\n"
                       for row in messages)

    def __call__(self, *, text, images, padding, truncation, return_tensors,
                 max_length=None):
        self.token_calls += 1
        ids = self.tokenizer.encode(text[0])
        return {"input_ids": [_Row(ids[:max_length] if truncation else ids)]}


def _record(tmp_path: Path, mode="argument_only", image_id="img_1", multi=False):
    Image.new("RGB", (3, 4), "red").save(tmp_path / "one.png")
    Image.new("RGB", (3, 4), "blue").save(tmp_path / "two.png")
    Image.new("RGB", (3, 4), "green").save(tmp_path / "three.png")
    call = lambda image: ("<tool_call>\n"
                          f'{{"name": "image_search", "arguments": {{"url": "{image}"}}}}'
                          "\n</tool_call>")
    assistant = "Thinking before.\n" + call(image_id)
    if multi:
        assistant += "\nUnrelated explanation.\n" + call("img_2")
    return {"_sample_id": "fvqa:9", "_source": "fvqa",
            "_repair_category": "img_1", "_repair_target_turn_index": 1,
            "_repair_target_tool": "image_search",
            "images": ["one.png", "two.png", "three.png"], "tools": [],
            "conversations": [{"from": "human", "value": "<image><image><image>Question"},
                              {"from": "gpt", "value": assistant},
                              {"from": "observation", "value": "Do not supervise this"},
                              {"from": "gpt", "value": "Natural-language answer."}]}


@pytest.mark.parametrize("image_id", ["img_1", "img_2", "img_3"])
def test_r1_only_argument_fragment_with_real_processor_prefix_checks(tmp_path, image_id):
    record = _record(tmp_path, image_id=image_id)
    processor = _Processor()
    full, spans = target_token_spans(processor, record, tmp_path / "repair.json",
                                     "argument_only")
    assert len(spans) == 1
    assert spans[0]["decoded"] == f'"arguments": {{"url": "{image_id}"}}'
    assert processor.render_calls >= 4 and processor.token_calls >= 4
    supervised = processor.tokenizer.decode(full[spans[0]["start"]:spans[0]["end_exclusive"]])
    assert "image_search" not in supervised and "Thinking" not in supervised
    assert "Do not supervise" not in supervised and "Question" not in supervised
    assert "Natural-language answer" not in supervised


def test_r1_supervises_all_valid_image_search_arguments_not_explanations(tmp_path):
    record = _record(tmp_path, multi=True)
    _, spans = target_token_spans(_Processor(), record, tmp_path / "repair.json",
                                  "argument_only")
    assert len(spans) == 2
    assert [span["decoded"] for span in spans] == [
        '"arguments": {"url": "img_1"}', '"arguments": {"url": "img_2"}']


def test_r2_only_selected_complete_tool_call(tmp_path):
    record = _record(tmp_path, multi=True)
    _, spans = target_token_spans(_Processor(), record, tmp_path / "repair.json",
                                  "full_tool_call")
    assert len(spans) == 1
    assert spans[0]["decoded"].startswith("<tool_call>")
    assert spans[0]["decoded"].endswith("</tool_call>")
    assert "Thinking" not in spans[0]["decoded"]
    assert "Unrelated explanation" not in spans[0]["decoded"]


def test_r2_no_tool_only_response_block(tmp_path):
    record = _record(tmp_path)
    record["conversations"] = [record["conversations"][0],
                               {"from": "gpt", "value": "<think>Private text</think>"
                                "<response>Final answer</response>"}]
    record["images"] = ["one.png", "two.png", "three.png"]
    record["_repair_target_tool"] = None
    assert repair_category(record, "full_tool_call")[0] == "no_tool"
    _, spans = target_token_spans(_Processor(), record, tmp_path / "repair.json",
                                  "full_tool_call")
    assert [span["decoded"] for span in spans] == ["<response>Final answer</response>"]


def test_alignment_failure_fails_closed(tmp_path):
    class _MergingProcessor(_Processor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            # Simulate tokenizer/context sensitivity at the target boundary.
            if "Thinking before." in kwargs["text"][0] and '"arguments"' not in kwargs["text"][0]:
                result["input_ids"][0].ids[-3] += 1
            return result
    record = _record(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        target_token_spans(_MergingProcessor(), record, tmp_path / "repair.json",
                           "argument_only")


def test_data_selection_deterministic_and_image_overlap_skipped():
    candidates = []
    for number in range(8):
        raw = {"conversations": [{"from": "human", "value": "<image>Q"},
                                  {"from": "gpt", "value": "<tool_call>"
                                   f'{{"name":"image_search","arguments":{{"url":"img_1"}}}}'
                                   "</tool_call>"}], "images": ["one.png"]}
        candidates.append({"sample_id": f"fvqa:{number}", "source": "fvqa",
                           "source_index": number, "record": raw})
    hashes = lambda item: ["forbidden" if item["sample_id"] == "fvqa:0" else item["sample_id"]]
    kwargs = {"mode": "argument_only", "image_hashes": hashes,
              "forbidden_hashes": {"forbidden"}, "seed": 7, "targets": {"img_1": 4}}
    first, report = select_repair_records(candidates, **kwargs)
    second, _ = select_repair_records(list(reversed(candidates)), **kwargs)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert "fvqa:0" not in [row["sample_id"] for row in first]
    assert len(first) == 4


def test_bad_image_contract_sample_can_be_filtered_before_selection():
    from opensearch_vl_repro.sft_image_grounding import audit_raw_image_contract
    raw = {"conversations": [{"from": "human", "value": "<image>Q"},
                              {"from": "gpt", "value": '<tool_call>{"name":"image_search",'
                               '"arguments":{"url":"img_2"}}</tool_call>'}],
           "images": ["one.png"]}
    assert not audit_raw_image_contract(raw)["passed"]


def test_source_scan_excludes_formal_pool_frozen_eval_question_and_bad_contract(
        tmp_path, monkeypatch):
    from opensearch_vl_repro import tool_protocol_dev
    from opensearch_vl_repro.sft_tool_audit import sha256_file

    def raw(question, url="img_1"):
        return {"conversations": [
            {"from": "human", "value": f"<image>{question}"},
            {"from": "gpt", "value": '<tool_call>{"name":"image_search",'
             f'"arguments":{{"url":"{url}"}}}}</tool_call>'}],
            "images": ["one.png"], "tools": []}

    source = tmp_path / "source.json"
    source.write_text(json.dumps([raw("pool"), raw("frozen"), raw("eval"),
                                  raw("bad", "img_2"), raw("clean")]), encoding="utf-8")
    monkeypatch.setattr(tool_protocol_dev, "SOURCE_FILES", {"fvqa": "source.json"})
    pool = {"membership": [{"sample_id": "fvqa:0"}],
            "source_files": {"fvqa": {"sha256": sha256_file(source)}}}
    candidates, counts = tool_protocol_dev.candidate_metadata(
        tmp_path, pool, {"eval"}, {"fvqa:1": "frozen"})
    assert [item["sample_id"] for item in candidates] == ["fvqa:4"]
    assert counts["image_contract_bad"] == 1
    assert counts["eval_question_overlap"] == 1


def test_r1_category_uses_real_non_img1_targets(tmp_path):
    record = _record(tmp_path, image_id="img_2")
    assert repair_category(record, "argument_only")[0] == "img_2"
    record["conversations"][1]["value"] = record["conversations"][1]["value"].replace(
        "img_2", "img_3")
    assert repair_category(record, "argument_only")[0] == "img_3_or_later"


def test_candidate_filter_excludes_dev_ids_and_question_overlaps():
    def row(identity, question):
        return {"sample_id": identity, "record": {"conversations": [
            {"from": "human", "value": f"<image>{question}"}]}}
    candidates = [row("fvqa:1", "Eval Q"), row("fvqa:2", "Dev Q"),
                  row("fvqa:3", "Other"), row("fvqa:4", "Clean")]
    filtered = filter_repair_candidates(candidates, excluded_ids={"fvqa:2"},
                                        forbidden_questions={"eval q", "dev q", "other"})
    assert [item["sample_id"] for item in filtered] == ["fvqa:4"]


def test_manifest_checksum_and_isolation_fail_closed(tmp_path):
    rows = [{"sample_id": "fvqa:1"}]
    records = [{"_sample_id": "fvqa:1"}]
    payload = json.dumps(records).encode()
    data = tmp_path / "repair.json"
    data.write_bytes(payload)
    manifest = {"repair_version": REPAIR_VERSION, "repair_mode": "argument_only",
                "selection_algorithm": SELECTION_VERSION,
                "dataset_id": "OpenSearch-VL/Search-VL-SFT-36K",
                "dataset_revision": "2c1c460af4fa15bd63210cbf426a96664b959944",
                "corrected_pool_manifest_sha256": CORRECTED_POOL_MANIFEST_SHA256,
                "repair_json_sha256": hashlib.sha256(payload).hexdigest(),
                "membership": rows, "membership_sha256": canonical_json_sha256(rows),
                "sample_count": 1}
    for key in ("eval300_id_overlap_count", "eval300_question_overlap_count",
                "eval300_image_overlap_count", "dev30_overlap_count",
                "dev50_id_overlap_count", "dev50_question_overlap_count",
                "dev50_image_overlap_count", "sft8k_overlap_count",
                "frozen_exclusion_overlap_count", "image_contract_bad_count"):
        manifest[key] = 0
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert validate_repair_manifest(data, path)["sample_count"] == 1
    data.write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        validate_repair_manifest(data, path)
    data.write_bytes(payload)
    manifest["dev50_image_overlap_count"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="isolation"):
        validate_repair_manifest(data, path)


def test_parent_metadata_and_repair_checkpoint_not_formal(tmp_path):
    parent = PROJECT_ROOT / "outputs/sft_main/checkpoint-3k"
    model = {"name_or_path": "Qwen/Qwen3-VL-4B-Instruct", "revision": "rev"}
    metadata = {"stage": "main_b_2k", "stage_complete": True,
                "checkpoint_complete": True, "lineage": ["main_a_1k", "main_b_2k"],
                "model": model["name_or_path"], "model_revision": "rev",
                "pool_manifest_sha256": CORRECTED_POOL_MANIFEST_SHA256,
                "sft_input_message_version": "runtime-image-id-grounding-v2"}
    validate_parent_metadata(metadata, model=model, parent_path=parent)
    with pytest.raises(ValueError):
        validate_parent_metadata({**metadata, "stage": "extra_1k"},
                                 model=model, parent_path=parent)
    repair = load_repair_config(PROJECT_ROOT / "configs/sft_repair_r1.yaml")
    checkpoint = repair_checkpoint_metadata(
        repair=repair, base={"model": model, "lora": {}}, manifest_sha="manifest",
        dataset_sha="data", parent_sha="parent", step=25)
    assert checkpoint["checkpoint_kind"] == REPAIR_KIND
    assert checkpoint["formal_sft_stage"] is False
    assert "stage" not in checkpoint and "global_step" not in checkpoint
    assert checkpoint["parent_checkpoint_metadata_sha256"] == "parent"
    assert checkpoint["repair_mask_version"] == MASK_VERSIONS["argument_only"]
    assert load_repair_config(PROJECT_ROOT / "configs/sft_repair_r2.yaml",
                              max_steps=100)["max_steps"] == 100


def test_repair_checkpoint_checksums_and_kind(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"test weights")
    metadata = {"checkpoint_kind": REPAIR_KIND, "formal_sft_stage": False,
                "checkpoint_complete": True, "repair_mode": "argument_only",
                "repair_mask_version": MASK_VERSIONS["argument_only"],
                "file_sha256": {"adapter/adapter_model.safetensors":
                                hashlib.sha256(b"test weights").hexdigest()}}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    assert validate_repair_checkpoint(tmp_path)["repair_mode"] == "argument_only"
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_repair_checkpoint(tmp_path)
    weights.write_bytes(b"test weights")
    metadata["stage"] = "main_b_2k"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="isolated"):
        validate_repair_checkpoint(tmp_path)


def test_original_expert_target_is_unchanged_by_repair_mask(tmp_path):
    record = _record(tmp_path)
    before = json.dumps(record["conversations"], ensure_ascii=False)
    build_messages(record, tmp_path / "repair.json")
    target_token_spans(_Processor(), record, tmp_path / "repair.json", "argument_only")
    assert json.dumps(record["conversations"], ensure_ascii=False) == before


def test_cached_qwen_processor_chat_template_alignment_if_available(tmp_path):
    pytest.importorskip("transformers")
    from opensearch_vl_repro.model import load_processor
    from opensearch_vl_repro.sft_train_plan import load_main_config
    base = load_main_config(PROJECT_ROOT / "configs/sft_main.yaml",
                            base_eval_config=PROJECT_ROOT / "configs/eval_base_300.yaml")
    try:
        processor = load_processor(base, local_files_only=True)
    except (OSError, ValueError) as exc:
        pytest.skip(f"pinned Qwen processor not cached locally: {exc}")
    record = _record(tmp_path)
    _, r1 = target_token_spans(processor, record, tmp_path / "repair.json",
                               "argument_only")
    _, r2 = target_token_spans(processor, record, tmp_path / "repair.json",
                               "full_tool_call")
    assert r1[0]["decoded"] == '"arguments": {"url": "img_1"}'
    assert r2[0]["decoded"].startswith("<tool_call>")


def test_torch_repair_labels_only_exact_targets_if_available(tmp_path):
    torch = pytest.importorskip("torch")

    class _TorchProcessor(_Processor):
        def __call__(self, **kwargs):
            ids = super().__call__(**kwargs)["input_ids"][0].tolist()
            if kwargs["truncation"]:
                ids = ids[:kwargs["max_length"]]
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

    processor = _TorchProcessor()
    record = _record(tmp_path, multi=True)
    data_path = tmp_path / "repair.json"
    for mode, expected_count in (("argument_only", 2), ("full_tool_call", 1)):
        batch = RepairCollator(processor, data_path, 10000, mode)([record])
        positions = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0].tolist()
        _, spans = target_token_spans(processor, record, data_path, mode)
        assert len(spans) == expected_count
        assert positions == [index for span in spans for index in
                             range(span["start"], span["end_exclusive"])]
