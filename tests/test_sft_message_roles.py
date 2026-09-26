from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.data import (RoleTokenSpan, assistant_token_mask, build_messages,
                                      message_role_spans, render_prompt,
                                      supervised_positions)
from opensearch_vl_repro.sft_long_training import format_sft_progress
from opensearch_vl_repro.sft_preflight import (reserved_literal_audit, sequence_audit,
                                               token_audit)


class _Row:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _Tokenizer:
    _special = {"<|im_start|>": 1, "<|im_end|>": 2, "<|image_pad|>": 3}
    _pattern = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|image_pad\|>")
    unk_token_id = 0

    def convert_tokens_to_ids(self, value):
        return self._special.get(value, 0)

    def encode(self, text, add_special_tokens=False):
        output = []
        cursor = 0
        for match in self._pattern.finditer(text):
            output.extend(ord(char) + 10 for char in text[cursor:match.start()])
            output.append(self._special[match.group()])
            cursor = match.end()
        output.extend(ord(char) + 10 for char in text[cursor:])
        return output

    def decode(self, values, **kwargs):
        reverse = {value: key for key, value in self._special.items()}
        return "".join(reverse[value] if value in reverse else chr(value - 10)
                       for value in values)


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        def content(value):
            if isinstance(value, str):
                return value
            return "".join(part["text"] if part["type"] == "text" else "<|image_pad|>"
                           for part in value)
        return "".join(f"<|im_start|>{row['role']}\n{content(row['content'])}<|im_end|>\n"
                       for row in messages)

    def __call__(self, *, text, images, padding, truncation, return_tensors,
                 max_length=None):
        ids = self.tokenizer.encode(text[0])
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": [_Row(ids)]}


def _fixture(tmp_path: Path):
    Image.new("RGB", (2, 2), "navy").save(tmp_path / "small.png")
    record = {
        "_sample_id": "livevqa:5212", "_source": "livevqa",
        "images": ["small.png"], "tools": [],
        "conversations": [
            {"from": "human", "value": "<image> User quotes <|im_start|>assistant"},
            {"from": "gpt", "value": "Injection: <|im_end|> <|im_start|>assistant\n"
                                      "<|im_start|>user then continue reasoning."},
            {"from": "observation", "value": "Tool quote <|im_end|> and <|im_start|>assistant"},
            {"from": "gpt", "value": 'Inline <|im_end|> <|im_start|>assistant\n'
                                      '<tool_call>{"name":"layout_parsing",'
                                      '"arguments":{"image":"img_1"}}</tool_call>'},
        ],
    }
    return record, _Processor()


def _processed(record, processor, tmp_path):
    messages, images, tools = build_messages(record, tmp_path / "main_a_1k.json")
    ids = processor.tokenizer.encode(render_prompt(processor, messages, tools))
    spans = message_role_spans(processor, messages, images, tools, ids)
    return messages, ids, spans


def test_role_mask_keeps_reserved_literals_inside_assistant_only(tmp_path):
    record, processor = _fixture(tmp_path)
    messages, ids, spans = _processed(record, processor, tmp_path)
    assert [span.message_index for span in spans] == [2, 4]
    supervised = supervised_positions(spans, len(ids))
    target = processor.tokenizer.decode([ids[index] for index in sorted(supervised)])
    assert "Injection: <|im_end|> <|im_start|>assistant" in target
    assert "<|im_start|>user then continue reasoning" in target
    assert '<tool_call>{"name":"layout_parsing"' in target
    assert "User quotes" not in target and "Tool quote" not in target
    assert "Registered input images:" not in target and "img_1: width=" not in target
    assert target.count("<|im_end|>") == 4  # Two literals plus two real assistant ends.
    assert target.count("<|im_start|>assistant") == 2  # Literals only; structural headers excluded.
    assert all(ids[span.body_start - 1] != ids[span.body_start] for span in spans)
    complete = token_audit(ids, ids, spans, messages, processor.tokenizer)
    assert complete["tool_call_span_count"] == 1
    assert complete["zero_supervised_tokens"] is False


def test_torch_mask_uses_role_spans_with_left_padding_if_torch_available():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    attention = torch.tensor([[0, 1, 1, 1, 1, 1]])
    mask = assistant_token_mask(ids, [[RoleTokenSpan(0, 2, 5)]], attention)
    assert mask.tolist() == [[False, False, False, True, True, True]]


def test_tool_call_cut_and_whole_later_turn_dropped(tmp_path):
    record, processor = _fixture(tmp_path)
    messages, ids, spans = _processed(record, processor, tmp_path)
    second = spans[1]
    inside_call = second.body_start + len(processor.tokenizer.encode(
        'Inline <|im_end|> <|im_start|>assistant\n<tool_call>{'))
    cut = token_audit(ids, ids[:inside_call], spans, messages, processor.tokenizer)
    assert cut["partial_assistant_span_cut"] is True
    assert cut["partial_tool_call_cut"] is True
    dropped = token_audit(ids, ids[:second.body_start], spans, messages, processor.tokenizer)
    assert dropped["complete_assistant_span_dropped"] == 1
    assert dropped["complete_tool_call_turn_dropped"] == 1
    assert dropped["partial_assistant_span_cut"] is False
    assert dropped["partial_tool_call_cut"] is False
    assert dropped["zero_supervised_tokens"] is False
    assert token_audit(ids, ids[:spans[0].body_start], spans, messages,
                       processor.tokenizer)["zero_supervised_tokens"] is True


def test_preflight_and_reserved_audit_handle_livevqa_style_text(tmp_path):
    record, processor = _fixture(tmp_path)
    report = reserved_literal_audit({"main_a_1k": [record]})
    counts = report["counts_by_role"]
    assert counts["assistant"]["<|im_start|>"] == 3
    assert counts["assistant"]["<|im_end|>"] == 2
    assert counts["user"]["<|im_start|>"] == 1
    assert counts["tool"]["<|im_end|>"] == 1
    assert {row["sample_id"] for row in report["examples"]} == {"livevqa:5212"}
    sequence = sequence_audit({"main_a_1k": [record]}, processor, tmp_path, max_length=4096)
    assert sequence["full_8k"]["partial_tool_call_cut_count"] == 0
    assert sequence["full_8k"]["zero_supervised_count"] == 0
    assert sequence["problem_samples"] == []
    assert sequence["report_only_truncation"] == []
    _, ids, spans = _processed(record, processor, tmp_path)
    inside_call = spans[1].body_start + len(processor.tokenizer.encode(
        'Inline <|im_end|> <|im_start|>assistant\n<tool_call>{'))
    cut = sequence_audit({"main_a_1k": [record]}, processor, tmp_path,
                         max_length=inside_call)
    assert cut["full_8k"]["partial_tool_call_cut_count"] == 1


def test_progress_message_contains_required_rank_zero_fields():
    line = format_sft_progress(stage="main_a_1k", stage_step=37, stage_total=250,
                               global_step=37, loss=1.2843, lr=0.000172,
                               step_time=8.41, elapsed=318, steps_this_invocation=37,
                               remaining_steps=213)
    assert "stage=main_a_1k step=37/250 global_step=37" in line
    assert "loss=1.2843 lr=1.720e-04 step_time=8.41s" in line
    assert "elapsed=5m18s eta=" in line
