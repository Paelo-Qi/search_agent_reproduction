from __future__ import annotations

import pytest
import torch

from opensearch_vl_repro.data import assistant_token_mask, validate_raw_sample


class FakeTokenizer:
    unk_token_id = -1

    def encode(self, value: str, add_special_tokens: bool = False) -> list[int]:
        assert value == "<|im_start|>assistant\n"
        assert add_special_tokens is False
        return [10, 11]

    def convert_tokens_to_ids(self, value: str) -> int:
        assert value == "<|im_end|>"
        return 12


def valid_sample() -> dict:
    return {
        "conversations": [
            {"from": "human", "value": "<image>Question"},
            {"from": "gpt", "value": "<think>Reason</think><tool_call>{}</tool_call>"},
            {"from": "observation", "value": "retrieved text"},
            {"from": "gpt", "value": "final answer"},
        ],
        "images": ["image.jpg"],
        "system": "system",
        "tools": "[]",
    }


def test_validate_sharegpt_agent_trajectory() -> None:
    validate_raw_sample(valid_sample())


def test_validate_rejects_image_mismatch() -> None:
    sample = valid_sample()
    sample["images"] = []
    with pytest.raises(ValueError, match="no image"):
        validate_raw_sample(sample)


def test_assistant_mask_selects_only_assistant_bodies_and_end_tokens() -> None:
    ids = torch.tensor([[1, 10, 11, 50, 51, 12, 2, 10, 11, 60, 12, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    mask = assistant_token_mask(ids, FakeTokenizer(), attention)
    expected = torch.tensor(
        [[False, False, False, True, True, True, False, False, False, True, True, False]]
    )
    assert torch.equal(mask, expected)


def test_assistant_mask_keeps_right_truncated_body() -> None:
    ids = torch.tensor([1, 10, 11, 50, 51])
    mask = assistant_token_mask(ids, FakeTokenizer())
    assert torch.equal(mask, torch.tensor([False, False, False, True, True]))

