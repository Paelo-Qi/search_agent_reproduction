from __future__ import annotations

import pytest

from opensearch_vl_repro.data import assistant_token_spans, validate_raw_sample


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


def test_assistant_spans_exclude_user_and_observation_tokens() -> None:
    # 70/71 represent an observation between two assistant messages.
    ids = [1, 10, 11, 50, 51, 12, 70, 71, 10, 11, 60, 12, 2]
    spans = assistant_token_spans(ids, [10, 11], 12)
    assert spans == [(3, 6), (10, 12)]
    supervised = {index for start, stop in spans for index in range(start, stop)}
    assert supervised.isdisjoint({0, 1, 2, 6, 7, 8, 9, 12})


def test_assistant_spans_keep_right_truncated_body() -> None:
    assert assistant_token_spans([1, 10, 11, 50, 51], [10, 11], 12) == [(3, 5)]
