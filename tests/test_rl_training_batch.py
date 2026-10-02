import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from opensearch_vl_repro.rl.training_batch import build_dataproto, mask_artifact, response_loss_mask, sampled_logprobs, training_rows
from test_rl_group import fixture_group


def test_real_sampled_token_logprobs_not_top_token():
    mapping = [{8: SimpleNamespace(logprob=-.1), 3: SimpleNamespace(logprob=-1.2)}]
    assert sampled_logprobs([3], mapping) == [-1.2]
    for value in (None, [], [{8: SimpleNamespace(logprob=-.1)}], [{3: SimpleNamespace(logprob=float("nan"))}]):
        with pytest.raises(ValueError): sampled_logprobs([3], value)


def test_fatal_cutoff_keeps_prefix_and_third_error_response_only():
    assert response_loss_mask(4, step_index=0, fatal_step=2) == [1] * 4
    assert response_loss_mask(4, step_index=2, fatal_step=2) == [1] * 4
    assert response_loss_mask(4, step_index=3, fatal_step=2) == [0] * 4
    assert response_loss_mask(4, step_index=10, fatal_step=None) == [1] * 4


def test_training_rows_broadcast_group_advantages_not_reference_or_observation():
    group = fixture_group()
    group["members"][0]["fatal"] = {"fatal": True, "fatal_step": 0}
    group["members"][0]["steps"].append(copy.deepcopy(group["members"][0]["steps"][0]))
    rows, audit = training_rows(group, [.8, -.8])
    assert len(rows) == 2 and [r["advantage"] for r in rows] == [.8, -.8]
    assert audit["masked_post_fatal_tokens"] == 2 and audit["supervised_response_tokens"] == 4
    assert all("reference_answer" not in r and "labels" not in r for r in rows)
    masks = mask_artifact(group, [.8, -.8])["per_step_masks"]
    assert masks[0]["loss_mask"] == [0, 0, 1, 1] and masks[1]["loss_mask"] == [0, 0, 0, 0]
    assert masks[1]["fatal_cutoff"] == 0
    group["members"][0]["fatal"]["fatal_step"] = 100
    with pytest.raises(ValueError, match="outside"): training_rows(group, [.8, -.8])


def test_dataproto_actual_ids_multimodal_positions_padding_and_masks(tmp_path, monkeypatch):
    import torch
    calls = []
    class Proto:
        @staticmethod
        def from_dict(**kwargs): return kwargs
    module = ModuleType("verl"); module.DataProto = Proto
    monkeypatch.setitem(sys.modules, "verl", module)
    class Rope:
        def get_rope_index(self, **kwargs):
            calls.append(kwargs)
            ids = kwargs["input_ids"]
            return torch.arange(ids.shape[-1]).view(1, 1, -1).repeat(3, 1, 1), None
    model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(model=Rope()))
    rows, _ = training_rows(fixture_group(), [.8, -.8])
    rows[1]["prompt_ids"] = [2]; rows[1]["responses"] = [3]
    rows[1]["old_log_probs"] = [-.4]; rows[1]["response_mask"] = [1]
    for index, row in enumerate(rows):
        row["multimodal_file"] = f"mm{index}.pt"
        torch.save({"input_ids": torch.tensor([row["prompt_ids"]]), "pixel_values": torch.ones(2, 8),
                    "image_grid_thw": torch.tensor([[1, 2, 2]])}, tmp_path / row["multimodal_file"])
    data = build_dataproto(rows, directory=tmp_path, model=model, pad_id=0, device="cpu", temperature=.7)
    tensors = data["tensors"]
    assert tensors["input_ids"].tolist() == [[1, 2, 3, 4], [0, 2, 3, 0]]
    assert tensors["responses"].tolist() == [[3, 4], [3, 0]]
    assert tensors["response_mask"].tolist() == [[1, 1], [1, 0]]
    assert tensors["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 1, 1, 0]]
    assert tensors["position_ids"].shape == (2, 3, 4) and len(calls) == 2
    assert len(data["non_tensors"]["multi_modal_inputs"]) == 2
    assert data["meta_info"]["temperature"] == .7 and "labels" not in tensors
    assert tensors["old_log_probs"][1].tolist() == pytest.approx([-.4, 0])
    assert tensors["advantages"][0].tolist() == pytest.approx([.8, .8])
    # Context is attention-visible, but loss uses only the separate response suffix mask.
    assert tensors["response_mask"].shape[-1] == 2 < tensors["input_ids"].shape[-1]
    rows[0]["prompt_ids"] = [99]
    with pytest.raises(ValueError, match="differ"): build_dataproto(rows, directory=tmp_path, model=model, pad_id=0, device="cpu", temperature=.7)
