"""Real per-generation rLLM tokens -> verl DataProto; no SFT labels/CE.

Each generation is one row with its own actual context (including observations).
All ranks use the same complete group in this small gate to keep FSDP collective
counts identical despite different trajectory lengths. No dummy rank padding.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from opensearch_vl_repro.rl.group import validate_group

_FORMAL_BATCH_SEAL = object()


def formal_training_rows(window, run, policy, groups, reward_window):
    """One logical row per actual generation; never recompute a row-level baseline."""
    from .checkpoint import check_seal, FORMAL_WEIGHTING
    from .training_window import validate_training_window
    from .reward import clamp_fatal_advantages
    from opensearch_vl_repro.eval_subset import canonical_json_sha256
    validate_training_window(window, run, policy, groups)
    check_seal(reward_window, "reward_window_sha256")
    if (reward_window["window_sha256"] != window["window_sha256"]
            or reward_window.get("estimator") != "official_verl_rloo"):
        raise ValueError("foreign RLOO window")
    expected = [(g["identity"]["trajectory_group_id"], m) for g in groups
                for m in sorted(g["members"], key=lambda m: m["rollout_index"])]
    if len(reward_window["rows"]) != len(expected):
        raise ValueError("RLOO trajectory membership mismatch")
    rows, masks = [], []
    for outcome, (gid, member) in zip(reward_window["rows"], expected, strict=True):
        rewards = [m["reward"]["total"] for g, m in expected if g == gid]
        reward = member["reward"]["total"]
        raw = reward - (math.fsum(rewards) - reward) / (len(rewards) - 1)
        final = clamp_fatal_advantages([raw], [member["fatal"]])[0]
        if (outcome["group_id"] != gid or outcome["member_id"] != member["member_id"]
                or outcome["rollout_index"] != member["rollout_index"]
                or outcome["fatal"] is not member["fatal"] or outcome["reward"] != reward
                or any(type(outcome[k]) not in (int, float) or not math.isfinite(outcome[k])
                       for k in ("raw_advantage", "final_advantage"))
                or not math.isclose(outcome["raw_advantage"], raw, abs_tol=1e-12)
                or not math.isclose(outcome["final_advantage"], final, abs_tol=1e-12)):
            raise ValueError("RLOO outcome/trajectory mismatch")
        cutoff = member.get("fatal_step") if member["fatal"] else None
        if member["fatal"] and (type(cutoff) is not int or not 0 <= cutoff < len(member["steps"])):
            raise ValueError("fatal trajectory requires an explicit actual generation cutoff")
        for index, step in enumerate(member["steps"]):
            identity = dict(group_id=gid, prompt_id=member["identity"]["prompt_id"],
                            rollout_index=member["rollout_index"], step_index=index,
                            member_id=member["member_id"])
            mask = response_loss_mask(len(step["response_ids"]), step_index=index, fatal_step=cutoff)
            masks.append({**identity, "fatal_step": cutoff,
                          "loss_mask": [0] * len(step["prompt_ids"]) + mask})
            # Entirely post-fatal generations are forensic masks, NOT dummy PPO rows.
            if not any(mask):
                continue
            rows.append({**identity, "logical_row_id": canonical_json_sha256(identity),
                         "prompt_ids": list(step["prompt_ids"]), "responses": list(step["response_ids"]),
                         "response_mask": mask, "rollout_log_probs": list(step["logprobs"]),
                         "advantage": float(final), "multimodal_file": step["multimodal_file"],
                         "multimodal_sha256": next(g["file_sha256"][step["multimodal_file"]]
                                                   for g in groups if g["identity"]["trajectory_group_id"] == gid),
                         "fatal_step": cutoff})
    if not rows or len({r["logical_row_id"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique logical generation rows required")
    return rows, dict(weighting=FORMAL_WEIGHTING, per_step_masks=masks,
                      logical_row_count=len(rows), supervised_response_tokens=sum(sum(r["response_mask"]) for r in rows))


def rank_local_rows(rows, plan, rank):
    from .training_window import deterministic_rank_plan
    if type(rank) is not int or not 0 <= rank < plan["world_size"]:
        raise ValueError("invalid physical rank")
    ids = [row["logical_row_id"] for row in rows]
    if plan != deterministic_rank_plan(ids, plan["world_size"]):
        raise ValueError("non-deterministic/unequal rank plan")
    by_id = dict(zip(ids, rows, strict=True))
    physical_ids = ids * plan["replication_factor"]
    mapping = [{"physical_index": i, "logical_row_id": rid, "rank": i % plan["world_size"]}
               for i, rid in enumerate(physical_ids)]
    return [by_id[rid] for rid in plan["rank_assignments"][rank]], dict(
        logical_row_count=len(ids), physical_row_count=len(mapping), replication_factor=plan["replication_factor"],
        rank_assignment=plan["rank_assignments"], logical_to_physical=mapping,
        local_row_count=plan["local_row_count"], world_size=plan["world_size"], rank=rank)


@dataclass(frozen=True)
class FormalBatchReceipt:
    artifact: dict
    data_id: int
    seal: object


def require_formal_batch(data, receipt, window):
    from .checkpoint import check_seal
    from .old_logprob import input_fingerprint, fingerprint
    if not isinstance(receipt, FormalBatchReceipt) or receipt.seal is not _FORMAL_BATCH_SEAL or receipt.data_id != id(data):
        raise ValueError("actual formal batch materialization receipt required")
    check_seal(receipt.artifact, "batch_receipt_sha256")
    if (receipt.artifact["window_sha256"] != window["window_sha256"]
            or receipt.artifact["input_sha256"] != input_fingerprint(data)
            or receipt.artifact["rollout_log_probs_sha256"] != fingerprint(data.batch["rollout_log_probs"])):
        raise ValueError("formal window/batch receipt mismatch")
    assert_cpu_multimodal(data.non_tensor_batch)
    if any(t.device.type != "cpu" for t in data.batch.values()):
        raise ValueError("formal DataProto must be CPU-backed outside active microbatches")
    return receipt.artifact


def build_rank_local_dataproto(window, run, policy, groups, reward_window, *, group_directories,
                              rank, model, pad_id, temperature):
    """CPU-only carrier using saved processor IDs/attention + real Qwen M-RoPE.

    No processor call/tokenization occurs here. All ranks pad to WINDOW maxima,
    allowing differing response token counts without differing collective calls.
    """
    import numpy as np
    import torch
    from verl import DataProto
    from .checkpoint import seal
    from .group import read_formal_group
    from .old_logprob import input_fingerprint, fingerprint
    from .training_window import deterministic_rank_plan
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    rows, masks = formal_training_rows(window, run, policy, groups, reward_window)
    if temperature != run["semantics"]["rollout"]["config"]["temperature"] or temperature != .7:
        raise ValueError("formal rollout/actor temperature mismatch")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("invalid pad token")
    for group in groups:
        if read_formal_group(group_directories[group["identity"]["trajectory_group_id"]]) != group:
            raise ValueError("actual committed group differs from window")
    plan = deterministic_rank_plan([r["logical_row_id"] for r in rows], run["semantics"]["world_size"])
    local, mapping = rank_local_rows(rows, plan, rank)
    p, r = max(len(row["prompt_ids"]) for row in rows), max(len(row["responses"]) for row in rows)
    tensors, mm = [], []
    rope = model.get_base_model().model
    for row in local:
        directory = Path(group_directories[row["group_id"]]).resolve()
        path = (directory / row["multimodal_file"]).resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != row["multimodal_sha256"]:
            raise ValueError("changed/unsafe saved processor input")
        original = torch.load(path, map_location="cpu", weights_only=True)
        if original["input_ids"].shape != (1, len(row["prompt_ids"])) or original["input_ids"][0].tolist() != row["prompt_ids"]:
            raise ValueError("actual generation/processor prompt IDs mismatch")
        attention = original["attention_mask"]
        if (attention.shape != original["input_ids"].shape or not bool(((attention == 0) | (attention == 1)).all())
                or attention[0, -1].item() != 1):
            raise ValueError("invalid saved processor attention")
        vision = {k: v for k, v in original.items() if k not in {
            "input_ids", "attention_mask", "position_ids", "labels", "rope_deltas", "mm_token_type_ids"}}
        if not {"pixel_values", "image_grid_thw"} <= vision.keys() or not vision["pixel_values"].numel():
            raise ValueError("real multimodal processor tensors required")
        assert_cpu_multimodal(vision)
        ids = torch.tensor([row["prompt_ids"] + row["responses"]], dtype=torch.long)
        full_attention = torch.cat((attention.long(), torch.ones(1, len(row["responses"]), dtype=torch.long)), -1)
        import inspect
        rope_kwargs, token_types = {}, original.get("mm_token_type_ids")
        if token_types is not None:
            if token_types.shape != original["input_ids"].shape:
                raise ValueError("saved processor modality token types misaligned")
            token_types = torch.cat((token_types, torch.zeros(1, len(row["responses"]), dtype=token_types.dtype)), -1)
            rope_kwargs["mm_token_type_ids"] = token_types
        elif "mm_token_type_ids" in inspect.signature(rope.get_rope_index).parameters:
            raise ValueError("this Qwen version requires saved processor mm_token_type_ids")
        positions, _ = rope.get_rope_index(input_ids=ids, image_grid_thw=vision["image_grid_thw"],
            attention_mask=full_attention, **rope_kwargs,
            **{k: vision[k] for k in ("video_grid_thw", "second_per_grid_ts") if k in vision})
        if positions.device.type != "cpu" or positions.shape != (3, 1, ids.shape[-1]):
            raise ValueError("CPU real Qwen M-RoPE positions required")
        if "position_ids" in original:
            saved = original["position_ids"]
            if not torch.equal(saved.reshape(3, 1, -1), positions[:, :, :len(row["prompt_ids"])]):
                raise ValueError("saved processor M-RoPE changed")
        left, right = p - len(row["prompt_ids"]), r - len(row["responses"])
        if token_types is not None:
            vision["mm_token_type_ids"] = torch.nn.functional.pad(token_types, (left, right))
        tensors.append(dict(input_ids=torch.nn.functional.pad(ids[0], (left, right), value=pad_id),
            attention_mask=torch.nn.functional.pad(full_attention[0], (left, right)),
            position_ids=torch.nn.functional.pad(positions[:, 0], (left, right)),
            responses=torch.tensor(row["responses"] + [pad_id] * right),
            response_mask=torch.tensor(row["response_mask"] + [0] * right),
            rollout_log_probs=torch.tensor(row["rollout_log_probs"] + [0.] * right, dtype=torch.float32),
            advantages=torch.tensor([row["advantage"]] * len(row["responses"]) + [0.] * right, dtype=torch.float32)))
        mm.append(vision)
    modal = np.empty(len(mm), dtype=object)
    modal[:] = mm
    data = DataProto.from_dict(tensors={k: torch.stack([t[k] for t in tensors]) for k in tensors[0]},
        non_tensors={"multi_modal_inputs": modal}, meta_info=dict(temperature=temperature, micro_batch_size=1,
        use_dynamic_bsz=False, formal_window_sha256=window["window_sha256"],
        row_ids=[row["logical_row_id"] for row in local], rank_plan=mapping))
    artifact = seal(dict(window_sha256=window["window_sha256"], **mapping,
        row_ids=data.meta_info["row_ids"], input_sha256=input_fingerprint(data),
        rollout_log_probs_sha256=fingerprint(data.batch["rollout_log_probs"]), mask_audit=masks), "batch_receipt_sha256")
    return data, FormalBatchReceipt(artifact, id(data), _FORMAL_BATCH_SEAL)


def assert_cpu_multimodal(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("formal window multimodal tensors must remain CPU-backed")
    elif isinstance(value, dict):
        for child in value.values():
            assert_cpu_multimodal(child)
    elif isinstance(value, (list, tuple, np.ndarray)):
        for child in value:
            assert_cpu_multimodal(child)


@contextmanager
def materialize_actor_microbatches(actor):
    """Intercept ONLY official _forward_micro_batch; never move the full window.

    Device copies live in the active forward/autograd graph only. Original nested
    CPU dictionaries are never mutated or retained as window-wide CUDA carriers.
    """
    import numpy as np
    import torch
    original = actor._forward_micro_batch
    def move(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v, device) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            result = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                result[index] = move(value[index], device)
            return result
        if isinstance(value, (list, tuple)):
            return type(value)(move(v, device) for v in value)
        return value
    def forward(micro_batch, *args, **kwargs):
        if micro_batch["responses"].shape[0] != 1:
            raise ValueError("formal actor requires microbatch=1")
        modal = micro_batch["multi_modal_inputs"]
        assert_cpu_multimodal(modal)
        device = micro_batch["input_ids"].device
        current = {**micro_batch, "multi_modal_inputs": move(modal, device)}
        return original(current, *args, **kwargs)
    actor._forward_micro_batch = forward
    try:
        yield
    finally:
        actor._forward_micro_batch = original


def sampled_logprobs(token_ids, logprobs):
    if logprobs is None or len(token_ids) != len(logprobs):
        raise ValueError("vLLM sampled logprobs missing/misaligned")
    result = []
    for token, mapping in zip(token_ids, logprobs, strict=True):
        if token not in mapping:
            raise ValueError("sampled token logprob absent")
        value = float(mapping[token].logprob)
        if not math.isfinite(value) or value > 1e-6:
            raise ValueError("invalid sampled logprob")
        result.append(value)
    return result


def response_loss_mask(length, *, step_index, fatal_step):
    if type(length) is not int or length < 1 or step_index < 0 or (fatal_step is not None and fatal_step < 0):
        raise ValueError("invalid response/fatal boundary")
    return [int(fatal_step is None or step_index <= fatal_step)] * length


def training_rows(group, final_advantages, *, rollout_schema=False):
    # Historical forensic readers retain their legacy row key. Formal Gate C
    # explicitly selects the new schema; neither key is ever a PPO denominator.
    validate_group(group)
    if len(final_advantages) != 2 or not all(math.isfinite(a) for a in final_advantages):
        raise ValueError("finite group advantages required")
    rows, audit = [], {"generated_tokens": 0, "masked_post_fatal_tokens": 0, "prompt_tokens": 0}
    for member in sorted(group["members"], key=lambda m: m["rollout_index"]):
        cutoff = member["fatal"]["fatal_step"] if member["fatal"]["fatal"] else None
        if cutoff is not None and not 0 <= cutoff < len(member["steps"]):
            raise ValueError("fatal cutoff outside actual rLLM steps")
        for index, step in enumerate(member["steps"]):
            mask = response_loss_mask(len(step["response_ids"]), step_index=index, fatal_step=cutoff)
            audit["generated_tokens"] += len(mask)
            audit["masked_post_fatal_tokens"] += len(mask) - sum(mask)
            audit["prompt_tokens"] += len(step["prompt_ids"])
            if any(mask):
                rows.append({"prompt_ids": step["prompt_ids"], "responses": step["response_ids"],
                             ("rollout_log_probs" if rollout_schema else "old_log_probs"): step["logprobs"], "response_mask": mask,
                             "advantage": float(final_advantages[member["rollout_index"]]),
                             "multimodal_file": step["multimodal_file"],
                             "rollout_index": member["rollout_index"], "step_index": index})
    if not rows:
        raise ValueError("no trainable generated prefix")
    audit["supervised_response_tokens"] = sum(sum(r["response_mask"]) for r in rows)
    audit["loss_mask_scope"] = "generated response only; all context/observation/padding excluded"
    return rows, audit


def mask_artifact(group, final_advantages, *, rollout_schema=False):
    rows, counts = training_rows(group, final_advantages, rollout_schema=rollout_schema)
    per_step = []
    for member in group["members"]:
        cutoff = member["fatal"]["fatal_step"] if member["fatal"]["fatal"] else None
        for index, step in enumerate(member["steps"]):
            mask = response_loss_mask(len(step["response_ids"]), step_index=index, fatal_step=cutoff)
            per_step.append({"rollout_index": member["rollout_index"], "step_index": index,
                             "fatal_cutoff": cutoff, "prompt_token_count": len(step["prompt_ids"]),
                             "response_mask": mask, "loss_mask": [0] * len(step["prompt_ids"]) + mask})
    return {"identity": group["identity"], "rows": rows, "per_step_masks": per_step,
            "counts": counts, "formal_rl_initialization_allowed": False}


def build_dataproto(rows, *, directory: Path, model, pad_id: int, device, temperature):
    import numpy as np
    import torch
    from verl import DataProto

    if not rows or not 0 < temperature or not isinstance(pad_id, int):
        raise ValueError("invalid training batch")
    p, r = max(len(row["prompt_ids"]) for row in rows), max(len(row["responses"]) for row in rows)
    tensor_rows, mm = [], []
    # Qwen3-VL-4B pinned real M-RoPE, not generic arange/text-only positions.
    rope_model = model.get_base_model().model
    for row in rows:
        # Deprecated legacy row old_log_probs is saved vLLM R, NOT actor O.
        rollout = row.get("rollout_log_probs", row.get("old_log_probs"))
        if (rollout is None or len(rollout) != len(row["responses"])
                or ("rollout_log_probs" in row and "old_log_probs" in row
                    and row["rollout_log_probs"] != row["old_log_probs"])):
            raise ValueError("missing/inconsistent saved rollout logprobs")
        path = (directory / row["multimodal_file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("unsafe multimodal artifact path")
        original = torch.load(path, map_location="cpu", weights_only=True)
        if original["input_ids"][0].tolist() != row["prompt_ids"]:
            raise ValueError("actual vLLM prompt IDs differ from saved processor inputs")
        vision = {k: v.to(device) for k, v in original.items() if k in {"pixel_values", "image_grid_thw"}}
        if set(vision) != {"pixel_values", "image_grid_thw"} or not vision["pixel_values"].numel():
            raise ValueError("real multimodal actor inputs required")
        ids = torch.tensor([row["prompt_ids"] + row["responses"]], dtype=torch.long, device=device)
        positions, _ = rope_model.get_rope_index(input_ids=ids, image_grid_thw=vision["image_grid_thw"],
                                                attention_mask=torch.ones_like(ids))
        left, right = p - len(row["prompt_ids"]), r - len(row["responses"])
        input_ids = [pad_id] * left + row["prompt_ids"] + row["responses"] + [pad_id] * right
        attention = [0] * left + [1] * ids.shape[-1] + [0] * right
        tensor_rows.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
            "position_ids": torch.nn.functional.pad(positions[:, 0], (left, right), value=0),
            "responses": torch.tensor(row["responses"] + [pad_id] * right, dtype=torch.long, device=device),
            "response_mask": torch.tensor(row["response_mask"] + [0] * right, dtype=torch.long, device=device),
            # Zeros below are PADDED, fully masked positions only, never fake sampled logprobs.
            "rollout_log_probs": torch.tensor(rollout + [0.] * right, dtype=torch.float32, device=device),
            "advantages": torch.tensor([row["advantage"]] * len(row["responses"]) + [0.] * right,
                                       dtype=torch.float32, device=device),
        })
        mm.append(vision)  # DataProto.to() does NOT move non_tensor dict tensors.
    tensors = {key: torch.stack([row[key] for row in tensor_rows]) for key in tensor_rows[0]}
    modal = np.empty(len(mm), dtype=object)
    modal[:] = mm
    return DataProto.from_dict(tensors=tensors, non_tensors={"multi_modal_inputs": modal},
                               meta_info={"temperature": temperature, "micro_batch_size": 1,
                                          "use_dynamic_bsz": False})
