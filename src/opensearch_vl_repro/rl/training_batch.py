"""Real per-generation rLLM tokens -> verl DataProto; no SFT labels/CE.

Each generation is one row with its own actual context (including observations).
All ranks use the same complete group in this small gate to keep FSDP collective
counts identical despite different trajectory lengths. No dummy rank padding.
"""
from __future__ import annotations

import math
from pathlib import Path

from opensearch_vl_repro.rl.group import validate_group


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


def training_rows(group, final_advantages):
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
                             "old_log_probs": step["logprobs"], "response_mask": mask,
                             "advantage": float(final_advantages[member["rollout_index"]]),
                             "multimodal_file": step["multimodal_file"],
                             "rollout_index": member["rollout_index"], "step_index": index})
    if not rows:
        raise ValueError("no trainable generated prefix")
    audit["supervised_response_tokens"] = sum(sum(r["response_mask"]) for r in rows)
    audit["loss_mask_scope"] = "generated response only; all context/observation/padding excluded"
    return rows, audit


def mask_artifact(group, final_advantages):
    rows, counts = training_rows(group, final_advantages)
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
            "old_log_probs": torch.tensor(row["old_log_probs"] + [0.] * right, dtype=torch.float32, device=device),
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
