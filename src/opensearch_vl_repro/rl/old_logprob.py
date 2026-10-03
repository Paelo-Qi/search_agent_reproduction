"""Production Gate C same-policy actor-old carrier and update-boundary receipt.

R is immutable rollout evidence. Only the FIRST independent actor computation
may install O. The second computation C is a strict check, not an update proxy.
No forensic source scanners, model proxies, losses or optimizer calls here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, local_tensor
from opensearch_vl_repro.rl.policy_alignment import (
    alignment_artifact, alignment_checks, compare_policy_logprobs, temperature_matches,
)
from opensearch_vl_repro.rl.rollout_sync import validate_merged_files
from opensearch_vl_repro.sft_tool_audit import sha256_file
from opensearch_vl_repro.rl.rl_actor_semantics import (
    RL_LORA_DROPOUT_RUNTIME_VERSION, contract_sft_config, require_execution_binding,
    require_saved_source_dropout, require_rl_lora_dropout_runtime, require_runtime_audit,
)

OLD_SOURCE = "actor_recomputed_pre_update"
ROLLOUT_SOURCE = "saved_vllm_processed_logprobs"
UPDATE_ROLLOUT_SOURCE = "vllm_processed_logprobs"
HANDOFF_REASON = "same-policy rollout/training execution-representation numerical handoff diagnostic"
ALIGNMENT_META = dict(comparison="actor_recomputed_old_vs_actor_current",
    old_logprob_source=OLD_SOURCE,
    current_logprob_source="second_independent_pre_update_actor_forward",
    rollout_logprob_source=ROLLOUT_SOURCE,
    rollout_log_probs_not_used_as_ppo_denominator=True)
OLD_LOGPROB_CHECKS = (
    "rollout_log_probs_preserved", "actor_old_log_probs_recomputed",
    "actor_old_log_probs_installed", "actor_old_current_alignment_passed",
    "same_policy_lineage_verified", "rollout_actor_handoff_finite",
    "rollout_actor_handoff_token_count_match", "ppo_denominator_source_verified",
)
_SEAL = object()


class SamePolicyLineage(dict):
    """In-process capability, issued only after file/source lineage validation."""
    def __init__(self, values, seal):
        super().__init__(values)
        self.seal = seal
        self.sha256 = canonical_json_sha256(values)


def fingerprint(value):
    """Typed full-byte SHA, including real local DTensor shards and vision data."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            t = local_tensor(v).detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(t.dtype), list(t.shape)]).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            digest.update(json.dumps(["array", str(v.dtype), list(v.shape)]).encode())
            for item in v.flat:
                visit(item)
        elif isinstance(v, dict):
            digest.update(b"dict:")
            for k in sorted(v):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            digest.update(type(v).__name__.encode() + str(len(v)).encode())
            for item in v:
                visit(item)
        elif isinstance(v, np.generic):
            visit(v.item())
        elif v is None or type(v) in (str, bool, int, float):
            digest.update(json.dumps([type(v).__name__, v], allow_nan=False).encode())
        else:
            raise TypeError(f"unsupported production fingerprint type: {type(v)}")
    visit(value)
    return digest.hexdigest()


def input_fingerprint(data):
    # Logprob carriers have separate receipts, never participate in input identity.
    return fingerprint(dict(tensors={k: data.batch[k] for k in sorted(data.batch.keys())
        if k not in {"old_log_probs", "rollout_log_probs"}},
        multimodal=data.non_tensor_batch, metadata=data.meta_info))


def parameter_fingerprint(actor):
    roster = [(name, list(p.shape), str(getattr(p, "placements", None)), p.requires_grad,
               fingerprint(p)) for name, p in sorted(actor.actor_module.named_parameters())]
    if not roster:
        raise ValueError("empty actor parameter roster")
    return fingerprint(roster)


def rng_fingerprint():
    import numpy as np
    import torch
    return fingerprint(dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))


def verify_same_policy_lineage(ctx, group, output):
    """Rebind collection to ORIGINAL checkpoint-3k and its actual static merge."""
    from opensearch_vl_repro.inference.adapter import adapter_file_identity
    from opensearch_vl_repro.rl.group import validate_group
    validate_group(group)
    identity, actor = ctx["identity"], ctx["actor"]
    contract, effective_fp = require_execution_binding(identity, group)
    gid = group["identity"]
    fp = actor["source_sft_adapter_fingerprint"]
    adapter = ctx["adapter"].resolve()
    formal_suffix = "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    if not adapter.as_posix().endswith("/" + formal_suffix):
        raise ValueError("actor source adapter is not formal checkpoint-3k")
    source_adapter = adapter_file_identity(adapter)
    require_saved_source_dropout(adapter)
    if (source_adapter["adapter_fingerprint"] != fp
            or actor.get("actor_adapter_fingerprint") != fp
            or actor.get("actor_source_kind") != "formal_sft_checkpoint_fallback"
            or identity.get("source_sft_actor") != actor
            or gid.get("pre_update_policy_fingerprint") != effective_fp
            or gid.get("context") != identity["identity_sha256"]
            or gid.get("prompt_id") != identity["prompt_id"]
            or gid.get("rollout_config_fingerprint") != canonical_json_sha256(ctx["gate"])
            or (actor.get("base_model"), actor.get("base_revision")) != (BASE_MODEL, BASE_REVISION)
            or (identity.get("base_model"), identity.get("base_revision")) != (BASE_MODEL, BASE_REVISION)
            or identity.get("logprobs_mode") != "processed_logprobs"):
        raise ValueError("stale/wrong collection policy or actor source lineage")
    directory = (output / ("merged-" + gid["collection_attempt"])).resolve()
    if directory.parent != output.resolve():
        raise ValueError("unsafe merged collection path")
    manifest_path = directory / "merge_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merge = manifest["identity"]
    digest = merge["merged_checkpoint_fingerprint"]
    payload = {k: v for k, v in merge.items() if k != "merged_checkpoint_fingerprint"}
    files = validate_merged_files(directory)
    files.pop("merge_manifest.json", None)
    if (canonical_json_sha256(payload) != digest
            or any(manifest.get(k) is not True for k in ("merge_complete", "no_active_peft", "fresh_hf_forward_finite", "merge_hf_destroyed", "reload_hf_destroyed"))
            or gid.get("context") != identity["identity_sha256"]
            or group.get("merged_checkpoint_fingerprint") != digest
            or manifest.get("actor_provenance") != actor
            or merge.get("merged_file_sha256") != files
            or merge.get("actor_adapter_fingerprint") != fp
            or merge.get("source_sft_adapter_fingerprint") != fp
            or merge.get("actor_source_kind") != actor["actor_source_kind"]
            or merge.get("source_sft_lineage") != actor["source_sft_lineage"]
            or merge.get("actor_gate_identity_sha256") != actor["actor_gate_identity_sha256"]
            or merge.get("merge_method") != "peft.merge_and_unload"
            or merge.get("software_versions") != ctx["versions"]
            or merge.get("merged_config_fingerprint") != files.get("config.json")
            or (merge.get("base_model"), merge.get("base_revision")) != (BASE_MODEL, BASE_REVISION)
            or merge.get("runtime_image_protocol_version") != identity["runtime_protocol"]
            or merge.get("formal_rl_initialization_allowed") is not False):
        raise ValueError("merged checkpoint collection/actor lineage mismatch")
    source_files = {str(manifest_path): sha256_file(manifest_path),
                    str(output / "group/group.json"): sha256_file(output / "group/group.json")}
    source_files.update({str(adapter / name): digest for name, digest in source_adapter["file_sha256"].items()})
    return SamePolicyLineage(dict(same_policy_lineage_verified=True,
        actor_adapter_fingerprint=fp, pre_update_policy_fingerprint=effective_fp,
        effective_pre_update_policy_fingerprint=effective_fp, rl_policy_execution_contract=contract,
        actor_source_adapter=str(adapter), actor_source_kind=actor["actor_source_kind"],
        base_model=BASE_MODEL, base_revision=BASE_REVISION,
        identity_sha256=identity["identity_sha256"], trajectory_group_id=gid["trajectory_group_id"],
        collection_policy_identity=gid, merged_checkpoint_fingerprint=digest,
        merge_manifest_sha256=sha256_file(manifest_path), group_sha256=sha256_file(output / "group/group.json"),
        source_file_sha256=source_files), _SEAL)


def require_source_files(lineage):
    from pathlib import Path
    files = lineage.get("source_file_sha256")
    if not files or any(not Path(name).is_file() or sha256_file(Path(name)) != digest for name, digest in files.items()):
        raise ValueError("source policy/group/merge evidence changed since lineage verification")


def independent_actor_compute(actor, data, *, parameters, inputs, rollout, rng, sft_config):
    """Actual eval/no-grad actor forward on a private copy of identical inputs."""
    import torch
    if parameter_fingerprint(actor) != parameters or input_fingerprint(data) != inputs:
        raise ValueError("actor parameters or inputs mutated before recompute")
    if fingerprint(data.batch["rollout_log_probs"]) != rollout or rng_fingerprint() != rng:
        raise ValueError("rollout carrier or RNG mutated before recompute")
    if any(p.grad is not None for p in actor.actor_module.parameters()) or actor.actor_optimizer.state:
        raise ValueError("backward/optimizer state present before actor-old recompute")
    batch = copy.deepcopy(data)
    old_before = fingerprint(batch.batch["old_log_probs"]) if "old_log_probs" in batch.batch else None
    forwards = []
    def observe(module, args):
        require_rl_lora_dropout_runtime(module, sft_config)
        if (module.training or torch.is_grad_enabled() or any(
                m.training for m in module.modules() if isinstance(m, torch.nn.modules.dropout._DropoutNd))):
            raise ValueError("actor recompute must execute deterministic eval/no_grad")
        if any(p.device.type == "cuda" for p in module.parameters()) and (
                not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16):
            raise ValueError("actor recompute CUDA forward must use BF16 autocast")
        forwards.append(True)
    hook = actor.actor_module.register_forward_pre_hook(observe)
    try:
        with torch.no_grad():
            result, _ = actor.compute_log_prob(batch, calculate_entropy=False)
        if (not forwards or result.requires_grad or result.shape != data.batch["responses"].shape
                or not bool(torch.isfinite(result).all())):
            raise ValueError("missing/nonfinite/misaligned independent actor logprobs")
        old_after = fingerprint(batch.batch["old_log_probs"]) if "old_log_probs" in batch.batch else None
        if (old_before != old_after or input_fingerprint(batch) != inputs
                or input_fingerprint(data) != inputs or parameter_fingerprint(actor) != parameters
                or fingerprint(batch.batch["rollout_log_probs"]) != rollout
                or fingerprint(data.batch["rollout_log_probs"]) != rollout or rng_fingerprint() != rng
                or actor.actor_optimizer.state or any(p.grad is not None for p in actor.actor_module.parameters())):
            raise ValueError("actor compute mutated parameters/input/carriers/RNG or ran backward/optimizer")
        return result.detach().clone(), len(forwards)
    finally:
        hook.remove()


def handoff_metrics(old, rollout, mask, expected_count):
    """Numerical magnitude is informational; structural/nonfinite errors block."""
    import torch
    audit = compare_policy_logprobs(old, rollout, mask, clip_ratio_low=.2, clip_ratio_high=.28,
                                    expected_masked_token_count=expected_count)
    if not all(audit.get(k) is True for k in ("shape_match", "token_count_match", "logprobs_finite", "ratio_finite", "all_finite")):
        raise ValueError("invalid rollout/actor handoff structure/finite token evidence")
    # Do not publish a misleading alignment PASS/FAIL for R/O magnitude.
    for k in ("passed", "failure", "max_abs_logprob_diff_bound"):
        audit.pop(k, None)
    diff = old.detach()[mask.bool()].double() - rollout.detach()[mask.bool()].double()
    audit["absolute_diff_percentiles"] = {str(q): torch.quantile(diff.abs(), q).item() for q in (.5, .9, .95, .99, 1.)}
    audit["outlier_counts"] = {str(v): int((diff.abs() >= v).sum().item()) for v in (.01, .05, .1, .2, .5)}
    return audit


@dataclass(frozen=True)
class OldLogprobReceipt:
    artifact: dict
    data_id: int
    actor_id: int
    old_tensor: object = field(repr=False)
    seal: object = field(repr=False)


def prepare_actor_old_log_probs(actor, data, gate, *, lineage, expected_masked_token_count, rank):
    import torch
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("actor-old recompute requires rank 0 or 1")
    if (not isinstance(lineage, SamePolicyLineage) or lineage.seal is not _SEAL
            or lineage.sha256 != canonical_json_sha256(dict(lineage))
            or lineage.get("same_policy_lineage_verified") is not True):
        raise ValueError("same-policy lineage verification required before actor recompute")
    require_source_files(lineage)
    sft_config = contract_sft_config(lineage.get("rl_policy_execution_contract"))
    boundary_audits = {}
    def boundary(name):
        audit = require_rl_lora_dropout_runtime(actor.actor_module, sft_config)
        boundary_audits[name] = audit
        return audit["lora_dropout_runtime_sha256"]
    if "old_log_probs" in data.batch or "rollout_log_probs" not in data.batch:
        raise ValueError("initial DataProto must carry rollout R and NO old_log_probs")
    if not temperature_matches(data.meta_info.get("temperature"), gate["vllm"]["temperature"]):
        raise ValueError("actor/rollout temperature mismatch")
    r, mask = data.batch["rollout_log_probs"], data.batch["response_mask"]
    if (r.dtype != torch.float32 or r.shape != data.batch["responses"].shape
            or r.shape != mask.shape or not bool(torch.isfinite(r).all())
            or not bool(((mask == 0) | (mask == 1)).all())
            or int(mask.sum().item()) != expected_masked_token_count or expected_masked_token_count < 1):
        raise ValueError("invalid saved rollout FP32 shape/finite mask/token count")
    inputs, parameters, rng, rollout = input_fingerprint(data), parameter_fingerprint(actor), rng_fingerprint(), fingerprint(r)
    signature = boundary("old_before")
    old, old_forwards = independent_actor_compute(actor, data, parameters=parameters, inputs=inputs, rollout=rollout, rng=rng, sft_config=sft_config)
    if boundary("old_after") != signature or boundary("current_before") != signature:
        raise ValueError("RL dropout semantics changed around O recompute")
    data.batch["old_log_probs"] = old.detach().clone()
    installed = data.batch["old_log_probs"]
    if installed.untyped_storage().data_ptr() == r.untyped_storage().data_ptr():
        raise ValueError("actor-old and rollout carrier alias")
    current, current_forwards = independent_actor_compute(actor, data, parameters=parameters, inputs=inputs, rollout=rollout, rng=rng, sft_config=sft_config)
    if boundary("current_after") != signature:
        raise ValueError("RL dropout semantics changed around C recompute")
    if not torch.equal(installed, old):
        raise ValueError("installed actor-old mutated during second forward")
    artifact = dict(schema_version=1, rank=rank, **lineage, **ALIGNMENT_META,
        input_sha256=inputs, old_log_probs_sha256=fingerprint(installed), rollout_log_probs_sha256=rollout,
        response_mask_sha256=fingerprint(mask), token_count=expected_masked_token_count, temperature=.7,
        pre_update_parameter_sha256=parameters, parameters_unchanged=True, rng_unchanged=True,
        independent_actor_compute_count=2, old_forward_count=old_forwards, current_forward_count=current_forwards,
        optimizer_step_count=0, formal_rl_initialization_allowed=False)
    artifact.update(rl_lora_dropout_runtime_version=RL_LORA_DROPOUT_RUNTIME_VERSION,
        source_adapter_lora_dropout=.05, runtime_effective_lora_dropout=0., lora_dropout_target_count=252,
        lora_dropout_runtime_sha256=signature, rl_dropout_boundary_audits=boundary_audits)
    artifact["receipt_sha256"] = canonical_json_sha256(artifact)
    receipt = OldLogprobReceipt(artifact, id(data), id(actor), installed, _SEAL)
    alignment = compare_policy_logprobs(current, installed, mask, clip_ratio_low=gate["clip_ratio_low"],
        clip_ratio_high=gate["clip_ratio_high"], expected_masked_token_count=expected_masked_token_count)
    alignment.update(rank=rank, logprobs_computed=True, temperature=.7, rollout_temperature=gate["vllm"]["temperature"],
        **ALIGNMENT_META, old_logprob_receipt_sha256=artifact["receipt_sha256"],
        old_log_probs_sha256=artifact["old_log_probs_sha256"], input_sha256=inputs)
    alignment["checks"] = alignment_checks(alignment)
    alignment["passed"] = all(alignment["checks"].values())
    handoff = dict(rank=rank, comparison="rollout_vs_actor_recomputed_old", **lineage,
        metrics=handoff_metrics(installed, r, mask, expected_masked_token_count), temperature=.7,
        token_count=expected_masked_token_count, old_logprob_source=OLD_SOURCE, rollout_logprob_source=ROLLOUT_SOURCE,
        old_log_probs_sha256=artifact["old_log_probs_sha256"], rollout_log_probs_sha256=rollout,
        input_sha256=inputs, old_logprob_receipt_sha256=artifact["receipt_sha256"],
        informational_only=True, gate_blocking=False,
        reason=HANDOFF_REASON,
        formal_rl_initialization_allowed=False)
    return dict(receipt=receipt, alignment=alignment, handoff=handoff)


def verify_update_receipt(actor, data, alignment, receipt):
    """The actual update boundary, not a caller-set boolean, owns this guard."""
    if (not isinstance(receipt, OldLogprobReceipt) or receipt.seal is not _SEAL
            or receipt.data_id != id(data) or receipt.actor_id != id(actor)):
        raise ValueError("missing/forged/stale actor-old receipt")
    a = receipt.artifact
    if a.get("receipt_sha256") != canonical_json_sha256({k: v for k, v in a.items() if k != "receipt_sha256"}):
        raise ValueError("actor-old receipt content changed")
    require_source_files(a)
    audit = require_rl_lora_dropout_runtime(actor.actor_module, contract_sft_config(a.get("rl_policy_execution_contract")))
    if audit["lora_dropout_runtime_sha256"] != a.get("lora_dropout_runtime_sha256"):
        raise ValueError("actor-old/update dropout runtime fingerprint mismatch")
    old = data.batch.get("old_log_probs")
    r = data.batch.get("rollout_log_probs")
    if (old is None or r is None or old is not receipt.old_tensor
            or old.untyped_storage().data_ptr() == r.untyped_storage().data_ptr()
            or fingerprint(old) != a["old_log_probs_sha256"] or fingerprint(r) != a["rollout_log_probs_sha256"]
            or fingerprint(data.batch["response_mask"]) != a["response_mask_sha256"]
            or input_fingerprint(data) != a["input_sha256"] or parameter_fingerprint(actor) != a["pre_update_parameter_sha256"]
            or data.meta_info.get("temperature") != a["temperature"]
            or a.get("same_policy_lineage_verified") is not True or actor.config.use_rollout_log_probs is not True):
        raise ValueError("PPO denominator/carrier/input/parameter/lineage receipt mismatch")
    evidence = alignment
    if "per_rank" in alignment:
        rows = [row for row in alignment["per_rank"] if row["rank"] == a["rank"]]
        if len(rows) != 1:
            raise ValueError("missing rank-specific old receipt alignment")
        evidence = rows[0]
    if (any(evidence.get(k) != v for k, v in ALIGNMENT_META.items())
            or evidence.get("old_logprob_receipt_sha256") != a["receipt_sha256"]
            or evidence.get("old_log_probs_sha256") != a["old_log_probs_sha256"]
            or evidence.get("input_sha256") != a["input_sha256"]
            or evidence.get("masked_token_count") != a["token_count"]):
        raise ValueError("strict actor alignment is not bound to installed O receipt")
    if not all(alignment_checks(evidence).values()):
        raise ValueError("strict actor alignment numeric evidence failed")
    if actor.actor_optimizer.state or any(p.grad is not None for p in actor.actor_module.parameters()):
        raise ValueError("backward/optimizer occurred before receipt consumption")


def actor_alignment_artifact(per_rank, **kwargs):
    for row in per_rank:
        if any(row.get(k) != v for k, v in ALIGNMENT_META.items()):
            raise ValueError("strict alignment must compare actor O/C, not rollout R")
    return {**alignment_artifact(per_rank, **kwargs), **ALIGNMENT_META}


def paired_artifact(per_rank, *, identity, kind):
    if len(per_rank) != 2 or any(type(r["rank"]) is not int for r in per_rank) or {r["rank"] for r in per_rank} != {0, 1}:
        raise ValueError("old/handoff evidence requires both actor ranks")
    return dict(schema_version=1, kind=kind, identity=identity, gate_version=identity["gate_version"],
                per_rank=sorted(per_rank, key=lambda r: r["rank"]), formal_rl_initialization_allowed=False,
                **(dict(informational_only=True, gate_blocking=False, reason=HANDOFF_REASON) if kind == "rollout_actor_handoff" else {}))


def verify_old_logprob_artifacts(output, update, group, identity, lineage):
    """Finalizer: require newly bound O receipts/handoff, not legacy R alignment.

    Tensor receipts certify observed forwards; source R and mask hashes can also be
    reconstructed here without any model/GPU/API invocation.
    """
    import math
    import torch
    from opensearch_vl_repro.rl.training_batch import mask_artifact, training_rows
    required = dict(old_logprob_source=OLD_SOURCE, rollout_logprob_source=UPDATE_ROLLOUT_SOURCE,
        rollout_log_probs_preserved=True, ppo_denominator_source='data.batch["old_log_probs"]',
        same_policy_lineage_verified=True, optimizer_step_count=1, formal_rl_initialization_allowed=False)
    if any(update.get(k) != v or type(update.get(k)) is not type(v) for k, v in required.items()):
        raise ValueError("new actor-old PPO denominator/source/step evidence required")
    if update.get("pre_update_actor_alignment_sha256") != update.get("pre_update_policy_alignment_sha256"):
        raise ValueError("actor alignment checksum alias mismatch")
    artifacts = {}
    for kind in ("actor_old_logprob_receipt", "rollout_actor_handoff"):
        path = output / (kind + ".json")
        if not path.is_file() or sha256_file(path) != update.get(kind + "_sha256"):
            raise ValueError(f"missing/changed {kind} artifact")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value != paired_artifact(value["per_rank"], identity=identity, kind=kind):
            raise ValueError(f"wrong {kind} identity/rank evidence")
        artifacts[kind] = value
    rows, counts = training_rows(group, update["final_advantages"], rollout_schema=True)
    masks = json.loads((output / "training_masks.json").read_text(encoding="utf-8"))
    if masks != mask_artifact(group, update["final_advantages"], rollout_schema=True) or counts != update["training_token_counts"]:
        raise ValueError("formal response/fatal masks or token counts changed")
    length = max(len(row["responses"]) for row in rows)
    rollout = torch.tensor([r["rollout_log_probs"] + [0.] * (length - len(r["responses"])) for r in rows], dtype=torch.float32)
    mask = torch.tensor([r["response_mask"] + [0] * (length - len(r["responses"])) for r in rows], dtype=torch.long)
    r_hash, mask_hash = fingerprint(rollout), fingerprint(mask)
    receipts = artifacts["actor_old_logprob_receipt"]["per_rank"]
    handoffs = artifacts["rollout_actor_handoff"]["per_rank"]
    actor_rows = sorted(update["per_rank"], key=lambda r: r["rank"])
    alignments = sorted(update["pre_update_policy_alignment"]["per_rank"], key=lambda r: r["rank"])
    for rank, (receipt, handoff, actor_row, alignment) in enumerate(zip(receipts, handoffs, actor_rows, alignments, strict=True)):
        contract = identity["rl_policy_execution_contract"]
        if (receipt.get("rl_lora_dropout_runtime_version") != RL_LORA_DROPOUT_RUNTIME_VERSION
                or receipt.get("source_adapter_lora_dropout") != .05
                or receipt.get("runtime_effective_lora_dropout") != 0.
                or receipt.get("lora_dropout_target_count") != 252):
            raise ValueError("missing old-logprob RL dropout execution evidence")
        for boundary in ("old_before", "old_after", "current_before", "current_after"):
            audit = require_runtime_audit(receipt.get("rl_dropout_boundary_audits", {}).get(boundary, {}), contract)
            if audit["lora_dropout_runtime_sha256"] != receipt.get("lora_dropout_runtime_sha256"):
                raise ValueError("old/current RL dropout execution fingerprint mismatch")
        if (receipt["rank"] != rank or actor_row["rank"] != rank or handoff["rank"] != rank
                or any(receipt.get(k) != v or handoff.get(k) != v for k, v in lineage.items())
                or any(receipt.get(k) != v for k, v in ALIGNMENT_META.items())
                or receipt.get("receipt_sha256") != canonical_json_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
                or receipt.get("rollout_log_probs_sha256") != r_hash or receipt.get("response_mask_sha256") != mask_hash
                or receipt.get("token_count") != counts["supervised_response_tokens"]
                or receipt.get("temperature") != .7 or receipt.get("independent_actor_compute_count") != 2
                or receipt.get("optimizer_step_count") != 0 or receipt.get("formal_rl_initialization_allowed") is not False
                or receipt.get("parameters_unchanged") is not True or receipt.get("rng_unchanged") is not True
                or type(receipt.get("old_forward_count")) is not int or receipt["old_forward_count"] < 1
                or type(receipt.get("current_forward_count")) is not int or receipt["current_forward_count"] < 1
                or actor_row.get("actor_old_logprob_receipt") != receipt or actor_row.get("rollout_actor_handoff") != handoff
                or actor_row.get("optimizer_step_count") != 1):
            raise ValueError("actor-old receipt lineage/forward/token/rank evidence mismatch")
        for key in ("old_log_probs_sha256", "input_sha256", "pre_update_parameter_sha256"):
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("missing actor-old tensor/input/parameter fingerprint")
        if (any(alignment.get(k) != receipt[k] for k in ("old_log_probs_sha256", "input_sha256"))
                or alignment.get("old_logprob_receipt_sha256") != receipt["receipt_sha256"]
                or any(handoff.get(k) != receipt[k] for k in ("old_log_probs_sha256", "rollout_log_probs_sha256", "input_sha256", "token_count", "temperature"))
                or handoff.get("old_logprob_receipt_sha256") != receipt["receipt_sha256"]
                or handoff.get("old_logprob_source") != OLD_SOURCE or handoff.get("rollout_logprob_source") != ROLLOUT_SOURCE
                or handoff.get("informational_only") is not True or handoff.get("gate_blocking") is not False
                or handoff.get("reason") != HANDOFF_REASON
                or "passed" in handoff or "alignment_passed" in handoff
                or handoff.get("formal_rl_initialization_allowed") is not False):
            raise ValueError("O alignment or R handoff not bound to receipt")
        metrics = handoff["metrics"]
        if (not all(metrics.get(k) is True for k in ("all_finite", "logprobs_finite", "ratio_finite", "shape_match", "token_count_match"))
                or metrics.get("masked_token_count") != counts["supervised_response_tokens"]
                or metrics.get("expected_masked_token_count") != counts["supervised_response_tokens"]
                or "passed" in metrics or "alignment_passed" in metrics):
            raise ValueError("R/O handoff structural/finite/count evidence failed")
        numeric = [metrics.get(k) for k in ("mean_abs_logprob_diff", "max_abs_logprob_diff", "mean_signed_logprob_diff",
            "mean_importance_ratio", "min_importance_ratio", "max_importance_ratio", "initial_clip_fraction")]
        percentiles, outliers = metrics.get("absolute_diff_percentiles", {}), metrics.get("outlier_counts", {})
        if (set(percentiles) != {"0.5", "0.9", "0.95", "0.99", "1.0"}
                or set(outliers) != {"0.01", "0.05", "0.1", "0.2", "0.5"}
                or any(type(v) is not int or not 0 <= v <= counts["supervised_response_tokens"] for v in outliers.values())
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric + list(percentiles.values()))):
            raise ValueError("missing/nonfinite handoff metrics/percentiles/outliers")
        if any(actor_row["checks"].get(k) is not True or update["checks"].get(k) is not True for k in OLD_LOGPROB_CHECKS):
            raise ValueError("not all new actor-old checks completed")
    if len({r["input_sha256"] for r in receipts}) != 1:
        raise ValueError("replicated actor group input identities differ across ranks")
    return artifacts
