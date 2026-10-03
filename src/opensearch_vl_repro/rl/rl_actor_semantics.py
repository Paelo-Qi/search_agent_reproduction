"""Formal RL execution semantics. Source PEFT metadata is never rewritten.

Import safe; torch is imported only while inspecting a live actor. This module
does not implement a loss, optimizer, merge, or forensic diagnostic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION

RL_LORA_DROPOUT_RUNTIME_VERSION = "rl-lora-dropout-disabled-v1"
RL_POLICY_EXECUTION_VERSION = "rl-policy-execution-v1-zero-lora-dropout"
RL_LORA_DROPOUT_EFFECTIVE_P = 0.0
SOURCE_LORA_DROPOUT = .05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_TARGET_COUNT = 36 * len(LORA_TARGETS)
_TARGET = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)$")
DROPOUT_CHECKS = (
    "rl_lora_dropout_runtime_configured", "rl_lora_dropout_source_config_preserved",
    "rl_lora_dropout_zero_before_old", "rl_lora_dropout_zero_before_update",
    "rl_lora_dropout_zero_during_update", "rl_update_train_mode_forward_seen",
    "rl_dropout_execution_contract_verified", "policy_nonzero_dropout_absent",
    "fresh_rl_dropout_runtime_verified", "native_rl_dropout_runtime_verified",
)


def execution_contract(sft_config):
    lora = sft_config["lora"]
    if (lora.get("dropout") != SOURCE_LORA_DROPOUT or lora.get("rank") != 16
            or lora.get("alpha") != 32 or tuple(lora.get("target_modules", ())) != LORA_TARGETS):
        raise ValueError("formal RL requires unchanged source r16/alpha32/dropout .05 target configuration")
    return dict(version=RL_POLICY_EXECUTION_VERSION,
        rl_lora_dropout_runtime_version=RL_LORA_DROPOUT_RUNTIME_VERSION,
        source_adapter_lora_dropout=SOURCE_LORA_DROPOUT,
        runtime_effective_lora_dropout=RL_LORA_DROPOUT_EFFECTIVE_P,
        active_adapter="default", target_modules=list(LORA_TARGETS),
        decoder_layer_count=36, lora_dropout_target_count=LORA_TARGET_COUNT)


def contract_sft_config(contract):
    """Accept only the frozen contract, not arbitrary receipt-supplied settings."""
    config = {"lora": dict(rank=16, alpha=32, dropout=.05, target_modules=list(LORA_TARGETS))}
    if contract != execution_contract(config):
        raise ValueError("wrong/missing RL dropout execution contract")
    return config


def effective_policy_fingerprint(source_fingerprint, contract):
    contract_sft_config(contract)
    return canonical_json_sha256(dict(source_adapter_fingerprint=source_fingerprint,
        base_model=BASE_MODEL, base_revision=BASE_REVISION, rl_policy_execution_contract=contract))


def _active_default(module):
    active = getattr(module, "active_adapters", None)
    if active is None:
        active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        active = [active]
    if active != ["default"]:
        raise ValueError("only the default active LoRA adapter is permitted")
    if getattr(module, "disable_adapters", False) or getattr(module, "_disable_adapters", False):
        raise ValueError("disabled LoRA adapters are not a formal RL policy")
    if getattr(module, "merged_adapters", []) or getattr(module, "merged", False):
        raise ValueError("merged LoRA adapters are not a trainable formal RL actor")


def _source_roster(model, sft_config):
    import torch
    contract = execution_contract(sft_config)
    configs = getattr(model, "peft_config", {})
    if set(configs) != {"default"}:
        raise ValueError("exactly one default PEFT configuration required")
    cfg = configs["default"]
    if (cfg.lora_dropout != .05 or cfg.r != 16 or cfg.lora_alpha != 32
            or set(cfg.target_modules) != set(LORA_TARGETS)):
        raise ValueError("source PEFT configuration changed; dropout must remain .05")
    _active_default(model)
    roster, modules, seen = [], [], set()
    for name, module in model.named_modules():
        if not any(hasattr(module, k) for k in ("lora_A", "lora_B", "lora_dropout")):
            continue
        match = _TARGET.search(name)
        if not match or match[3] not in LORA_TARGETS:
            raise ValueError(f"unexpected non-language/vision/projector LoRA target: {name}")
        layer, branch, target = int(match[1]), match[2], match[3]
        if layer not in range(36) or branch != ("self_attn" if target in LORA_TARGETS[:4] else "mlp"):
            raise ValueError(f"unexpected LoRA layer/branch: {name}")
        key = (layer, target)
        if key in seen:
            raise ValueError(f"duplicate LoRA target: {key}")
        seen.add(key)
        _active_default(module)
        for field in ("lora_A", "lora_B", "lora_dropout"):
            container = getattr(module, field, None)
            if not isinstance(container, torch.nn.ModuleDict) or set(container) != {"default"}:
                raise ValueError(f"missing/extra PEFT ModuleDict {field}: {name}")
        a, b = module.lora_A["default"], module.lora_B["default"]
        if any(not list(m.parameters()) or not all(p.requires_grad for p in m.parameters()) for m in (a, b)):
            raise ValueError(f"nontrainable LoRA A/B: {name}")
        dropout = module.lora_dropout["default"]
        if not isinstance(dropout, torch.nn.Dropout):
            raise ValueError(f"source .05 must produce a real PEFT nn.Dropout: {name}")
        roster.append(dict(name=name, layer=layer, target=target, active_adapter="default",
                           lora_A_trainable=True, lora_B_trainable=True))
        modules.append(dropout)
    expected = {(layer, target) for layer in range(36) for target in LORA_TARGETS}
    if seen != expected or len(modules) != LORA_TARGET_COUNT or len({id(m) for m in modules}) != LORA_TARGET_COUNT:
        raise ValueError("formal RL requires exactly 36 x 7 = 252 unique LoRA dropout targets")
    return contract, sorted(roster, key=lambda r: r["name"]), modules


def _dropouts(model):
    import torch
    return [dict(name=n, p=float(m.p), training=bool(m.training)) for n, m in model.named_modules()
            if isinstance(m, torch.nn.modules.dropout._DropoutNd)]


def _functional_dropout(model):
    # Attention implementations may use F.dropout instead of a registered
    # module. Inspect config and live attention attributes without zeroing them.
    fields = ("attention_dropout", "hidden_dropout", "activation_dropout", "classifier_dropout",
              "embd_pdrop", "attn_pdrop", "resid_pdrop")
    records = []
    seen = set()
    for name, module in model.named_modules():
        for suffix, obj in (("", module), (".config", getattr(module, "config", None)),
                            (".config.text_config", getattr(getattr(module, "config", None), "text_config", None))):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            for field in fields:
                value = getattr(obj, field, None)
                if value is not None:
                    records.append(dict(name=name + suffix + "." + field, p=float(value)))
    return sorted(records, key=lambda r: r["name"])


def audit_rl_lora_dropout_runtime(model, sft_config):
    contract, roster, _ = _source_roster(model, sft_config)
    dropouts = _dropouts(model)
    functional = _functional_dropout(model)
    stable = dict(rl_policy_execution_contract=contract, source_adapter_lora_dropout=.05,
        runtime_effective_lora_dropout=0., lora_dropout_target_count=len(roster),
        lora_target_roster=roster,
        dropout_roster=[{k: r[k] for k in ("name", "p")} for r in dropouts],
        functional_dropout_roster=functional)
    return dict(**stable, rl_lora_dropout_runtime_version=RL_LORA_DROPOUT_RUNTIME_VERSION,
        lora_dropout_runtime_sha256=canonical_json_sha256(stable),
        model_training=bool(model.training), dropout_modules=dropouts,
        lora_dropout_p_values=sorted({r["p"] for r in dropouts if r["name"].endswith(".lora_dropout.default")}),
        dropout_training_true_count=sum(r["training"] for r in dropouts if r["name"].endswith(".lora_dropout.default")),
        nonzero_dropout_count=sum(r["p"] != 0. for r in dropouts + functional))


def require_runtime_audit(audit, contract):
    """CPU finalization rederives the stable signature and complete target roster."""
    contract_sft_config(contract)
    keys = ("rl_policy_execution_contract", "source_adapter_lora_dropout", "runtime_effective_lora_dropout",
            "lora_dropout_target_count", "lora_target_roster", "dropout_roster", "functional_dropout_roster")
    stable = {key: audit.get(key) for key in keys}
    if (stable["rl_policy_execution_contract"] != contract
            or audit.get("rl_lora_dropout_runtime_version") != RL_LORA_DROPOUT_RUNTIME_VERSION
            or stable["source_adapter_lora_dropout"] != .05
            or stable["runtime_effective_lora_dropout"] != 0.
            or stable["lora_dropout_target_count"] != LORA_TARGET_COUNT
            or canonical_json_sha256(stable) != audit.get("lora_dropout_runtime_sha256")):
        raise ValueError("missing/changed RL dropout runtime evidence")
    roster = stable["lora_target_roster"]
    expected = {(layer, target) for layer in range(36) for target in LORA_TARGETS}
    if not isinstance(roster, list) or len(roster) != LORA_TARGET_COUNT:
        raise ValueError("incomplete RL dropout target evidence")
    actual = set()
    names = set()
    for row in roster:
        match = _TARGET.search(row.get("name", ""))
        if (not match or int(match[1]) != row.get("layer") or match[3] != row.get("target")
                or match[2] != ("self_attn" if match[3] in LORA_TARGETS[:4] else "mlp")
                or row.get("active_adapter") != "default" or row.get("lora_A_trainable") is not True
                or row.get("lora_B_trainable") is not True):
            raise ValueError("invalid RL dropout target evidence")
        actual.add((row["layer"], row["target"]))
        names.add(row["name"] + ".lora_dropout.default")
    rows = audit.get("dropout_modules", [])
    if (actual != expected or len(names) != LORA_TARGET_COUNT
            or stable["dropout_roster"] != [{k: r[k] for k in ("name", "p")} for r in rows]
            or not names <= {r["name"] for r in rows}
            or len({r["name"] for r in rows}) != len(rows)
            or any(type(r.get("training")) is not bool for r in rows)
            or any(r["p"] != 0. for r in rows + stable["functional_dropout_roster"])
            or audit.get("lora_dropout_p_values") != [0.]
            or audit.get("dropout_training_true_count") != sum(r["training"] for r in rows if r["name"] in names)
            or audit.get("nonzero_dropout_count") != 0):
        raise ValueError("nonzero/invalid policy dropout evidence")
    return audit


def require_rl_lora_dropout_runtime(model, sft_config):
    return require_runtime_audit(audit_rl_lora_dropout_runtime(model, sft_config), execution_contract(sft_config))


def configure_rl_lora_dropout_runtime(model, sft_config):
    contract, _, modules = _source_roster(model, sft_config)
    ids = {id(m) for m in modules}
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            if id(module) in ids:
                if module.p not in (.05, 0.):
                    raise ValueError("unexpected initial LoRA dropout p")
            elif module.p != 0.:
                raise ValueError("unknown non-LoRA policy dropout; never auto-zero it")
    if any(row["p"] != 0. for row in _functional_dropout(model)):
        raise ValueError("nonzero functional policy dropout; never auto-zero it")
    for module in modules:
        module.p = RL_LORA_DROPOUT_EFFECTIVE_P  # preserve module/config/weights/modes
    audit = require_rl_lora_dropout_runtime(model, sft_config)
    if audit["rl_policy_execution_contract"] != contract:
        raise ValueError("RL dropout contract changed during setup")
    return audit


def require_saved_source_dropout(adapter):
    value = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    if value.get("lora_dropout") != .05:
        raise ValueError("saved/source adapter config must preserve SFT lora_dropout=.05")
    return dict(adapter_config_lora_dropout=.05)


def require_execution_binding(identity, group):
    contract = identity.get("rl_policy_execution_contract")
    contract_sft_config(contract)
    fp = effective_policy_fingerprint(identity["source_sft_actor"]["source_sft_adapter_fingerprint"], contract)
    if (identity.get("effective_pre_update_policy_fingerprint") != fp
            or identity.get("identity_sha256") != canonical_json_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})
            or group.get("rl_policy_execution_contract") != contract
            or group.get("effective_pre_update_policy_fingerprint") != fp
            or group["identity"].get("pre_update_policy_fingerprint") != fp
            or group["identity"]["context"] != identity["identity_sha256"]):
        raise ValueError("collection/actor effective policy dropout execution contract mismatch")
    return contract, fp


def require_update_forward_audit(evidence, contract, signature):
    rows = evidence.get("forwards", [])
    if (not rows or evidence.get("forward_count") != len(rows)
            or evidence.get("train_mode_forward_seen") is not True
            or evidence.get("nonzero_dropout_count") != 0
            or evidence.get("lora_dropout_p_values") != [0.]
            or evidence.get("dropout_training_true_count_per_forward") != [252] * len(rows)):
        raise ValueError("missing/invalid actual train-mode RL forward evidence")
    for audit in rows:
        require_runtime_audit(audit, contract)
        if (audit["lora_dropout_runtime_sha256"] != signature or audit["model_training"] is not True
                or audit.get("grad_enabled") is not True
                or not all(r["training"] for r in audit["dropout_modules"]
                           if r["name"].endswith(".lora_dropout.default"))):
            raise ValueError("actual RL forward must train with gradient-enabled p0 LoRA")
    if evidence.get("audit_sha256") != canonical_json_sha256({k: v for k, v in evidence.items() if k != "audit_sha256"}):
        raise ValueError("RL update forward audit checksum mismatch")
    return evidence
