"""Synthetic CPU metadata fixtures only; NEVER proof of a real RL/GPU update."""
import copy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensearch_vl_repro.inference.adapter import adapter_identity
from opensearch_vl_repro.inference.config import load_inference_config
from opensearch_vl_repro.inference.model_loader import load_inference_bundle
from opensearch_vl_repro.evaluation.run_manifest import build_run_manifest, manifest_mismatches
from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl.run_state import new_update_attempt, advance_update_attempt
from opensearch_vl_repro.rl.training_window import build_training_window
from opensearch_vl_repro.sft_tool_audit import sha256_file
from test_rl_formal_contracts import draft_group, fixture_policy, digest
from test_rl_formal_s4_main import main_run
from test_sft_main import (_fake_checkpoint, _FakeModelClass, _FakeProcessorClass,
                          _FakeTorch, _FakePeft, MODEL, REVISION)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle_factory(tmp_path):
    def build(*, cpu=False, kind="main_checkpoint", model=MODEL, revision=REVISION,
              config_base=MODEL, tool_protocol=None, image_protocol=None, readme=False):
        # An arbitrary portable bundle name, not policy-000001. No native files
        # or trajectory files are materialized, and no update is executed.
        checkpoint = tmp_path / f"portable-bundle-{len(list(tmp_path.iterdir()))}"
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": config_base}), encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"INERT CPU BYTES, NOT A REAL MODEL")
        if readme:
            (adapter / "README.md").write_text("declared PEFT README", encoding="utf-8")
        original = main_run()
        semantics = copy.deepcopy(original["semantics"])
        semantics["base_model"].update(name=model, revision=revision)
        if tool_protocol is not None: semantics["tool_protocol_version"] = tool_protocol
        if image_protocol is not None: semantics["image_protocol_version"] = image_protocol
        run = cp.build_training_run_identity("synthetic-metadata-only", semantics=semantics,
            prompt_ids=original["prompt_ids"], prompt_sources=original["prompt_sources"])
        policy = fixture_policy(run)
        scope = "cpu_fixture" if cpu else "runtime"
        groups = []
        for prompt in run["prompt_ids"][:4]:
            group = draft_group(run, policy, prompt)
            files = {name: digest(["inert missing member file", name]) for member in group["members"]
                     for name in (member["trajectory_file"], member["steps"][0]["multimodal_file"])}
            groups.append(cp.seal({**group, "committed": True, "evidence_scope": scope,
                                   "file_sha256": files}, "group_payload_sha256"))
        window = build_training_window(run, policy, groups, window_id="synthetic-window")
        attempt = new_update_attempt(window)
        for phase in ("started", "step_may_have_run", "checkpoint_staging"):
            attempt = advance_update_attempt(attempt, phase)
        rows = []
        for group in groups:
            total = math.fsum(m["reward"]["total"] for m in group["members"])
            for member in group["members"]:
                reward = member["reward"]["total"]
                advantage = reward - (total - reward) / (len(group["members"]) - 1)
                rows.append(dict(group_id=group["identity"]["trajectory_group_id"], member_id=member["member_id"],
                    rollout_index=member["rollout_index"], reward=reward, fatal=False,
                    raw_advantage=advantage, final_advantage=advantage))
        rewards = cp.seal(dict(window_sha256=window["window_sha256"], estimator="official_verl_rloo",
            test_estimator_injected=cpu, status="signal", rows=rows), "reward_window_sha256")
        role_files = {"adapter": cp.artifact_inventory(checkpoint)}
        # Declared checksums ONLY: these files intentionally do not exist.
        for role in ("native", "optimizer", "rng"):
            role_files[role] = {f"distributed/{role}-rank-{rank}.pt": digest([role, rank]) for rank in range(4)}
        inventory = {name: sha for files in role_files.values() for name, sha in files.items()}
        roles = cp.artifact_role_identities(role_files, inventory)
        evidence = dict(scope=scope, reloaded_artifact_roles=roles, adapter_reloaded=True,
            native_reloaded=True, optimizer_reloaded=True, rng_reloaded=True, execution_contract_verified=True)
        manifest = cp.build_checkpoint_manifest(run, policy, groups, window, attempt, rewards,
            artifact_role_files=role_files, file_sha256=inventory, kind=kind, reload_evidence=evidence, cpu_fixture=cpu)
        (checkpoint / "checkpoint.json").write_text(json.dumps(manifest), encoding="utf-8")
        return adapter, manifest
    return build


def identify(adapter, **kwargs):
    return adapter_identity(adapter, base_model=kwargs.get("base_model", MODEL),
                            base_revision=kwargs.get("base_revision", REVISION))


def test_runtime_main_eval_bundle_requires_no_non_adapter_files(bundle_factory):
    adapter, manifest = bundle_factory(readme=True)
    identity = identify(adapter)
    assert identity["kind"] == "peft_lora_adapter" and identity["training_origin"] == "formal_rl_main"
    assert identity["checkpoint_identity"] == manifest["checkpoint_manifest_sha256"]
    assert identity["checkpoint_manifest_fingerprint"] == sha256_file(adapter.parent / "checkpoint.json")
    assert identity["adapter_role_fingerprint"] == manifest["artifact_roles"]["adapter"]
    assert identity["run_identity_sha256"] == manifest["run"]["run_identity_sha256"]
    assert identity["policy_iteration"] == identity["global_optimizer_step"] == 1
    assert all(not (adapter.parent / name).exists() for role in ("native", "optimizer", "rng")
               for name in manifest["artifact_role_files"][role])
    assert set(p.name for p in adapter.parent.iterdir()) == {"adapter", "checkpoint.json"}


def test_sft_identity_remains_exactly_the_existing_shape_and_no_rl_fallback(tmp_path, bundle_factory):
    adapter = _fake_checkpoint(tmp_path / "sft")
    identity = identify(adapter)
    assert set(identity) == {"kind", "path", "base_model", "base_revision", "adapter_fingerprint",
        "adapter_config_fingerprint", "training_cumulative_stage", "source_checkpoint_lineage",
        "checkpoint_metadata_fingerprint", "runtime_tool_protocol_version"}
    assert identity["training_cumulative_stage"] == "main_a_1k"
    rl, _ = bundle_factory()
    (rl.parent / "metadata.json").write_text(json.dumps({"checkpoint_complete": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint base model/revision"):
        identify(rl)


@pytest.mark.parametrize("options", [
    dict(cpu=True), dict(kind="smoke_final"), dict(model="wrong-base"), dict(revision="wrong-revision"),
    dict(config_base="wrong-config-base"), dict(tool_protocol="tool-v2"), dict(image_protocol="image-v2"),
])
def test_rl_scope_kind_base_revision_config_and_protocol_fail_closed(bundle_factory, options):
    adapter, _ = bundle_factory(**options)
    with pytest.raises(ValueError): identify(adapter)


@pytest.mark.parametrize("mutation", ["missing_manifest", "malformed_manifest", "nonobject_manifest",
    "missing_manifest_field", "manifest_identity", "resealed_iteration", "resealed_bool_step",
    "weights_tamper", "config_tamper", "weights_missing", "config_missing", "extra_file", "missing_role_config"])
def test_rl_adapter_inventory_and_manifest_tamper_fail_closed(bundle_factory, mutation):
    adapter, manifest = bundle_factory()
    path = adapter.parent / "checkpoint.json"
    if mutation == "missing_manifest": path.unlink()
    elif mutation == "malformed_manifest": path.write_text("{invalid", encoding="utf-8")
    elif mutation == "nonobject_manifest": path.write_text("[]", encoding="utf-8")
    elif mutation == "weights_tamper": (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    elif mutation == "config_tamper": (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    elif mutation == "weights_missing": (adapter / "adapter_model.safetensors").unlink()
    elif mutation == "config_missing": (adapter / "adapter_config.json").unlink()
    elif mutation == "extra_file": (adapter / "undeclared.txt").write_text("extra", encoding="utf-8")
    else:
        if mutation == "missing_manifest_field": del manifest["run"]
        elif mutation == "manifest_identity": manifest["checkpoint_manifest_sha256"] = "0" * 64
        elif mutation == "resealed_iteration": manifest["policy_iteration"] = 2
        elif mutation == "resealed_bool_step": manifest["global_optimizer_step"] = True
        elif mutation == "missing_role_config": del manifest["artifact_role_files"]["adapter"]["adapter/adapter_config.json"]
        if mutation != "manifest_identity":
            manifest = cp.seal({k: v for k, v in manifest.items() if k != "checkpoint_manifest_sha256"}, "checkpoint_manifest_sha256")
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError): identify(adapter)


def test_manifest_build_records_rl_identity_and_resume_binds_it(bundle_factory, tmp_path):
    adapter, manifest = bundle_factory()
    dataset = tmp_path / "dataset.fixture"
    dataset.write_bytes(b"CPU data, not Eval300")
    common = dict(run_id="cpu-eval", model_name_or_path=MODEL, model_revision=REVISION,
        inference_config_path=ROOT / "configs/eval_base_300.yaml", dataset_path=dataset,
        eval_manifest_path=None, start=0, limit=1, max_agent_turns=16,
        search_config_path=ROOT / "configs/search_backends.example.yaml",
        layout_config_path=ROOT / "configs/layout_parsing.example.yaml")
    run = build_run_manifest(**common, adapter_path=adapter)
    assert run["adapter_identity"]["training_origin"] == "formal_rl_main"
    assert run["adapter_identity"]["checkpoint_identity"] == manifest["checkpoint_manifest_sha256"]
    json.dumps(run)
    assert "adapter_identity" in manifest_mismatches(build_run_manifest(**common), run)


def test_inference_loader_uses_unified_rl_identity_with_only_inert_model_mocks(bundle_factory):
    adapter, _ = bundle_factory()
    config = replace(load_inference_config(ROOT / "configs/eval_base_300.yaml"), device="cpu", adapter_path=adapter)
    bundle = load_inference_bundle(config, model_class=_FakeModelClass, processor_class=_FakeProcessorClass,
        torch_module=_FakeTorch, transformers_module=SimpleNamespace(__version__="fake"), peft_model_class=_FakePeft)
    assert bundle.environment["adapter_identity"]["training_origin"] == "formal_rl_main"
    assert _FakePeft.adapter_path == adapter
    assert not bundle.model.training and not bundle.model.grad_enabled


def test_offline_preflight_uses_only_config_and_adapter_validator(bundle_factory, capsys, monkeypatch):
    adapter, _ = bundle_factory()
    spec = importlib.util.spec_from_file_location("eval_adapter_preflight", ROOT / "scripts/preflight_eval_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Guard against any future accidental CUDA/model/API initialization.
    import builtins
    original_import = builtins.__import__
    def no_framework_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "vllm", "verl", "rllm"}:
            raise AssertionError("preflight must not import model/GPU frameworks")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_framework_import)
    identity = module.main(["--adapter", str(adapter), "--config", str(ROOT / "configs/eval_base_300.yaml")])
    output = capsys.readouterr().out
    assert "adapter_kind=formal_rl_main" in output and "policy_iteration=1" in output
    assert "checkpoint_identity=" + identity["checkpoint_identity"] in output and "PASS" in output
