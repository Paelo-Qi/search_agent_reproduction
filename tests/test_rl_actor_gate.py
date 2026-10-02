"""Gate A2.2 contract tests: CPU only, no verl/model download/API calls."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from opensearch_vl_repro.inference.adapter import adapter_file_identity
from opensearch_vl_repro.rl.actor_gate import (
    BASE_MODEL, BASE_REVISION, REQUIRED_CHECKS, aggregate_reports, atomic_json,
    finite_loss, gate_identity, gradient_checks, load_gate_config, lora_snapshot,
    reload_matches, select_rank_sample, temporary_sft_record, trainable_policy,
    update_checks, validate_output_paths, validate_software, load_smoke_records,
)
from opensearch_vl_repro.rl.checkpoint import build_rl_lineage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    return load_gate_config(ROOT / "configs/rl_gate_a22.yaml")


@pytest.mark.parametrize("name,value", [("dtype", "float16"), ("microbatch", 2),
                                         ("microbatch", True), ("fsdp_mode", "fsdp1"),
                                         ("gradient_checkpointing", False)])
def test_invalid_gate_contract(tmp_path, gate, name, value):
    gate[name] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(gate), encoding="utf-8")
    with pytest.raises(ValueError, match=name):
        load_gate_config(path)


def test_gate_software_fail_closed(gate):
    versions = {"torch": "2.8.0+cu128", "transformers": "4.57.1", "peft": "0.21.1", "verl": "0.6.1"}
    validate_software(versions, gate)
    for name in versions:
        wrong = {**versions, name: "99.0"}
        with pytest.raises(ValueError):
            validate_software(wrong, gate)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_rank_sample_independent_of_rollout_and_topology(world):
    records = [{"source_sample_id": f"rl_{i:06d}"} for i in range(20)]
    selected = [select_rank_sample(records, rank=r, world_size=world) for r in range(world)]
    assert selected == records[:world]
    assert all(a is b for a, b in zip(selected, records))
    with pytest.raises(ValueError):
        select_rank_sample(records, rank=0, world_size=world, max_samples=world - 1)
    with pytest.raises(ValueError):
        select_rank_sample(records[:1], rank=0, world_size=world)


def test_world_one_is_not_gate_a21():
    with pytest.raises(ValueError):
        select_rank_sample([{}], rank=0, world_size=1)


def test_missing_adapter_uses_existing_lineage_validator(tmp_path):
    sft = yaml.safe_load((ROOT / "configs/sft_main_imageid_v3.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/rl_main.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="formal SFT adapter"):
        build_rl_lineage(config=config, sft_config=sft, adapter_path=tmp_path / "missing", run_id="gate")


def test_output_refuses_overwrite_and_protected_inputs(tmp_path):
    existing = tmp_path / "ws2"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        validate_output_paths(existing, tmp_path / "r2", [])
    with pytest.raises(FileExistsError):
        validate_output_paths(tmp_path / "ws4", existing, [])
    with pytest.raises(ValueError, match="protected"):
        validate_output_paths(tmp_path / "data/new", tmp_path / "r", [tmp_path / "data"])
    with pytest.raises(ValueError, match="separate"):
        validate_output_paths(tmp_path / "new", tmp_path / "new/report", [])
    validate_output_paths(tmp_path / "outputs/ws4", tmp_path / "reports/ws4", [existing])
    assert not (tmp_path / "outputs").exists()


def identity(gate, world=2, adapter_fingerprint="a" * 64):
    sft = yaml.safe_load((ROOT / "configs/sft_main_imageid_v3.yaml").read_text(encoding="utf-8"))
    return gate_identity(gate=gate, sft=sft,
                         lineage={"sft_adapter_fingerprint": adapter_fingerprint,
                                  "sft_lineage": ["main_a_1k", "main_b_2k"]},
                         data_manifest={"manifest_sha256": "d" * 64}, sample_ids=["rl_000001"],
                         world_size=world, versions={"verl": "0.6.1"}, seed=20260506)


def test_identity_binds_logical_inputs_worldsize_not_snapshot(gate):
    a, b = identity(gate), identity(gate, world=4)
    assert a["base_model"] == BASE_MODEL and a["base_revision"] == BASE_REVISION
    assert a["world_size"] == 2 and b["world_size"] == 4
    assert a["gate_identity_sha256"] != b["gate_identity_sha256"]
    assert a["gate_identity_sha256"] != identity(gate, adapter_fingerprint="b" * 64)["gate_identity_sha256"]
    # Runtime locators live outside identity (different machines, same logical input).
    records = [{"identity": a, "runtime_locators": {"base_snapshot": p}}
               for p in ("/root/autodl-tmp/cache/snapshot", "D:/models/snapshot")]
    assert records[0]["identity"] == records[1]["identity"]
    assert "snapshot" not in json.dumps(a) and "source_root" not in a


def test_adapter_file_fingerprint_preserves_existing_algorithm(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fake test weights")
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
             for name in ("adapter_config.json", "adapter_model.safetensors")}
    result = adapter_file_identity(tmp_path)
    assert result["file_sha256"] == files
    assert result["adapter_fingerprint"] == hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (tmp_path / "adapter_model.safetensors").write_bytes(b"modified")
    assert adapter_file_identity(tmp_path) != result


def mock_lora_model():
    import torch
    model = torch.nn.Module()
    model.visual = torch.nn.Module()
    model.visual.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.visual.merger = torch.nn.Linear(2, 2)
    model.visual.merger.requires_grad_(False)
    model.base_weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.lora_B = torch.nn.Parameter(torch.ones(2))
    return model


def test_trainable_policy_optimizer_only_lora():
    import torch
    model = mock_lora_model()
    optimizer = torch.optim.AdamW([model.lora_B], lr=1e-6)
    audit = trainable_policy(model, optimizer)
    assert audit["trainable_parameter_names"] == ["lora_B"]
    bad = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="optimizer"):
        trainable_policy(model, bad)
    model.visual.merger.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="unexpected trainable"):
        trainable_policy(model)


def test_real_cpu_gradient_update_reload_checks():
    import torch
    model = mock_lora_model()
    before = lora_snapshot(model)
    assert gradient_checks(model) == {"lora_grad_finite": False, "nonzero_lora_grad": False}
    (model.lora_B.square().sum()).backward()
    assert all(gradient_checks(model).values())
    optimizer = torch.optim.AdamW([model.lora_B], lr=1e-6, weight_decay=0)
    optimizer.step()
    after = lora_snapshot(model)
    assert all(update_checks(before, after).values())
    assert reload_matches(after, {k: v.clone() for k, v in after.items()})
    assert not reload_matches(after, before)
    assert not update_checks(after, after)["lora_param_changed"]
    model.lora_B.grad.fill_(float("nan"))
    assert not gradient_checks(model)["lora_grad_finite"]
    model.lora_B.grad.zero_()
    assert not gradient_checks(model)["nonzero_lora_grad"]
    after["lora_B"].fill_(float("inf"))
    assert not update_checks(before, after)["parameters_finite"]


def test_loss_does_not_need_decrease():
    assert finite_loss(10) and finite_loss(20)
    assert not finite_loss(float("nan")) and not finite_loss(float("inf"))


@pytest.mark.parametrize("world", [2, 4])
def test_pass_requires_every_check_every_rank(world):
    rows = [{"rank": r, "checks": dict.fromkeys(REQUIRED_CHECKS, True),
             "peak_allocated_bytes": r + 10, "peak_reserved_bytes": r + 20}
            for r in range(world)]
    assert aggregate_reports(rows, world)["passed"] is True
    assert aggregate_reports(rows, world)["peak_reserved_bytes"] == world - 1 + 20
    assert not aggregate_reports(rows[:-1], world)["passed"]
    for name in REQUIRED_CHECKS:
        failed = copy.deepcopy(rows)
        failed[-1]["checks"][name] = False
        assert not aggregate_reports(failed, world)["passed"], name
    rows[0]["checks"]["loss_finite"] = 1  # not a literal bool success
    assert not aggregate_reports(rows, world)["passed"]


def test_atomic_json_failure_preserves_previous_false_report(tmp_path):
    report = tmp_path / "report.json"
    atomic_json(report, {"passed": False, "stage": "forward"})
    with pytest.raises(ValueError):
        atomic_json(report, {"passed": True, "loss": float("nan")})
    assert json.loads(report.read_text())["passed"] is False
    assert list(tmp_path.iterdir()) == [report]


def test_temporary_batch_record_real_images_and_no_source_mutation(tmp_path):
    from PIL import Image
    from opensearch_vl_repro.agent.reliability import image_sha256
    from opensearch_vl_repro.rl.data import question_sha256
    Image.new("RGB", (28, 28)).save(tmp_path / "one.png")
    row = {"source_sample_id": "rl_000123", "prompt_id": "rl_000123",
           "question": "What color?", "reference_answer": "Black", "image_relpaths": ["one.png"],
           "question_hash": question_sha256("What color?"), "image_hashes": [image_sha256(tmp_path / "one.png")]}
    previous = copy.deepcopy(row)
    record = temporary_sft_record(row, tmp_path)
    assert record["conversations"] == [{"from": "human", "value": "<image>\nWhat color?"},
                                        {"from": "gpt", "value": "Black"}]
    assert Path(record["images"][0]).is_file()
    assert row == previous
    for changed in ({"trajectory_group_id": "bad"}, {"prompt_id": "other"},
                    {"image_relpaths": ["../escape.png"]}):
        with pytest.raises((ValueError, FileNotFoundError)):
            temporary_sft_record({**row, **changed}, tmp_path)
    with pytest.raises(ValueError, match="hashes"):
        temporary_sft_record({**row, "image_hashes": ["0" * 64]}, tmp_path)


def test_cli_help_and_module_import_without_heavy_dependencies():
    code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','peft','verl','ray','vllm'}:
            raise AssertionError('heavy top-level import: ' + fullname)
sys.meta_path.insert(0, Block())
from opensearch_vl_repro.rl import actor_gate, verl_actor_gate
import runpy
sys.argv = ['validate_rl_actor_fsdp.py', '--help']
runpy.run_path('scripts/validate_rl_actor_fsdp.py', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--base-model-path" in result.stdout and "--world-size" not in result.stdout


def test_cli_defaults_and_static_scope():
    spec = importlib.util.spec_from_file_location("gate_script", ROOT / "scripts/validate_rl_actor_fsdp.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = script.build_parser().parse_args(["--source-root", "source", "--output-dir", "ws4", "--report-dir", "r4"])
    assert args.config.name == "rl_main.yaml" and args.data.name == "smoke20.json"
    assert args.max_samples is None and args.base_revision == BASE_REVISION
    source = (ROOT / "src/opensearch_vl_repro/rl/verl_actor_gate.py").read_text(encoding="utf-8")
    assert "from verl.workers.actor.dp_actor import DataParallelPPOActor" in source
    assert "apply_fsdp2(model" in source and "manager.save_checkpoint" in source
    assert "get_fsdp_full_state_dict" in source and '"extra_state"' in source
    assert "world == 2" not in source and "world_size == 2" not in source
    assert "rollout_n=" not in source and "update_policy(" not in source
    assert "lr_scheduler=None" in source and "scheduler.step" not in source
    assert "clip_grad_norm" not in source and "merge_and_unload(" not in source
    assert source.count('stages.run("optimizer_step"') == 1
    assert "HF_HUB_OFFLINE" in source and "TRANSFORMERS_OFFLINE" in source


def test_quality_clean_smoke_manifest_binding_no_mutation(tmp_path):
    from test_rl_formal_data import plan
    from opensearch_vl_repro.rl.data import prepare_formal_dataset
    from opensearch_vl_repro.eval_subset import canonical_json_sha256

    kwargs, *_ = plan(tmp_path)
    prepared = prepare_formal_dataset(**kwargs)
    data = tmp_path / "smoke2.json"
    manifest = data.with_name("smoke2_manifest.json")
    data.write_text(json.dumps(prepared["smoke"]), encoding="utf-8")
    manifest.write_text(json.dumps(prepared["smoke_manifest"]), encoding="utf-8")
    config = {"data": {key: kwargs[key] for key in (
        "quality_audit_dir", "main_count", "smoke_count", "dataset_id", "dataset_revision", "seed")}}
    config["data"]["selection_version"] = prepared["smoke_manifest"]["selection_version"]
    original = {p: p.read_bytes() for p in (data, manifest)}
    records, bound = load_smoke_records(data, config)
    assert records == prepared["smoke"] and bound == prepared["smoke_manifest"]
    assert all(path.read_bytes() == raw for path, raw in original.items())
    # Even a self-consistent manifest cannot substitute a non-ok candidate.
    records[0]["source_sample_id"] = "rl_999999"
    bad = copy.deepcopy(bound)
    bad["membership"][0] = "rl_999999"
    bad["samples_sha256"] = canonical_json_sha256(records)
    bad["manifest_sha256"] = canonical_json_sha256({k: v for k, v in bad.items() if k != "manifest_sha256"})
    data.write_text(json.dumps(records), encoding="utf-8")
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="quality-ok"):
        load_smoke_records(data, config)


def test_collective_stage_failure_persists_false_and_rank_stage(tmp_path):
    from types import SimpleNamespace
    from opensearch_vl_repro.rl.verl_actor_gate import CollectiveStages

    dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 2,
                           all_gather_object=lambda rows, local: rows.__setitem__(slice(None), [
                               local, {**local, "rank": 1, "error": {"rank": 1, "stage": "backward", "message": "OOM"}}]))
    cuda = SimpleNamespace(synchronize=lambda _: None,
                           memory_allocated=lambda _: 1, memory_reserved=lambda _: 2,
                           max_memory_allocated=lambda _: 3, max_memory_reserved=lambda _: 4)
    report = tmp_path / "gate.json"
    stages = CollectiveStages(SimpleNamespace(distributed=dist, cuda=cuda), report, {"world_size": 2}, 0)
    with pytest.raises(RuntimeError, match="backward"):
        stages.run("backward", lambda: None)
    result = json.loads(report.read_text())
    assert result["passed"] is False and result["errors"][0]["rank"] == 1
    assert result["stage"] == "backward"
    assert result["completed_stages"][0]["per_rank"][0]["peak_allocated_bytes"] == 3


def test_gate_sample_uses_real_shared_collator_and_assistant_only_mask(tmp_path):
    import torch
    from PIL import Image
    from test_sft_message_roles import _Processor
    from opensearch_vl_repro.data import OpenSearchVLCollator
    from opensearch_vl_repro.agent.reliability import image_sha256
    from opensearch_vl_repro.rl.data import question_sha256

    class CPUProcessor(_Processor):
        def __call__(self, **kwargs):
            assert isinstance(kwargs["images"][0][0], Image.Image)
            values = super().__call__(**kwargs)["input_ids"][0].tolist()
            return {"input_ids": torch.tensor([values]),
                    "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
                    "pixel_values": torch.zeros((1, 3)), "image_grid_thw": torch.tensor([[1, 1, 1]])}

    Image.new("RGB", (28, 28)).save(tmp_path / "image.png")
    row = {"source_sample_id": "rl_000001", "prompt_id": "rl_000001", "question": "What color?",
           "reference_answer": "Black", "image_relpaths": ["image.png"],
           "question_hash": question_sha256("What color?"), "image_hashes": [image_sha256(tmp_path / "image.png")]}
    processor = CPUProcessor()
    batch = OpenSearchVLCollator(processor, tmp_path / "smoke.json", 32000)([temporary_sft_record(row, tmp_path)])
    supervised = batch["labels"][0][batch["labels"][0] != -100].tolist()
    assert processor.tokenizer.decode(supervised) == "Black<|im_end|>"
    assert batch["input_ids"].shape == batch["labels"].shape
    assert "pixel_values" in batch and "image_grid_thw" in batch
