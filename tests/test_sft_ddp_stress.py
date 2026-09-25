from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_sft_ddp_stress", ROOT / "scripts/diagnose_sft_ddp_stress.py")
assert SPEC is not None and SPEC.loader is not None
stress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stress)


def test_ddp_scan_reuses_untruncated_longest_selection_without_mutating_source(monkeypatch):
    records = [{"_source_index": index, "tools": "[]", "_sample_id": f"sample:{index}"}
               for index in range(4)]
    original = copy.deepcopy(records)
    lengths = [12, 17, 30977, 41]
    observed = []
    monkeypatch.setattr(stress, "processor_token_length",
                        lambda processor, record, path: lengths[record["_source_index"]])
    real_select = stress.select_sample

    def spy_select(*args, **kwargs):
        observed.append((kwargs["sample_index"], kwargs["select_longest"]))
        return real_select(*args, **kwargs)

    monkeypatch.setattr(stress, "select_sample", spy_select)
    index, sample, length = stress.select_longest_record(
        records, object(), Path("main_a_1k.json"), rank=1)
    assert observed == [(None, True)]
    assert (index, sample["_sample_id"], length) == (2, "sample:2", 30977)
    assert len(sample["tools"]) > 0
    assert records == original


def test_adamw_spec_uses_only_trainable_parameters_and_formal_lr_weight_decay():
    config = yaml.safe_load((ROOT / "configs/sft_main.yaml").read_text(encoding="utf-8"))
    frozen = SimpleNamespace(requires_grad=False)
    lora_a = SimpleNamespace(requires_grad=True)
    lora_b = SimpleNamespace(requires_grad=True)
    model = SimpleNamespace(parameters=lambda: iter((frozen, lora_a, lora_b)))
    parameters, options = stress.optimizer_spec(model, config)
    assert parameters == [lora_a, lora_b]
    assert options == {"lr": float(config["scheduler"]["phase_1"]["peak_lr"]),
                       "weight_decay": float(config["training"]["weight_decay"])}
    assert options["lr"] == 0.0002
    assert options["weight_decay"] == 0.0
    with pytest.raises(ValueError, match="no trainable"):
        stress.optimizer_spec(SimpleNamespace(parameters=lambda: iter((frozen,))), config)
    stress.validate_stress_config(config)
    changed = copy.deepcopy(config)
    changed["training"]["per_device_train_batch_size"] = 2
    with pytest.raises(ValueError, match="micro=1"):
        stress.validate_stress_config(changed)


def test_rank_summary_aggregates_peaks_and_rejects_mismatched_samples():
    reports = [
        {"rank": 0, "sample_id": "livevqa:7073", "sample_index": 177,
         "actual_token_length": 30977, "loss": 1.5, "peak_allocated_gib": 76.2,
         "peak_reserved_gib": 77.1, "optimizer_step_completed": True},
        {"rank": 1, "sample_id": "livevqa:7073", "sample_index": 177,
         "actual_token_length": 30977, "loss": 1.6, "peak_allocated_gib": 78.0,
         "peak_reserved_gib": 78.5, "optimizer_step_completed": True},
    ]
    summary = stress.aggregate_summary(reports)
    assert summary["passed"] is True
    assert summary["overall_peak_allocated_gib"] == 78.0
    assert summary["overall_peak_reserved_gib"] == 78.5
    assert summary["per_rank_peak_reserved_gib"] == {"0": 77.1, "1": 78.5}
    assert summary["optimizer_step_completed"] is True
    assert "not_formal_sampler" in summary["stress_semantics"]
    failed = copy.deepcopy(reports)
    failed[1]["optimizer_step_completed"] = False
    assert stress.aggregate_summary(failed)["passed"] is False
    mismatched = copy.deepcopy(reports)
    mismatched[1]["sample_id"] = "other"
    with pytest.raises(ValueError, match="different longest samples"):
        stress.aggregate_summary(mismatched)


def test_oom_event_marks_rank_stage_and_memory():
    memory = {"allocated_gib": 75.5, "reserved_gib": 76.5,
              "max_allocated_gib": 75.5, "max_reserved_gib": 76.5}
    event = stress.oom_event(1, "optimizer_step", memory)
    assert event == {"passed": False, "rank": 1, "oom_stage": "optimizer_step", **memory}
    source = inspect.getsource(stress.main)
    for stage in ("ddp_wrap", "forward", "backward", "optimizer_step"):
        assert f'stage = "{stage}"' in source


def test_ddp_stress_has_one_step_and_no_scheduler_or_checkpoint_writes():
    source = inspect.getsource(stress.main)
    assert "DistributedDataParallel(model" in source
    assert "find_unused_parameters=False" in source
    assert source.count("optimizer.step()") == 1
    assert source.count("optimizer.zero_grad(set_to_none=True)") == 1
    assert "loss.backward()" in source
    assert "no_sync()" not in source.replace("# One micro-batch only; no accumulation or no_sync().", "")
    for forbidden in ("scheduler.step(", "LambdaLR(", "save_pretrained(",
                      "torch.save(", "_save_checkpoint(", "DistributedSampler("):
        assert forbidden not in source
