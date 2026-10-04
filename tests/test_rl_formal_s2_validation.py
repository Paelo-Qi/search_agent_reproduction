"""CPU/AST fixtures only: none of these results is an S2 GPU PASS."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from opensearch_vl_repro.rl import formal_s2_validation as validation
from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl.group import read_formal_group
from opensearch_vl_repro.rl.rloo import assemble_window_rloo
from opensearch_vl_repro.rl.training_batch import formal_training_rows, rank_local_rows
from opensearch_vl_repro.rl.training_window import build_training_window, deterministic_rank_plan
from opensearch_vl_repro.rl.policy_alignment import compare_policy_logprobs, formal_alignment_artifact
from opensearch_vl_repro.rl.old_logprob import ALIGNMENT_META
from opensearch_vl_repro.rl.data import question_sha256
from opensearch_vl_repro.agent.reliability import image_sha256
from test_rl_formal_contracts import fixture_run, cpu_estimator

ROOT = Path(__file__).resolve().parents[1]


def diagnostic_run():
    base = fixture_run(groups_per_window=2, prompt_count=4)
    semantics = copy.deepcopy(base["semantics"])
    semantics["diagnostic_version"] = validation.VERSION
    return cp.build_training_run_identity("s2-cpu-orchestration", semantics=semantics,
        prompt_ids=base["prompt_ids"], prompt_sources=base["prompt_sources"])


def saved_state(step):
    return dict(global_optimizer_step=step, parameter_sha256="a" * 64,
                optimizer_state_sha256="b" * 64, native_rng_sha256="c" * 64)


def receipt(step):
    return dict(**saved_state(step), native_identity_match=True, optimizer_identity_match=True,
                rng_identity_match=True, global_step_match=True)


def oc_evidence(iteration):
    rows = []
    for rank in (0, 1):
        probs = torch.tensor([[-.2, -.3]])
        row = compare_policy_logprobs(probs, probs.clone(), torch.ones_like(probs),
            clip_ratio_low=.2, clip_ratio_high=.28, expected_masked_token_count=2)
        row.update(rank=rank, window_sha256=str(iteration) * 64, logprobs_computed=True,
                   temperature=.7, rollout_temperature=.7, **ALIGNMENT_META)
        rows.append(row)
    return formal_alignment_artifact(rows, world_size=2, window_sha256=str(iteration) * 64)


def rank_evidence(rank):
    windows = []
    for iteration in (0, 1):
        count = 4 if iteration == 0 else 5
        plan = deterministic_rank_plan([f"i{iteration}r{i}" for i in range(count)], 2)
        _, mapping = rank_local_rows([dict(logical_row_id=r) for r in plan["multiplicity"]], plan, rank)
        state = saved_state(iteration + 1)
        windows.append(dict(checkpoint=dict(policy_iteration=iteration + 1, global_optimizer_step=iteration + 1,
                eligibility=dict(eligible_for_main_init=False), checkpoint_manifest_sha256=str(iteration) * 64),
            rank_plan=mapping, deterministic_plan=plan, full_lora_sha256="d" * 64,
            rng_optimizer_save=state,
            actor_contract=dict(passed=True, fsdp2=True, bf16=True, model_training=True,
                language_model_training=True, decoder_layers=36, decoder_layers_training=36,
                decoder_layers_gradient_checkpointing=36, effective_attention_implementation="flash_attention_2",
                dropout=dict(source_adapter_lora_dropout=.05, runtime_effective_lora_dropout=0.)),
            update=dict(before_step=iteration, after_step=iteration + 1, alignment=oc_evidence(iteration),
                rollout_actor_handoff=dict(ratio_finite=True), metrics={"actor/pg_loss": [-.01]},
                update_audit=dict(lora_grad_finite=True, nonzero_lora_grad=True,
                    vision_projector_base_frozen=True, optimizer_step_count=1)),
            residency=dict(passed=True, active_microbatches=[{"pixel_values": {"device": "cuda:0"}}]),
            memory=[dict(phase=p) for p in validation.MEMORY_PHASES],
            reload=dict(original_actor_destroyed=True, fresh_multimodal_forward_finite=True,
                adapter_reloaded=True, native_reloaded=True, optimizer_reloaded=True, rng_reloaded=True,
                execution_contract_verified=True, per_rank=[dict(rank=r, state=state) for r in (0, 1)]),
            continuation=None if iteration == 0 else validation.require_continuation(
                saved_state(1), receipt(1), optimizer_nonempty=True, moment_step=1)))
    return dict(rank=rank, windows=windows, rollout_executed=False, scope="cpu_fixture")


@pytest.mark.parametrize("world", [0, 1, 3, 4, -1])
def test_launcher_rejects_not_two_before_loading(world):
    with pytest.raises(ValueError, match="WORLD_SIZE == 2"):
        validation.require_launcher(dict(WORLD_SIZE=str(world), RANK="0", LOCAL_RANK="0"))


def test_cli_and_launcher_fixed_contract():
    spec = importlib.util.spec_from_file_location("validate_s2_cli", ROOT / "scripts/validate_rl_formal_s2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["--run-id", "cpu-test", "--source-root", "source", "--base-model-path", "base"])
    assert args.config == ROOT / "configs/rl_main.yaml"
    assert args.data == ROOT / "data/rl/smoke20.json"
    assert args.sft_adapter == ROOT / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    assert validation.require_launcher(dict(WORLD_SIZE="2", RANK="1", LOCAL_RANK="1")) == (1, 1)
    assert not hasattr(args, "resume") and not hasattr(args, "windows") and not hasattr(args, "rollout_n")


def test_paths_isolated_protected_and_one_shot(tmp_path):
    output, report = validation.validation_paths(tmp_path, "new", [tmp_path / "base", tmp_path / "source"])
    assert output == tmp_path / "outputs/rl_formal_s2_validation/new"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="new run"):
        validation.validation_paths(tmp_path, "new", [])
    with pytest.raises(ValueError, match="overlap"):
        validation.validation_paths(tmp_path, "other", [tmp_path / "outputs"])
    with pytest.raises(ValueError, match="run-id"):
        validation.validation_paths(tmp_path, "../escape", [])


def test_snapshot_mismatch_fails_closed():
    with pytest.raises(ValueError, match="snapshot SHA mismatch"):
        validation.require_snapshot_sha({"model.safetensors": "f" * 64})
    assert validation.SNAPSHOT_SHA256 == "5c655eb7bd80fb959428f2194acb217424d0e658dfd54ff1eb27f30bcedc236b"


@pytest.mark.parametrize("field", ["parameter_sha256", "optimizer_state_sha256", "native_rng_sha256", "global_optimizer_step"])
def test_continuation_requires_saved_native_optimizer_rng(field):
    wrong = receipt(1)
    wrong[field] = "wrong"
    with pytest.raises(ValueError, match="continuation|step"):
        validation.require_continuation(saved_state(1), wrong, optimizer_nonempty=True, moment_step=1)


@pytest.mark.parametrize("nonempty,step", [(False, 1), (True, 0), (True, 2)])
def test_empty_or_reset_optimizer_not_continuation(nonempty, step):
    with pytest.raises(ValueError, match="nonempty AdamW"):
        validation.require_continuation(saved_state(1), receipt(1), optimizer_nonempty=nonempty, moment_step=step)


def test_continuation_success_and_no_reseed():
    assert validation.require_continuation(saved_state(1), receipt(1), optimizer_nonempty=True, moment_step=1)["passed"]
    source = inspect.getsource(validation.RuntimeSession.load_continuation)
    assert "load_formal_actor" in source and "checkpoint_directory=directory" in source
    assert "initial_seed" not in source and "manual_seed" not in source


def test_aggregation_complete_and_cpu_cannot_pass():
    report = validation.aggregate_report(diagnostic_run(), [rank_evidence(1), rank_evidence(0)], scope="cpu_fixture")
    assert not report["passed"] and not report["checks"]["runtime_scope"]
    assert report["checks"]["window_0_passed"] and report["checks"]["window_1_passed"]
    assert report["optimizer_steps"] == [1, 2]
    assert [r["rank"] for r in report["per_rank"]] == [0, 1]
    assert not report["eligible_for_main_init"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing_rank", "duplicate_rank", "third_window", "missing_window",
    "wrong_step", "eligible", "unequal_full_lora", "replication", "assignment", "saved_rng", "loss_nan"])
def test_aggregation_fail_closed(mutation):
    ranks = [rank_evidence(0), rank_evidence(1)]
    row = ranks[1]["windows"][1]
    if mutation == "missing_rank": ranks.pop()
    elif mutation == "duplicate_rank": ranks[1]["rank"] = 0
    elif mutation == "third_window": ranks[1]["windows"].append(copy.deepcopy(row))
    elif mutation == "missing_window": ranks[1]["windows"].pop()
    elif mutation == "wrong_step": row["update"]["after_step"] = 3
    elif mutation == "eligible": row["checkpoint"]["eligibility"]["eligible_for_main_init"] = True
    elif mutation == "unequal_full_lora": row["full_lora_sha256"] = "e" * 64
    elif mutation == "replication": row["rank_plan"]["replication_factor"] = 1
    elif mutation == "assignment": row["rank_plan"]["rank_assignment"][0][0] = "wrong"
    elif mutation == "saved_rng": row["rng_optimizer_save"] = {**row["rng_optimizer_save"], "native_rng_sha256": "wrong"}
    elif mutation == "loss_nan": row["update"]["metrics"]["actor/pg_loss"] = [float("nan")]
    with pytest.raises(ValueError):
        validation.aggregate_report(diagnostic_run(), ranks, scope="cpu_fixture")


@pytest.mark.parametrize("field", ["alignment", "gradient", "fresh_forward", "residency", "continuation", "memory"])
def test_required_evidence_blocks_pass(field):
    ranks = [rank_evidence(0), rank_evidence(1)]
    row = ranks[1]["windows"][1]
    if field == "alignment": row["update"]["alignment"]["passed"] = False
    elif field == "gradient": row["update"]["update_audit"]["nonzero_lora_grad"] = False
    elif field == "fresh_forward": row["reload"]["fresh_multimodal_forward_finite"] = False
    elif field == "residency": row["residency"]["active_microbatches"] = []
    elif field == "continuation": row["continuation"]["passed"] = False
    else: row["memory"] = []
    report = validation.aggregate_report(diagnostic_run(), ranks, scope="cpu_fixture")
    assert not report["checks"].get("window_1_passed") or not report["checks"]["optimizer_rng_continuation"]
    assert not report["passed"]


@pytest.mark.parametrize("field,value", [("max_abs_logprob_diff", .1), ("max_importance_ratio", 1.3),
    ("masked_token_count", 0), ("initial_clip_fraction", .01)])
def test_oc_summary_rederives_numeric_contract_not_pass_boolean(field, value):
    evidence = oc_evidence(0)
    evidence["per_rank"][1][field] = value
    assert evidence["passed"] is True  # stale/forged summary cannot authorize PASS.
    assert not validation.strict_oc_summary(evidence)


@pytest.mark.parametrize("field,value", [("fsdp2", False), ("bf16", False), ("decoder_layers_training", 35),
    ("decoder_layers_gradient_checkpointing", 35), ("effective_attention_implementation", "sdpa")])
def test_actor_contract_required_in_aggregate(field, value):
    ranks = [rank_evidence(0), rank_evidence(1)]
    ranks[1]["windows"][1]["actor_contract"][field] = value
    result = validation.aggregate_report(diagnostic_run(), ranks, scope="cpu_fixture")
    assert result["checks"]["window_1_passed"] is False


class MockSession:
    """Call-order only; no GPU evidence or optimizer substitute."""
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
    def load_initial(self): self.calls.append("initial")
    def load_continuation(self): self.calls.append("native_checkpoint1")
    def validate_window(self, iteration):
        self.calls.append(("OC", iteration))
        if self.fail == "OC": raise ValueError("strict OC failed before update")
        self.calls.append(("step", iteration + 1))
        if self.fail == "fresh": raise ValueError("fresh reload failed after step")
        self.calls.append(("fresh_reload", iteration + 1))
        self.calls.append(("commit", iteration + 1))
    def finish(self, scope):
        self.calls.append("finish")
        return validation.aggregate_report(diagnostic_run(), [rank_evidence(0), rank_evidence(1)], scope=scope)


def test_exact_two_window_order_checkpoint1_then2():
    session = MockSession()
    report = validation.execute_two_windows(session, cpu_fixture=True)
    assert session.calls == ["initial", ("OC", 0), ("step", 1), ("fresh_reload", 1), ("commit", 1),
        "native_checkpoint1", ("OC", 1), ("step", 2), ("fresh_reload", 2), ("commit", 2), "finish"]
    assert not report["passed"]
    with pytest.raises(ValueError, match="production runtime"):
        validation.execute_two_windows(session)


@pytest.mark.parametrize("failure", ["OC", "fresh"])
def test_orchestration_failure_stops_and_marks_new_run_required(tmp_path, failure):
    output, reports = validation.validation_paths(tmp_path, "cpu-failure", [])
    output.mkdir(parents=True)
    reports.mkdir(parents=True)
    session = MockSession(fail=failure)
    with pytest.raises(ValueError) as error:
        validation.execute_two_windows(session, cpu_fixture=True)
    result = validation.failure_result(output, reports, stage=failure, error=error.value,
                                       evidence=dict(calls=session.calls))
    assert not result["passed"] and not result["resume_allowed"] and result["requires_new_run_id"]
    assert "finish" not in session.calls and "native_checkpoint1" not in session.calls
    if failure == "OC": assert ("step", 1) not in session.calls
    assert not json.loads((output / "manifest.json").read_text())["passed"]
    assert not json.loads((reports / "report.json").read_text())["passed"]
    with pytest.raises(FileExistsError): validation.validation_paths(tmp_path, "cpu-failure", [])


@pytest.mark.parametrize("fail_at", ["report.json", "manifest.json"])
def test_publication_failure_never_leaves_pass_manifest(tmp_path, monkeypatch, fail_at):
    output, reports = validation.validation_paths(tmp_path, "publication", [])
    output.mkdir(parents=True)
    reports.mkdir(parents=True)
    writes, original = [], validation.atomic_json
    # Publication mechanics fixture ONLY, not a claimed GPU report.
    report = dict(passed=True, scope="runtime", checks=dict(publication_fixture=True))
    def fail(path, value):
        writes.append(path.name)
        original(path, value)
        if path.name == fail_at: raise OSError("after replace failure")
    monkeypatch.setattr(validation, "atomic_json", fail)
    with pytest.raises(OSError): validation.publish_result(output, reports, report)
    assert not (output / "manifest.json").exists()
    assert writes[0] == "report.json"


def test_cpu_publication_forbidden(tmp_path):
    with pytest.raises(ValueError, match="cannot publish GPU PASS"):
        validation.publish_result(tmp_path, tmp_path, dict(passed=True, scope="cpu_fixture", checks=dict(fake=True)))


class ProcessorFixture:
    """Explicit CPU encoder fixture; production always uses the real local Qwen processor."""
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [11 if text == "Yes." else 12], eos_token_id=13)
    def __init__(self): self.messages = []
    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        assert kwargs == dict(tools=[], tokenize=False, add_generation_prompt=True)
        return "continuation" if len(messages) == 3 else "initial"
    def __call__(self, *, text, images, **kwargs):
        assert isinstance(images[0][0], Image.Image)
        assert kwargs == dict(return_tensors="pt", truncation=False)
        ids = torch.tensor([[1, 2, 3] if text == ["continuation"] else [1, 2]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.tensor([[.5]]),
            image_grid_thw=torch.tensor([[1, 1, 1]]), mm_token_type_ids=torch.ones_like(ids))


def source_row(tmp_path, index):
    file = tmp_path / f"image-{index}.png"
    Image.new("RGB", (8, 8), "blue").save(file)
    return dict(prompt_id=f"rl_{index:06d}", source_sample_id=f"rl_{index:06d}", question="What color?",
        reference_answer="NEVER PUT THIS REFERENCE IN MODEL INPUT", question_hash=question_sha256("What color?"),
        image_relpaths=[file.name], image_hashes=[image_sha256(file)])


def test_diagnostic_groups_real_source_images_continuation_and_formal_rows(tmp_path):
    records = [source_row(tmp_path, i) for i in range(4)]
    template = diagnostic_run()
    run = cp.build_training_run_identity("diagnostic-cpu", semantics=template["semantics"],
        prompt_ids=[r["prompt_id"] for r in records], prompt_sources=[dict(prompt_id=r["prompt_id"],
        source_identity=validation.source_identity(r)) for r in records])
    policy = cp.initial_policy(run, optimizer_identity="e" * 64, rng_identity="f" * 64)
    processor = ProcessorFixture()
    groups = [validation.diagnostic_group(tmp_path / "formal", run, policy, row, processor, tmp_path,
                continuation=i == 0, cpu_fixture=True) for i, row in enumerate(records[:2])]
    for group in groups:
        directory = tmp_path / "formal/groups" / group["identity"]["trajectory_group_id"]
        assert read_formal_group(directory) == group
        for member in group["members"]:
            assert not member["fatal"] and member["diagnostic_fixture"] and not member["rollout_executed"]
            for step in member["steps"]:
                assert step["info"]["actual_token_origin"] == "deterministic_processor_encoded_completion"
                assert step["logprobs"] == [-1.] * len(step["response_ids"])
                saved = torch.load(directory / step["multimodal_file"], weights_only=True)
                assert saved["input_ids"][0].tolist() == step["prompt_ids"]
    window = build_training_window(run, policy, groups, window_id="cpu-odd-rows")
    reward = assemble_window_rloo(window, run, policy, groups, estimator=cpu_estimator)
    rows, _ = formal_training_rows(window, run, policy, groups, reward)
    assert len(rows) == 5
    assert [r["final_advantage"] for r in reward["rows"]] == pytest.approx([.8, -.8, .8, -.8])
    assert [len(m["steps"]) for m in groups[0]["members"]] == [2, 1]
    plan = deterministic_rank_plan([r["logical_row_id"] for r in rows], 2)
    assert plan["physical_count"] == 10 and plan["replication_factor"] == 2 and plan["local_row_count"] == 5
    assert set(plan["multiplicity"].values()) == {2}
    for messages in processor.messages:
        assert records[0]["reference_answer"] not in str(messages)
    assert not group["evidence_scope"] == "runtime"


def test_source_mutation_rejected_before_group_publication(tmp_path):
    row = source_row(tmp_path, 0)
    Image.new("RGB", (8, 8), "red").save(tmp_path / row["image_relpaths"][0])
    with pytest.raises(ValueError, match="hashes differ"):
        validation.diagnostic_group(tmp_path / "formal", diagnostic_run(), {}, row, ProcessorFixture(), tmp_path, cpu_fixture=True)
    assert not (tmp_path / "formal").exists()


def test_residency_rejects_cpu_active_micro_and_restores_methods():
    modal = np.array([dict(pixel_values=torch.ones(1, 3), image_grid_thw=torch.ones(1, 3))], dtype=object)
    data = SimpleNamespace(non_tensor_batch=dict(multi_modal_inputs=modal))
    original = lambda *args, **kwargs: None
    actor = SimpleNamespace(_forward_micro_batch=original, compute_log_prob=original)
    with pytest.raises(ValueError, match="CUDA multimodal"):
        with validation.observe_residency(actor, data, {}, lambda _: None):
            actor._forward_micro_batch(dict(multi_modal_inputs=modal, responses=torch.ones(1, 1)))
    assert actor._forward_micro_batch is original and actor.compute_log_prob is original


def test_full_fingerprint_materializes_full_shards_not_local_hashes():
    class Shard:
        requires_grad = True
        def __init__(self, full): self.full, self.calls = full, 0
        def detach(self): return self
        def full_tensor(self): self.calls += 1; return self.full
        def to_local(self): pytest.fail("must not compare local shard hashes")
    one, two = Shard(torch.tensor([1., 2.])), Shard(torch.tensor([1., 2.]))
    actors = [SimpleNamespace(actor_module=SimpleNamespace(named_parameters=lambda p=p: [("lora_A", p)])) for p in (one, two)]
    assert validation.full_lora_fingerprint(actors[0]) == validation.full_lora_fingerprint(actors[1])
    assert one.calls == two.calls == 1


def test_static_no_providers_sampling_scheduler_or_schema_changes():
    for filename in (ROOT / "scripts/validate_rl_formal_s2.py", Path(validation.__file__)):
        tree = ast.parse(filename.read_text(encoding="utf-8"))
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        assert not any(any(part in name.lower() for part in ("judge", "deepseek", "vllm", "rllm", "requests", "httpx")) for name in imports)
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert not any(name in calls for name in ("generate", "sample", "request", "get_scheduler"))
    source = inspect.getsource(validation.RuntimeSession.validate_window)
    for primitive in ("build_rank_local_dataproto", "deterministic_rank_plan", "update_formal_window",
                      "save_formal_staging", "fresh_reload_staging", "commit_verified_checkpoint"):
        assert primitive in source
    assert "eligible_for_main_init" not in source  # never reseal eligibility
    assert cp.RL_RUN_SCHEMA_VERSION == cp.RL_CHECKPOINT_SCHEMA_VERSION == 2
    assert cp.checkpoint_eligibility("smoke_continuation")["eligible_for_main_init"] is False
