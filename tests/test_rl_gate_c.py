"""Gate C CPU contracts; mocks below are NOT a real RL/GPU PASS."""
import asyncio
import copy
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry, ToolResult
from opensearch_vl_repro.rl import gate_c
from opensearch_vl_repro.rl.actor_gate import atomic_json
from opensearch_vl_repro.rl.gate_c import (
    ALL_CHECKS, GATE_C_VERSION, bind_run, failure_report, final_passed, load_gate_c_config,
    model_task, paths_for, publish_pass, validate_gate_b,
)
from opensearch_vl_repro.rl.live_workflow import LiveRLWorkflowAdapter, ProviderInterruption
from opensearch_vl_repro.rl.rollout_gate import GATE_B_CHECKS, create_gate_tool_registry
from opensearch_vl_repro.rl.verl_policy_update import audited_policy_update, configure_one_update
from opensearch_vl_repro.rl.workflow_adapter import build_rllm_workflow
from test_rl_group import fixture_group

ROOT = Path(__file__).resolve().parents[1]


def tool(name="crop", **args):
    if name == "crop" and not args:
        args = dict(image="img_1", x=0, y=0, width=4, height=3)
    return '<tool_call>' + json.dumps(dict(name=name, arguments=args)) + '</tool_call>'


def new_adapter(registry=None):
    value = LiveRLWorkflowAdapter(registry or create_gate_tool_registry())
    value.initialize_episode(question="real question", images=[Image.new("RGB", (8, 6), "red")], sample_id="s")
    return value


def apply(adapter, text):
    adapter.apply_tool_calls(adapter.handle_model_output(text))


def test_gate_config_and_formal_main_untouched():
    import yaml
    gate = load_gate_c_config(ROOT / "configs/rl_gate_c.yaml")
    assert gate["rollout_n"] == 2 and gate["actor_world_size"] == 2 and gate["optimizer_steps"] == 1
    assert gate["learning_rate"] == 1e-6 and gate["clip_ratio_high"] == .28
    assert gate["vllm"]["temperature"] == .7 and gate["agent"]["max_turns"] == 16
    formal = yaml.safe_load((ROOT / "configs/rl_main.yaml").read_text(encoding="utf-8"))
    assert formal["rollout_n"] == 4 and formal["algorithm"]["fatal_consecutive_errors"] == 3
    assert formal["model"]["sft_adapter"] == "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    source = inspect.getsource(gate_c.prepare_context)
    assert 'formal_path = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"' in source
    assert 'gate_manifest=None' in source and 'gate_manifest=args.gate_b_manifest' not in source


@pytest.mark.parametrize("initializer", ["outputs/rl_gate_a22/temporary/adapter", "outputs/rl_gate_b/run/merged_model", "outputs/rl_gate_c/run/updated_actor/adapter"])
def test_a22_gate_b_gate_c_initializers_rejected_before_model_load(tmp_path, monkeypatch, initializer):
    from opensearch_vl_repro.rl import config
    from opensearch_vl_repro import sft_train_plan
    monkeypatch.setattr(config, "load_rl_config", lambda *a: {"model":{"sft_config":"ignored", "sft_adapter":initializer}})
    monkeypatch.setattr(sft_train_plan, "load_main_config", lambda *a, **k:{})
    monkeypatch.setattr(gate_c, "validate_actor_adapter", lambda **kw: pytest.fail("initializer must be rejected before loading"))
    with pytest.raises(ValueError, match="NOT Gate A/B/C"):
        gate_c.prepare_context(SimpleNamespace(config=tmp_path, gate_config=ROOT / "configs/rl_gate_c.yaml"), tmp_path)


def test_gate_only_adapter_lacks_formal_metadata(tmp_path):
    from opensearch_vl_repro.inference.adapter import adapter_identity
    adapter = tmp_path / "updated_actor/adapter"; adapter.mkdir(parents=True)
    atomic_json(adapter / "adapter_config.json", {"base_model_name_or_path":gate_c.BASE_MODEL})
    (adapter / "adapter_model.safetensors").write_bytes(b"CPU fixture")
    with pytest.raises(ValueError, match="formal SFT adapter requires"):
        adapter_identity(adapter, base_model=gate_c.BASE_MODEL, base_revision=gate_c.BASE_REVISION)


def test_vllm_capture_requests_processed_logprobs_and_binds_actual_processor(tmp_path, monkeypatch):
    import torch
    from opensearch_vl_repro import model
    from opensearch_vl_repro.rl.rollout_gate import VLLMStaticBackend
    llm_args, calls = [], []
    image = Image.new("RGB", (8, 6))
    class Processor:
        def apply_chat_template(self, messages, **kw): return "actual fixture rendered prompt"
        def __call__(self, **kw):
            assert kw["images"] == [[image]]
            return {"input_ids":torch.tensor([[1,2]]), "pixel_values":torch.ones(2,8), "image_grid_thw":torch.tensor([[1,2,2]])}
    monkeypatch.setattr(model, "load_processor", lambda *a, **k:Processor())
    monkeypatch.setattr(torch.cuda, "device_count", lambda:1)
    class LLM:
        def __init__(self, **kw):
            llm_args.append(kw)
            self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(logprobs_mode=kw["logprobs_mode"])))
        def generate(self, prompts, **kw):
            calls.append(prompts)
            return [SimpleNamespace(prompt_token_ids=[1,2], outputs=[SimpleNamespace(text="answer", token_ids=[3],
                finish_reason="stop", logprobs=[{3:SimpleNamespace(logprob=-.7), 4:SimpleNamespace(logprob=-.1)}])])]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw:kw))
    backend = VLLMStaticBackend(checkpoint=tmp_path, sft_config={"model":{"image_max_pixels":262144}},
        gate=load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"), seed=1, capture_tokens=True)
    backend.sampling = dict(temperature=.7, logprobs=1)
    value = backend.generate(messages=[{"role":"user","content":[{"type":"image","image":image}]}], tools=[])
    assert llm_args[0]["logprobs_mode"] == "processed_logprobs" and value["logprobs"] == [-.7]
    assert value["prompt_ids"] == [1,2] and backend.training_inputs[0]["pixel_values"].shape == (2,8)
    assert calls[0][0]["multi_modal_data"]["image"][0] is image


def test_model_task_never_copies_reference_or_runtime_group_id():
    row = {"source_sample_id": "rl_000001", "prompt_id": "rl_000001", "question": "Q", "reference_answer": "SECRET REFERENCE"}
    assert model_task(row, ["fixture image"]) == {"sample_id": "rl_000001", "question": "Q", "images": ["fixture image"]}
    with pytest.raises(ValueError): model_task({**row, "trajectory_group_id": "forbidden"}, [])


def test_live_fatal_at_third_not_first_no_fourth_call():
    adapter = new_adapter()
    for _ in range(2): apply(adapter, '<tool_call>{broken}</tool_call>')
    assert adapter.state().status == "running"
    apply(adapter, '<tool_call>{broken}</tool_call>')
    assert adapter.fatal_turn == 2 and adapter.fatal_step == 2 and adapter.state().status == "fatal"
    with pytest.raises(RuntimeError, match="terminated"): apply(adapter, "fourth response")
    adapter = new_adapter()
    apply(adapter, '\n'.join(tool("unknown", a=i) for i in range(4)))
    assert len(adapter.state().turns) == 3 and adapter.state().status == "fatal"


def test_success_resets_counter_direct_answer_legal():
    adapter = new_adapter()
    bad = '<tool_call>{broken}</tool_call>'
    apply(adapter, bad); apply(adapter, bad); apply(adapter, tool())
    assert adapter.consecutive_errors == 0
    apply(adapter, bad); apply(adapter, bad)
    assert adapter.fatal_step is None
    apply(adapter, "Final answer.")
    assert adapter.state().status == "success"
    direct = new_adapter(); apply(direct, "Direct answer.")
    assert not direct.state().turns and not direct.fatal_metadata(termination="env_done", step_count=1)["fatal"]


@pytest.mark.parametrize("error", ["quota_error", "authentication_error", "configuration_error", "provider_error", "network_error", "timeout", "invalid_response"])
def test_provider_failures_interrupt_without_increment_or_reward(error):
    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["web_search"],
        lambda a, c: ToolResult("error", "<observation>CPU provider failure</observation>", error)))
    adapter = new_adapter(registry)
    with pytest.raises(ProviderInterruption): apply(adapter, tool("web_search", q="test"))
    assert adapter.consecutive_errors == 0 and adapter.fatal_step is None


def test_pass_requires_every_literal_check_and_gate_only_artifact():
    checks = dict.fromkeys(ALL_CHECKS, True)
    assert final_passed(checks)
    for name in checks:
        assert not final_passed({**checks, name: 1})
        assert not final_passed({**checks, name: False})


@pytest.mark.parametrize("failed_target", ["report", "manifest"])
def test_pass_publication_failures_never_leave_pass_manifest(tmp_path, failed_target):
    output = tmp_path / "out"; output.mkdir()
    report_path = tmp_path / "reports/report.json"
    report = {"checks": dict.fromkeys(ALL_CHECKS, True), "formal_rl_initialization_allowed": False}
    calls = []
    def writer(path, value):
        calls.append(path)
        if (failed_target == "report" and path == report_path) or (failed_target == "manifest" and path.name == "gate_manifest.json"):
            atomic_json(path, value)  # even failure AFTER writing must revoke unsafe marker
            raise OSError("CPU simulated publication failure")
        atomic_json(path, value)
    with pytest.raises(OSError): publish_pass(output, report_path, report, writer)
    assert not (output / "gate_manifest.json").exists()
    assert calls[0] == report_path


def test_final_report_before_manifest_last_and_failure_history(tmp_path):
    output, reports = tmp_path / "out", tmp_path / "reports"
    output.mkdir(); reports.mkdir()
    calls = []
    def writer(path, value): calls.append(path); atomic_json(path, value)
    report = {"checks": dict.fromkeys(ALL_CHECKS, True), "formal_rl_initialization_allowed": False}
    publish_pass(output, reports / "gate_c_report.json", report, writer)
    assert calls == [reports / "gate_c_report.json", output / "gate_manifest.json"]
    for _ in range(2): failure_report(output, reports, {}, "rewarding", ProviderInterruption("quota_error"))
    assert not (output / "gate_manifest.json").exists()
    assert len(list((reports / "failures").glob("*.json"))) == 2
    value = json.loads((reports / "gate_c_report.json").read_text())
    assert value["passed"] is False and value["status"] == "interrupted" and value["interrupt_reason"] == "quota_exhausted"


def test_gate_b_only_evidence_pass_checks_and_source_binding(tmp_path):
    manifest = tmp_path / "gate_b.json"
    value = dict(passed=True, gate_version="actor-rollout-roundtrip-b-v1", formal_rl_initialization_allowed=False,
        base_model=gate_c.BASE_MODEL, base_revision=gate_c.BASE_REVISION,
        source_sft_adapter_fingerprint="a" * 64, runtime_image_protocol_version=gate_c.RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
        checks=dict.fromkeys(GATE_B_CHECKS, True), merged_checkpoint_fingerprint="m" * 64)
    atomic_json(manifest, value)
    assert validate_gate_b(manifest, "a" * 64)["evidence_only_not_rl_initialization"] is True
    for wrong in ({**value, "passed": False}, {**value, "source_sft_adapter_fingerprint": "x"}, {**value, "checks": {}}):
        atomic_json(manifest, wrong)
        with pytest.raises(ValueError): validate_gate_b(manifest, "a" * 64)


def test_resume_identity_changed_fails_closed_and_paths_safe(tmp_path):
    out, rep = paths_for(tmp_path, "one")
    bind_run(out, rep, {"policy": "a"}); bind_run(out, rep, {"policy": "a"})
    with pytest.raises(ValueError, match="changed"): bind_run(out, rep, {"policy": "b"})
    for run in ("../data/rl", "../sft", "/tmp/x", ""):
        with pytest.raises(ValueError): paths_for(tmp_path, run)


def test_configure_one_real_update_no_scheduler():
    @dataclass
    class Config:
        ppo_mini_batch_size: int = 256
        ppo_micro_batch_size_per_gpu: int = 1
        ppo_epochs: int = 2
        shuffle: bool = True
        clip_ratio_low: float = .2
        clip_ratio_high: float = .2
        entropy_coeff: float = 1.
        use_kl_loss: bool = True
        use_rollout_log_probs: bool = False
        loss_agg_mode: str = "token-mean"
    actor = SimpleNamespace(config=Config())
    configure_one_update(actor, 7, load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"))
    assert actor.config.ppo_mini_batch_size == 7 and actor.config.ppo_epochs == 1
    assert actor.config.ppo_micro_batch_size_per_gpu == 1 and actor.config.use_rollout_log_probs
    assert actor.config.clip_ratio_high == .28 and actor.config.entropy_coeff == 0


def toy_actor():
    import torch
    model = torch.nn.Module()
    model.visual = torch.nn.Module(); model.visual.merger = torch.nn.Linear(1, 1)
    model.base_weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    model.q_proj = torch.nn.Module(); model.q_proj.lora_B = torch.nn.Parameter(torch.ones(1))
    for p in model.visual.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.q_proj.lora_B], lr=1e-6)
    return SimpleNamespace(actor_module=model, actor_optimizer=optimizer)


def test_audited_real_step_counter_nonzero_grad_parameter_change_and_frozen():
    import torch
    actor = toy_actor(); before = actor.actor_module.q_proj.lora_B.detach().clone()
    def cpu_update(data):
        # Unit fixture for the AUDIT boundary only, not the formal verl/GPU path.
        actor.actor_module.q_proj.lora_B.sum().backward()
        actor.actor_optimizer.step(); actor.actor_optimizer.zero_grad(set_to_none=True)
        return {"actor/pg_loss": [.1]}
    actor.update_policy = cpu_update
    metrics, audit = audited_policy_update(actor, object())
    assert audit["optimizer_step_count"] == 1 and audit["nonzero_lora_grad"] and audit["vision_projector_base_frozen"]
    assert not torch.equal(before, actor.actor_module.q_proj.lora_B)
    assert all(p.grad is None for p in actor.actor_module.parameters() if not p.requires_grad)


@pytest.mark.parametrize("mode", ["no_step", "second_step", "zero_grad", "nan_loss"])
def test_update_fail_closed_on_missing_duplicate_or_zero_update(mode):
    actor = toy_actor()
    def cpu_update(data):
        if mode == "no_step": return {"actor/pg_loss": [.1]}
        (actor.actor_module.q_proj.lora_B * (0 if mode == "zero_grad" else 1)).sum().backward()
        actor.actor_optimizer.step()
        if mode == "second_step": actor.actor_optimizer.step()
        return {"actor/pg_loss": [float("nan") if mode == "nan_loss" else .1]}
    actor.update_policy = cpu_update
    with pytest.raises(RuntimeError): audited_policy_update(actor, object())


def test_gpu_stack_lazy_import_and_no_supervised_loss_or_formal_loop():
    command = "import sys; import opensearch_vl_repro.rl.gate_c, opensearch_vl_repro.rl.verl_policy_update; assert not any(k in sys.modules for k in ['torch','rllm','verl','vllm']); print('OK')"
    result = subprocess.run([sys.executable, "-c", command], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    from opensearch_vl_repro.rl import verl_policy_update
    source = inspect.getsource(verl_policy_update)
    assert "actor.update_policy(data)" in source and "actor.actor_module(**batch).loss" not in source
    assert "temporary_sft_record" not in source and "scheduler.step" not in source and "labels=" not in source
    assert "range(2)" in inspect.getsource(gate_c.collect) and "AgentRuntime.run" not in inspect.getsource(gate_c)
    assert not (ROOT / "scripts/run_rl_smoke20.py").exists()


def test_collect_interruption_recollects_whole_group_then_durable_resume(tmp_path, monkeypatch):
    """CPU orchestration injection: no real model/provider and no Gate PASS."""
    import torch
    from opensearch_vl_repro.agent import phase3_registry
    from opensearch_vl_repro.rl import reward_judges
    from opensearch_vl_repro.rl.group import read_group
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 0)
    monkeypatch.setattr(phase3_registry, "create_phase3_tool_registry", lambda **kw: create_gate_tool_registry())
    row = {"source_sample_id":"s", "prompt_id":"s", "question":"Q", "reference_answer":"SECRET"}
    args = SimpleNamespace(run_id="cpu-fixture", seed=1, source_root=tmp_path, search_config=tmp_path,
                           layout_config=tmp_path, base_model_path=tmp_path, tool_cache_dir=None)
    ctx = dict(identity={"identity_sha256":"context", "effective_pre_update_policy_fingerprint": "e" * 64,
        "rl_policy_execution_contract": {"CPU collective isolation": True}}, row=row, images=[Image.new("RGB", (8, 6))],
        gate=load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"), sft={}, versions={}, judge=SimpleNamespace(),
        actor={"source_sft_adapter_fingerprint":"a" * 64}, adapter=tmp_path / "formal-checkpoint3k/adapter",
        rl={"paths":{"tool_cache_dir":"outputs/shared_cache"}})
    from _rl_actor_fixture import sft_config
    from opensearch_vl_repro.rl.rl_actor_semantics import execution_contract, effective_policy_fingerprint
    from opensearch_vl_repro.eval_subset import canonical_json_sha256
    contract = execution_contract(sft_config())
    ctx["identity"] = dict(source_sft_actor=ctx["actor"], rl_policy_execution_contract=contract,
        effective_pre_update_policy_fingerprint=effective_policy_fingerprint("a" * 64, contract))
    ctx["identity"]["identity_sha256"] = canonical_json_sha256(ctx["identity"])
    monkeypatch.setattr(gate_c, "prepare_context", lambda *a: ctx)
    merges = []
    def merge(**kw):
        merges.append(kw["adapter"])
        assert "reference_answer" not in json.dumps(kw["validation_messages"], default=str)
        return {"identity":{"merged_checkpoint_fingerprint":"merged"}}
    monkeypatch.setattr(gate_c, "merge_actor_adapter", merge)
    counters = {"generated":0,"closed":0,"reward":0}
    class Backend:
        def __init__(self, **kw): self.training_inputs = []
        def close(self): counters["closed"] += 1
    monkeypatch.setattr(gate_c, "VLLMStaticBackend", Backend)
    module = SimpleNamespace(SamplingParams=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "vllm", module)
    def build(**kw):
        adapter, backend = kw["adapter"], kw["backend"]
        class Workflow:
            async def run_with_termination_handling(self, task, uid):
                assert set(task) == {"sample_id","question","images"} and "SECRET" not in str(task)
                counters["generated"] += 1
                adapter.initialize_episode(**task); adapter.handle_model_output("Answer")
                backend.training_inputs.append({"input_ids":torch.tensor([[1,2]]), "pixel_values":torch.ones(2,8), "image_grid_thw":torch.tensor([[1,2,2]])})
                step = copy.deepcopy(fixture_group()["members"][0]["steps"][0])
                return SimpleNamespace(termination_reason=SimpleNamespace(value="env_done"),
                    trajectories=[SimpleNamespace(steps=[SimpleNamespace(to_dict=lambda:step, info=step["info"])])], id=uid, info={})
        return Workflow(), {"cpu_fixture":True}
    monkeypatch.setattr(gate_c, "build_rllm_workflow", build)
    def rewards(**kw):
        counters["reward"] += 1
        if counters["reward"] == 2: raise ProviderInterruption("quota_error", judge=True)
        return dict(format=1., accuracy=1., query=.5, total=.9)
    monkeypatch.setattr(reward_judges, "live_rewards", rewards)
    from opensearch_vl_repro.evaluation import judge
    monkeypatch.setattr(judge, "DeepSeekJudge", lambda *a: "cpu-fixture-client")
    with pytest.raises(ProviderInterruption): gate_c.collect(args, tmp_path)
    output, reports = paths_for(tmp_path, args.run_id)
    assert not (output / "group").exists() and not (output / "gate_manifest.json").exists()
    failure = json.loads((reports / "gate_c_report.json").read_text())
    first_id = failure["group_identity"]["trajectory_group_id"]
    assert failure["status"] == "interrupted" and len(list((reports / "failures").glob("*.json"))) == 1
    assert counters["generated"] == 2 and counters["closed"] == 1
    assert gate_c.collect(args, tmp_path) == 0
    group = read_group(output / "group")
    assert len(group["members"]) == 2 and group["identity"]["trajectory_group_id"] != first_id
    assert counters["generated"] == 4 and counters["closed"] == 2
    assert gate_c.collect(args, tmp_path) == 0
    assert counters["generated"] == 4 and counters["closed"] == 2 and len(merges) == 2
    assert all(path == ctx["adapter"] for path in merges)
    assert not (output / "gate_manifest.json").exists()  # collect alone is NEVER Gate C PASS
    assert all(m["rllm_trajectory_reward"] == .9 and m["steps"][0]["mc_return"] == .9 for m in group["members"])
    assert all(m["steps"][0]["info"]["reward_computed"] for m in group["members"])


def test_finalize_cross_group_or_missing_actor_evidence_cannot_pass(tmp_path, monkeypatch):
    args = SimpleNamespace(run_id="bad-finalize")
    ctx = {"identity":{"identity_sha256":"context"}}
    monkeypatch.setattr(gate_c, "prepare_context", lambda *a:ctx)
    with pytest.raises(FileNotFoundError): gate_c.finalize(args, tmp_path)
    out, reports = paths_for(tmp_path, args.run_id)
    assert not (out / "gate_manifest.json").exists()
    assert json.loads((reports / "gate_c_report.json").read_text())["passed"] is False


def test_real_installed_rllm_tokens_survive_crop_and_step_if_available(monkeypatch):
    pytest.importorskip("rllm.workflows.multi_turn_workflow", reason="real rLLM not installed locally; AutoDL must NOT skip")
    from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
    monkeypatch.setattr(AgentRuntime, "run", lambda *a, **k: pytest.fail("Eval loop forbidden"))
    adapter = LiveRLWorkflowAdapter(create_gate_tool_registry())
    class Backend:
        def __init__(self): self.index = 0
        def generate(self, **kwargs):
            index = self.index; self.index += 1
            if index: assert isinstance(kwargs["messages"][-1]["content"][0]["image"], Image.Image)
            return dict(text=tool() if index == 0 else "Final red answer.", prompt_ids=[100, index, 102],
                        completion_ids=[200 + index, 201 + index], logprobs=[-.1 - index, -.2 - index],
                        logprobs_mode="processed_logprobs", finish_reason="stop")
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow, _ = build_rllm_workflow(adapter=adapter, backend=Backend(), executor=executor, max_turns=16, capture_tokens=True)
        assert workflow.run.__func__ is MultiTurnWorkflow.run
        episode = asyncio.run(workflow.run_with_termination_handling(task=model_task(
            {"source_sample_id": "s", "prompt_id": "s", "question": "What red detail?", "reference_answer": "SECRET"},
            [Image.new("RGB", (8, 6), "red")]), uid="cpu-rllm-token-regression"))
    assert episode.termination_reason.value == "env_done"
    steps = episode.trajectories[0].steps
    assert len(steps) == 2 and adapter.state().context.image_registry.exists("img_2")
    for index, step in enumerate(steps):
        assert step.prompt_ids == [100, index, 102] and step.response_ids == [200 + index, 201 + index]
        assert step.logprobs == [-.1 - index, -.2 - index] and step.model_output.completion_ids == step.response_ids
        assert step.info["token_origin"] == "vllm.RequestOutput"
    assert "SECRET" not in json.dumps(episode.to_dict())
    json.dumps(episode.to_dict(), allow_nan=False)
