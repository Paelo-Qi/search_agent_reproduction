"""CPU bridge/collector fixtures; never substitute for actual rLLM/GPU PASS."""
import asyncio
import copy
import inspect
import sys
from enum import Enum
from types import SimpleNamespace

import pytest
from PIL import Image

from opensearch_vl_repro.rl.context_budget import CONTEXT_BUDGET_POLICY, ResponseContextBudgetExhausted
from opensearch_vl_repro.rl import rollout_gate, workflow_adapter, formal_collection
from opensearch_vl_repro.rl.live_workflow import LiveRLWorkflowAdapter
from opensearch_vl_repro.rl.workflow_types import RLInfrastructureError


@pytest.fixture
def backend(monkeypatch):
    import torch
    image = Image.new("RGB", (8, 6))
    class Processor:
        length = 7000
        def apply_chat_template(self, messages, **kw):
            return "FULL UNTRUNCATED CPU PROMPT"
        def __call__(self, **kw):
            assert kw == dict(text=["FULL UNTRUNCATED CPU PROMPT"], images=[[image]],
                              return_tensors="pt", truncation=False)
            return dict(input_ids=torch.arange(self.length).unsqueeze(0), pixel_values=torch.ones(2, 8))
    calls = []
    def generate(prompts, **kw):
        assert prompts[0]["prompt"] == "FULL UNTRUNCATED CPU PROMPT"
        assert prompts[0]["multi_modal_data"]["image"][0] is image
        calls.append(kw["sampling_params"])
        return [SimpleNamespace(prompt_token_ids=list(range(value.processor.length)), outputs=[SimpleNamespace(
            text='<tool_call>{"name":"crop","arguments":{"image":"img_1","x":0,"y":0,"width":4,"height":3}}</tool_call>',
            token_ids=[3], logprobs=[{3: SimpleNamespace(logprob=-.7)}], finish_reason=value.finish_reason)])]
    value = rollout_gate.VLLMStaticBackend.__new__(rollout_gate.VLLMStaticBackend)
    value.processor, value.llm = Processor(), SimpleNamespace(generate=generate)
    value.context_budget_policy = CONTEXT_BUDGET_POLICY
    value.capture_tokens, value.training_inputs, value.receipts = True, [], []
    value.settings = dict(max_model_len=8192, max_new_tokens=512)
    value.sampling = SimpleNamespace(max_tokens=512, temperature=.7, top_p=.9, top_k=20,
                                    logprobs=1, seed=123, stop=["STOP"], extra_args={"fixture": True})
    value.max_images, value.finish_reason = 17, "stop"
    value.messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    value.calls, value.image = calls, image
    return value


@pytest.mark.parametrize("length,effective", [(7000, 512), (7900, 292), (7680, 512), (8191, 1)])
def test_main_budget_real_call_and_exact_captures(backend, length, effective):
    backend.processor.length = length
    original = copy.deepcopy(vars(backend.sampling))
    result = backend.generate(messages=backend.messages, tools=[])
    assert backend.calls[0].max_tokens == effective
    assert backend.calls[0] is not backend.sampling
    assert vars(backend.calls[0]) == {**original, "max_tokens": effective}
    assert vars(backend.sampling) == original
    assert backend.training_inputs[0]["input_ids"][0].tolist() == result["prompt_ids"]
    assert backend.training_inputs[0]["pixel_values"].shape == (2, 8)
    budget = result["context_budget"]
    assert budget == dict(processor_token_length=length, max_model_len=8192,
        requested_max_tokens=512, effective_max_tokens=effective, remaining_context_tokens=8192-length,
        context_budget_policy=CONTEXT_BUDGET_POLICY, context_limited=effective < 512)
    assert all(backend.receipts[0][k] == v for k, v in budget.items())
    backend.processor.length = 7000
    backend.generate(messages=backend.messages, tools=[])
    assert backend.calls[1].max_tokens == 512
    assert backend.sampling.max_tokens == 512


@pytest.mark.parametrize("length", [8192, 8300])
def test_exhausted_call_has_no_phantom_capture_or_generation(backend, length):
    backend.generate(messages=backend.messages, tools=[])
    backend.processor.length = length
    with pytest.raises(ResponseContextBudgetExhausted) as info:
        backend.generate(messages=backend.messages, tools=[])
    assert info.value.diagnostics["remaining_context_tokens"] == 8192-length
    assert len(backend.training_inputs) == len(backend.receipts) == len(backend.calls) == 1


def test_requested_budget_comes_from_current_member_sampling(backend):
    backend.processor.length = 7900
    backend.sampling.max_tokens = 128
    backend.generate(messages=backend.messages, tools=[])
    assert backend.calls[0].max_tokens == 128
    assert backend.receipts[0]["requested_max_tokens"] == 128
    assert backend.receipts[0]["context_limited"] is False


def test_default_gate_s3_backend_keeps_fixed_budget_failure(backend):
    assert inspect.signature(rollout_gate.VLLMStaticBackend).parameters["context_budget_policy"].default is None
    backend.context_budget_policy = None
    backend.processor.length = 7900
    with pytest.raises(RuntimeError, match="no truncation"):
        backend.generate(messages=backend.messages, tools=[])
    assert not backend.calls
    backend.processor.length = 7000
    result = backend.generate(messages=backend.messages, tools=[])
    assert backend.calls[0] is backend.sampling
    assert "context_budget" not in result


def test_unknown_policy_rejected_before_framework_import():
    with pytest.raises(ValueError, match="unsupported"):
        rollout_gate.VLLMStaticBackend(checkpoint=None, sft_config={}, gate={}, seed=1,
                                      context_budget_policy="truncate")


# API shells exercise our real LocalEngine/ProjectAgent, NOT an imitation of
# upstream's turn loop. The optional installed-rLLM test below covers that loop.
class CPUReason(Enum):
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"


class CPUTerminationEvent(Exception):
    def __init__(self, reason): self.reason = reason


class CPUBaseAgent: pass
class CPUBaseEnv: pass
class CPURolloutEngine:
    async def get_model_response(self, messages, **kw): raise AssertionError("abstract shell")
class CPUModelOutput(SimpleNamespace): pass
class CPUAction:
    def __init__(self, action): self.action = action
class CPUStep(SimpleNamespace):
    def to_dict(self):
        return {k: v for k, v in vars(self).items() if k not in {"action", "model_output"}}
class CPUTrajectory:
    def __init__(self, name): self.name, self.steps = name, []
class CPUWorkflow: pass
class CPUMultiTurnWorkflow(CPUWorkflow):
    def __init__(self, agent_cls, env_cls, rollout_engine, **kw):
        self.agent, self.env, self.rollout_engine = agent_cls(), env_cls(), rollout_engine
    async def run(self, **kw): raise AssertionError("CPU shell must not own rollout loop")
    def start_timing(self): pass
    def finalize_timing(self): return {}


@pytest.fixture
def bridge(monkeypatch, backend):
    modules = {
        "rllm.agents.agent": dict(Action=CPUAction, BaseAgent=CPUBaseAgent, Step=CPUStep, Trajectory=CPUTrajectory),
        "rllm.engine.rollout.rollout_engine": dict(ModelOutput=CPUModelOutput, RolloutEngine=CPURolloutEngine),
        "rllm.environments.base.base_env": dict(BaseEnv=CPUBaseEnv),
        "rllm.workflows.multi_turn_workflow": dict(MultiTurnWorkflow=CPUMultiTurnWorkflow),
        "rllm.workflows.workflow": dict(Workflow=CPUWorkflow, TerminationEvent=CPUTerminationEvent,
                                       TerminationReason=CPUReason),
    }
    for name, attributes in modules.items(): monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))
    adapter = LiveRLWorkflowAdapter(rollout_gate.create_gate_tool_registry())
    adapter.initialize_episode(question="CPU budget", images=[backend.image], sample_id="rl_fixture")
    workflow, provenance = workflow_adapter.build_rllm_workflow(
        adapter=adapter, backend=backend, executor=None, max_turns=16, capture_tokens=True)
    workflow.uid, workflow.task = "fixture", dict(sample_id="rl_fixture", question="CPU budget")
    return adapter, workflow, provenance


def generate_step(backend, bridge):
    adapter, workflow, _ = bridge
    output = asyncio.run(workflow.rollout_engine.get_model_response(backend.messages))
    action = workflow.agent.update_from_model(output.text)
    workflow.env.step(action)
    return output


def test_reduced_budget_length_is_returned_to_upstream_unchanged(backend, bridge):
    adapter, workflow, _ = bridge
    backend.processor.length, backend.finish_reason = 7900, "length"
    output = generate_step(backend, bridge)
    assert output.finish_reason == "length" and backend.calls[0].max_tokens == 292
    assert workflow.agent.trajectory.steps[0].info["finish_reason"] == "length"
    assert workflow.agent.trajectory.steps[0].info["context_budget"]["effective_max_tokens"] == 292
    assert adapter.infrastructure_failure is None and adapter.state().error_origin is None


def test_exhausted_after_real_prefix_uses_termination_without_infrastructure_failure(backend, bridge):
    adapter, workflow, _ = bridge
    generate_step(backend, bridge)
    backend.processor.length = 8192
    with pytest.raises(CPUTerminationEvent) as info:
        asyncio.run(workflow.rollout_engine.get_model_response(backend.messages))
    assert info.value.reason.value == "max_response_length_exceeded"
    assert adapter.infrastructure_failure is None
    assert adapter.state().error_origin is None and adapter.state().status == "running"
    assert len(workflow.agent.trajectory.steps) == len(backend.training_inputs) == len(backend.calls) == 1
    assert adapter.state().context_budget_exhaustion["prompt_id"] == "rl_fixture"
    episode = workflow.postprocess_episode(SimpleNamespace(), info.value.reason)
    assert "error" not in episode.info
    assert episode.info["context_budget_exhaustion"]["remaining_context_tokens"] == 0


def test_zero_step_is_not_trainable_and_has_length_diagnostics(backend, bridge):
    adapter, workflow, _ = bridge
    backend.processor.length = 8300
    with pytest.raises(RLInfrastructureError, match="zero-step member cannot enter Formal training") as info:
        asyncio.run(workflow.rollout_engine.get_model_response(backend.messages))
    assert "rl_fixture" in str(info.value) and "8300" in str(info.value) and "-108" in str(info.value)
    assert not workflow.agent.trajectory.steps and not backend.training_inputs and not backend.calls
    assert adapter.infrastructure_failure is None and adapter.state().error_origin is None


def test_formal_collector_preserves_fatal_prefix_and_existing_reward_rules(backend, bridge, tmp_path, monkeypatch):
    import torch
    from opensearch_vl_repro.rl import reward_judges
    from opensearch_vl_repro.rl.reward import clamp_fatal_advantages
    from opensearch_vl_repro.evaluation.judge import JudgeConfig
    adapter, workflow, provenance = bridge
    generate_step(backend, bridge)
    backend.processor.length = 8192
    with pytest.raises(CPUTerminationEvent) as info:
        asyncio.run(workflow.rollout_engine.get_model_response(backend.messages))
    episode = workflow.postprocess_episode(SimpleNamespace(trajectories=[workflow.agent.trajectory]), info.value.reason)
    monkeypatch.setattr(workflow_adapter, "build_rllm_workflow", lambda **kw: (workflow, provenance))
    monkeypatch.setattr("opensearch_vl_repro.rl.live_workflow.LiveRLWorkflowAdapter", lambda registry: adapter)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    # Collector takes its capture start before rollout. Reinsert the real captured
    # prefix when our completed-workflow fixture is invoked (no fake token IDs).
    captures = backend.training_inputs[:]
    backend.training_inputs.clear()
    async def completed(**kw):
        backend.training_inputs.extend(captures)
        return episode
    workflow.run_with_termination_handling = completed
    durable_json = formal_collection.cp.durable_json
    monkeypatch.setattr(formal_collection.cp, "durable_json", lambda path, value:
                        durable_json(path, value, cpu_fixture=True))
    judge = SimpleNamespace(config=JudgeConfig(provider="deepseek", base_url="https://fixture.invalid", model="fixture"))
    def cached_request(client, directory, identity, messages, parser):
        assert identity["trajectory"]["final_answer"] is None
        assert identity["trajectory"]["fatal"]["fatal"] is True
        if identity["kind"] == "accuracy":
            return ("incorrect", "CPU fixture null final answer"), {"status": "success"}
        return SimpleNamespace(score=.5, reason="CPU fixture"), {"status": "success"}
    monkeypatch.setattr(reward_judges, "cached_request", cached_request)
    member = formal_collection.collect_member(row=dict(source_sample_id="rl_fixture", prompt_id="rl_fixture",
        question="CPU budget", reference_answer="NOT IN MODEL PROMPT"),
        images=[backend.image], identity=dict(trajectory_group_id="fixture"), index=0,
        backend=backend, registry=adapter.tool_registry, judge=judge, reward_cache=tmp_path,
        staging=tmp_path, rollout=dict(temperature=.7, top_p=.9, top_k=20, max_new_tokens=512, max_turns=16), seed=123)
    assert member["fatal"] is True and member["fatal_step"] == 0
    assert member["termination"] == "max_response_length_exceeded" and member["complete"] is True
    assert len(member["steps"]) == 1 and member["trajectory"]["final_answer"] is None
    assert member["reward"]["total"] == pytest.approx(.1)
    assert member["reward"]["total"] == member["reward"]["format"] * (
        .8 * member["reward"]["accuracy"] + .2 * member["reward"]["query"])
    assert clamp_fatal_advantages([-1., 1.], [member["fatal"], member["fatal"]]) == [0., 1.]
    assert member["steps"][0]["info"]["context_budget"]["effective_max_tokens"] == 512
    assert member["trajectory"]["metadata"]["context_budget_exhaustion"]["remaining_context_tokens"] == 0
    tensors = torch.load(tmp_path / member["steps"][0]["multimodal_file"], weights_only=True)
    assert tensors["input_ids"][0].tolist() == member["steps"][0]["prompt_ids"]
    assert not backend.training_inputs


@pytest.mark.parametrize("case", ["length", "prefix_exhausted", "initial_exhausted"])
def test_installed_rllm_budget_termination_if_available(case, backend):
    pytest.importorskip("rllm.workflows.multi_turn_workflow", reason="real rLLM is not installed locally")
    from concurrent.futures import ThreadPoolExecutor
    adapter = LiveRLWorkflowAdapter(rollout_gate.create_gate_tool_registry())
    real_generate = backend.generate
    def generate(**kw):
        if case == "prefix_exhausted" and backend.calls: backend.processor.length = 8192
        return real_generate(messages=backend.messages, tools=[])
    backend.generate = generate
    if case == "length":
        backend.processor.length, backend.finish_reason = 7900, "length"
    if case == "initial_exhausted": backend.processor.length = 8192
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow, _ = workflow_adapter.build_rllm_workflow(adapter=adapter, backend=backend,
            executor=executor, max_turns=16, capture_tokens=True)
        episode = asyncio.run(workflow.run_with_termination_handling(
            task=dict(sample_id="rl_fixture", question="CPU budget", images=[backend.image]), uid="fixture"))
    assert adapter.infrastructure_failure is None
    if case == "initial_exhausted":
        assert episode.termination_reason.value == "error"
        assert "zero-step member" in str(episode.info)
        assert not episode.trajectories and not backend.training_inputs and not backend.calls
    else:
        assert episode.termination_reason.value == "max_response_length_exceeded"
        assert "error" not in episode.info
        assert len(episode.trajectories[0].steps) == len(backend.training_inputs) == 1
        assert adapter.fatal_metadata(termination=episode.termination_reason.value, step_count=1)["fatal_step"] == 0
        if case == "length": assert backend.calls[0].max_tokens == 292
