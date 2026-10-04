"""Step-oriented project semantics plus a lazy, real rLLM 0.2.1 binding.

There is no generation loop here: MultiTurnWorkflow.run owns every model turn.
No reward, fatal cascade, group commit or policy update is implemented.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.interaction import AgentInteraction, AgentTrajectory, AgentTurn
from opensearch_vl_repro.agent.question_normalization import normalize_model_question
from opensearch_vl_repro.agent.tool_parser import ParsedAssistantOutput, ToolCallParser
from opensearch_vl_repro.agent.tool_registry import ToolContext, ToolRegistry
from opensearch_vl_repro.agent.reliability import redact_secrets
from opensearch_vl_repro.data import messages_to_json_safe
from .context_budget import ResponseContextBudgetExhausted

RLLM_SOURCE_COMMIT = "c5c02a49780e26ae9cb6f1fb56731d1e594d59f0"


@dataclass
class InteractionEpisode:
    context: ToolContext
    messages: list[dict[str, Any]]
    turns: list[AgentTurn] = field(default_factory=list)
    assistant_outputs: list[str] = field(default_factory=list)
    seen_tool_calls: dict[str, int] = field(default_factory=dict)
    final_answer: str | None = None
    status: str = "running"
    error: str | None = None
    error_origin: str | None = None
    context_budget_exhaustion: dict[str, Any] | None = None


class RLWorkflowAdapter(AgentInteraction):
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.parser = ToolCallParser(tool_registry.list_tools())
        self.episode: InteractionEpisode | None = None

    def initialize_episode(self, *, question: str, images: Sequence[Any], sample_id: str,
                           benchmark: str = "rl-gate") -> InteractionEpisode:
        if not isinstance(question, str) or not question.strip() or not images:
            raise ValueError("episode requires question and initial images")
        registry = ImageRegistry()
        ids = [registry.register_initial_image(image) for image in images]
        self.episode = InteractionEpisode(
            ToolContext(registry, sample_id=sample_id, benchmark=benchmark),
            self._initial_messages(normalize_model_question(question), images, ids))
        return self.episode

    def state(self) -> InteractionEpisode:
        if self.episode is None:
            raise RuntimeError("episode not initialized")
        return self.episode

    def handle_model_output(self, output: str) -> ParsedAssistantOutput:
        state = self.state()
        if state.status != "running":
            raise RuntimeError("cannot append output to a terminated episode")
        parsed = self.parser.parse(output)
        state.assistant_outputs.append(parsed.raw_text)
        if parsed.kind == "final_answer":
            state.final_answer, state.status = parsed.final_answer, "success"
            state.messages.append({"role": "assistant", "content": parsed.raw_text})
        elif parsed.kind == "malformed_tool_call":
            result = self._error_result("invalid_tool_call", parsed.error or "unknown")
            state.turns.append(AgentTurn(parsed.raw_text, None, result.observation, "error",
                                         result.error_type, {"error_origin": "malformed_tool_call"}))
            state.messages.extend([{"role": "assistant", "content": parsed.raw_text},
                                   {"role": "tool", "content": result.observation}])
        else:
            state.messages.append(self._structured_assistant_message(parsed))
        return parsed

    def execute_tool_call(self, call: Any) -> Any:
        state = self.state()
        return self._execute_once(call, state.context, state.seen_tool_calls, len(state.turns))

    def commit_tool_result(self, *, call: Any, result: Any, assistant_output: str,
                           tool_latency_seconds: float | None = None) -> AgentTurn:
        state = self.state()
        turn = self._commit_safely(result=result, call=call, context=state.context,
                                   messages=state.messages, assistant_output=assistant_output,
                                   tool_latency_seconds=tool_latency_seconds)
        if turn.status == "error":
            turn.metadata["error_origin"] = (turn.error if turn.error in {
                "unknown_tool", "unknown_image_id", "duplicate_tool_call"} else "tool_backend_error")
        state.turns.append(turn)
        return turn

    def apply_tool_calls(self, parsed: ParsedAssistantOutput) -> None:
        # This iterates calls from ONE model output, never schedules generation.
        for call in parsed.tool_calls:
            started = time.perf_counter()
            result = self.execute_tool_call(call)
            self.commit_tool_result(call=call, result=result, assistant_output=parsed.raw_text,
                                    tool_latency_seconds=time.perf_counter() - started)

    def build_next_messages(self) -> list[dict[str, Any]]:
        return self.state().messages

    def record_failure(self, *, origin: str, message: str) -> None:
        state = self.state()
        state.status, state.error, state.error_origin = "model_error", message, origin

    def finalize_episode(self, *, termination: str | None = None) -> AgentTrajectory:
        state = self.state()
        if state.status == "running":
            state.status = "max_agent_turns_exceeded" if termination == "max_turns_exceeded" else "workflow_error"
            state.error_origin = "max_turns" if termination == "max_turns_exceeded" else "rllm_workflow_failure"
            state.error = f"workflow terminated: {termination}"
        registry = state.context.image_registry
        return AgentTrajectory(state.context.sample_id or "episode", state.context.benchmark or "rl",
                               state.turns, state.final_answer, state.status,
                               [entry.image_id for entry in registry.list_images()], error=state.error,
                               metadata={"error_origin": state.error_origin,
                                         "model_turn_count": len(state.assistant_outputs),
                                         **({"context_budget_exhaustion": state.context_budget_exhaustion}
                                            if state.context_budget_exhaustion is not None else {})},
                               images=self._image_summaries(registry))


def build_rllm_workflow(*, adapter: RLWorkflowAdapter, backend: Any, executor: Any,
                        max_turns: int, capture_tokens: bool = False) -> tuple[Any, dict[str, Any]]:
    """Instantiate the verified upstream loop; only adapters/postprocess are local."""
    from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
    from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
    from rllm.environments.base.base_env import BaseEnv
    from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
    from rllm.workflows.workflow import Workflow, TerminationEvent, TerminationReason
    from opensearch_vl_repro.sft_tool_audit import sha256_file

    if not inspect.iscoroutinefunction(MultiTurnWorkflow.run) or not inspect.iscoroutinefunction(RolloutEngine.get_model_response):
        raise RuntimeError("unsupported rLLM workflow/engine async API")
    pending: dict[str, Any] = {}

    class ProjectAgent(BaseAgent):
        def __init__(self):
            self.reset()

        def reset(self):
            self._trajectory = Trajectory(name="project_visual_gate")

        @property
        def trajectory(self):
            return self._trajectory

        @property
        def chat_completions(self):
            return adapter.build_next_messages()

        def update_from_model(self, response, **kwargs):
            parsed = adapter.handle_model_output(response)
            action = Action(parsed)
            self._trajectory.steps.append(Step(
                chat_completions=messages_to_json_safe(adapter.build_next_messages()),
                model_response=response, action=Action({"kind": parsed.kind,
                    "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in parsed.tool_calls]})))
            if capture_tokens:
                output = pending.pop("output")
                if output.text != response or not output.logprobs or len(output.logprobs) != len(output.completion_ids):
                    raise RuntimeError("real rLLM Step/output token alignment failed")
                step = self._trajectory.steps[-1]
                step.prompt_ids = list(output.prompt_ids)
                step.response_ids = list(output.completion_ids)
                step.logprobs = list(output.logprobs)
                step.model_output = output
                step.chat_completions = pending.pop("prompt_messages")
                step.info = {"token_count": len(step.response_ids), "parsed_kind": parsed.kind,
                             "finish_reason": output.finish_reason, "token_origin": "vllm.RequestOutput",
                             "logprobs_mode": pending.pop("logprobs_mode")}
                if "context_budget" in pending:
                    step.info["context_budget"] = pending.pop("context_budget")
            return action

        def update_from_env(self, observation, reward, done, info, **kwargs):
            if self._trajectory.steps:
                step = self._trajectory.steps[-1]
                step.observation = messages_to_json_safe(adapter.build_next_messages())[-1]
                step.done = done
                step.info = {**step.info, "reward_computed": False, **info}

    class ProjectEnvironment(BaseEnv):
        def reset(self, task=None):
            if task is not None:
                adapter.initialize_episode(question=task["question"], images=task["images"], sample_id=task["sample_id"])
            return adapter.build_next_messages(), {"reward_computed": False}

        def step(self, action):
            adapter.apply_tool_calls(action.action)
            return adapter.build_next_messages(), 0.0, adapter.state().status in {"success", "fatal"}, {"reward_computed": False}

        @staticmethod
        def from_dict(info):
            raise RuntimeError("project environment requires an explicit shared adapter")

    class LocalEngine(RolloutEngine):
        async def get_model_response(self, messages, **kwargs):
            prompt_messages = messages_to_json_safe(messages) if capture_tokens else None
            try:
                output = backend.generate(messages=messages, tools=adapter.tool_registry.declarations_for_model())
            except ResponseContextBudgetExhausted as exc:
                state = adapter.state()
                diagnostics = {**exc.diagnostics, "prompt_id": state.context.sample_id}
                state.context_budget_exhaustion = diagnostics
                if not state.assistant_outputs:
                    from .workflow_types import RLInfrastructureError
                    raise RLInfrastructureError(
                        "initial Main prompt exceeds usable model context; "
                        f"zero-step member cannot enter Formal training: {diagnostics}") from exc
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) from exc
            except Exception as exc:
                adapter.record_failure(origin="model_generation_error", message=f"{type(exc).__name__}: {exc}")
                raise
            result = ModelOutput(text=output["text"], content=output["text"],
                               prompt_ids=output["prompt_ids"], completion_ids=output["completion_ids"],
                               prompt_length=len(output["prompt_ids"]), completion_length=len(output["completion_ids"]),
                               finish_reason=output["finish_reason"])
            if capture_tokens:
                if output["logprobs_mode"] != "processed_logprobs":
                    raise RuntimeError("real rollout processed-logprobs contract missing")
                result.logprobs = output["logprobs"]
                pending.update(output=result, prompt_messages=prompt_messages,
                               logprobs_mode=output["logprobs_mode"])
                if "context_budget" in output:
                    pending["context_budget"] = output["context_budget"]
            return result

    class NoRewardWorkflow(MultiTurnWorkflow):
        # Deliberately inherit run/reset/run_with_termination_handling unchanged.
        # Upstream default postprocess computes rewards/accuracy; Gate B must not.
        def postprocess_episode(self, episode, termination_reason=None, error=None):
            episode.id, episode.task = self.uid, {"sample_id": self.task["sample_id"], "question": self.task["question"]}
            episode.termination_reason = termination_reason
            episode.info = {"reward_computed": False, "timing": self.finalize_timing()}
            if adapter.episode is not None and adapter.episode.context_budget_exhaustion is not None:
                episode.info["context_budget_exhaustion"] = adapter.episode.context_budget_exhaustion
            if error is not None:
                episode.info["error"] = redact_secrets(error)
            return episode

    workflow = NoRewardWorkflow(agent_cls=ProjectAgent, env_cls=ProjectEnvironment,
                               max_steps=max_turns, rollout_engine=LocalEngine(), executor=executor)
    if workflow.run.__func__ is not MultiTurnWorkflow.run:
        raise RuntimeError("rLLM must own the generation loop")
    workflow.start_timing()
    components = (MultiTurnWorkflow, Workflow, BaseAgent, BaseEnv, RolloutEngine, ModelOutput, Step, Trajectory)
    provenance = {"components_used": [f"{cls.__module__}.{cls.__name__}" for cls in components],
                  "api_reference_commit": RLLM_SOURCE_COMMIT,
                  "installed_source_sha256": {f"{cls.__module__}.{cls.__name__}": sha256_file(inspect.getfile(cls))
                                              for cls in components},
                  "generation_loop": "rllm.workflows.multi_turn_workflow.MultiTurnWorkflow.run",
                  "compatibility_layer": ["BaseAgent/BaseEnv project bindings", "synchronous local vLLM RolloutEngine",
                                          "no-reward postprocess override"]}
    return workflow, provenance
