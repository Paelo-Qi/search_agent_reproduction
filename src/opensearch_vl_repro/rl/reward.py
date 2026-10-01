"""Deterministic format reward and strict composition around existing judges."""

from __future__ import annotations

import math
from typing import Protocol

from opensearch_vl_repro.agent.runtime import AgentTrajectory
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS
from opensearch_vl_repro.agent.tool_parser import ToolCallParser
from opensearch_vl_repro.evaluation.judge import JudgeResult, JudgeSample

from .fatal import classify_turn, detect_fatal
from .workflow_types import RLInfrastructureError, RewardBreakdown


class CorrectnessJudge(Protocol):
    def judge(self, sample: JudgeSample) -> JudgeResult: ...


def unit_reward(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def compose_reward(format_reward: float, accuracy_reward: float, query_reward: float,
                   *, accuracy_weight: float = .8, query_weight: float = .2) -> RewardBreakdown:
    fmt = unit_reward(format_reward, "format_reward")
    acc = unit_reward(accuracy_reward, "accuracy_reward")
    query = unit_reward(query_reward, "query_reward")
    a = unit_reward(accuracy_weight, "accuracy_weight")
    q = unit_reward(query_weight, "query_weight")
    if not math.isclose(a + q, 1, abs_tol=1e-12):
        raise ValueError("reward weights must sum to one")
    return RewardBreakdown(fmt, acc, query, fmt * (a * acc + q * query))


def format_reward(trajectory: AgentTrajectory) -> float:
    """Score generated calls and termination in the current tool protocol.

    Fatal suffix is excluded; AgentTrajectory does not retain final-answer raw
    text as a turn, so terminal validity uses status and final_answer.
    """
    fatal = detect_fatal(trajectory)
    declarations = {item.name: item for item in TOOL_DECLARATIONS}
    parser = ToolCallParser(declarations)
    scores: list[float] = []
    for turn in trajectory.turns[:fatal.preserved_prefix_length]:
        if classify_turn(turn) == "model_error":
            scores.append(0.0)
            continue
        parsed = parser.parse(turn.assistant_output)
        if parsed.kind != "valid_tool_call" or not parsed.tool_calls or turn.tool_call is None:
            scores.append(0.0)
            continue
        try:
            for call in parsed.tool_calls:
                declarations[call.name].validate_arguments(call.arguments)
        except (KeyError, ValueError):
            scores.append(0.0)
        else:
            scores.append(1.0)
    if not fatal.fatal:
        scores.append(float(trajectory.status == "success" and bool(trajectory.final_answer and trajectory.final_answer.strip())))
    return sum(scores) / len(scores) if scores else 0.0


def accuracy_reward(sample: JudgeSample, judge: CorrectnessJudge) -> float:
    if sample.upstream_status not in {"success", "max_agent_turns_exceeded"}:
        raise RLInfrastructureError(f"unscorable upstream rollout: {sample.upstream_status}")
    if sample.upstream_status == "max_agent_turns_exceeded" or not sample.model_answer:
        return 0.0
    try:
        result = judge.judge(sample)
    except Exception as exc:
        raise RLInfrastructureError("correctness judge failed") from exc
    if result.status != "success" or result.verdict not in {"correct", "incorrect"}:
        raise RLInfrastructureError(f"correctness judge failure: {result.error_type or result.status}")
    return float(result.verdict == "correct")


def clamp_fatal_advantages(advantages: list[float], fatal: list[bool]) -> list[float]:
    """Preserve non-negative prefix learning for fatal rollouts.

    The future RLOO trainer owns advantage estimation using *all* trajectories,
    policy loss and token-level fatal response masking. This pure function only
    applies the paper's one-sided max(advantage, 0) clamp for fatal prefixes.
    """
    if not advantages or len(advantages) != len(fatal):
        raise ValueError("aligned, nonempty advantage and fatal lists required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in advantages) or any(not isinstance(value, bool) for value in fatal):
        raise ValueError("advantages must be finite and fatal flags boolean")
    return [max(float(advantage), 0.0) if is_fatal else float(advantage)
            for advantage, is_fatal in zip(advantages, fatal, strict=True)]
