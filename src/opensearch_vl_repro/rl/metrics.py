"""Stable metric names and lightweight aggregation for future trainer wiring."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from opensearch_vl_repro.agent.runtime import AgentTrajectory

from .workflow_types import FatalInfo, RewardBreakdown

METRIC_NAMES = (
    "reward/total", "reward/format", "reward/accuracy", "reward/query",
    "group/reward_mean", "group/reward_std", "group/zero_variance",
    "fatal/rate", "fatal/reason", "fatal/start", "fatal/preserved_prefix_length",
    "trajectory/turns", "trajectory/generated_tokens", "trajectory/tool_calls",
    "trajectory/tool_success", "trajectory/tool_error", "trajectory/duplicate_call",
    "trajectory/invalid_image_id", "infra/judge_api_failure", "infra/retry_count",
)


@dataclass(frozen=True)
class GroupMetrics:
    reward_mean: float
    reward_std: float
    zero_variance: bool
    fatal_rate: float


def aggregate_group(rewards: list[RewardBreakdown], fatals: list[FatalInfo]) -> GroupMetrics:
    if not rewards or len(rewards) != len(fatals):
        raise ValueError("group rewards and fatals must be aligned and nonempty")
    values = [item.total for item in rewards]
    std = pstdev(values)
    return GroupMetrics(mean(values), std, std == 0, mean(float(item.fatal) for item in fatals))


def trajectory_metrics(trajectory: AgentTrajectory, *, generated_tokens: int | None = None) -> dict[str, int | None]:
    turns = trajectory.turns
    return {
        "trajectory/turns": len(turns),
        "trajectory/generated_tokens": generated_tokens,
        "trajectory/tool_calls": sum(turn.tool_call is not None for turn in turns),
        "trajectory/tool_success": sum(turn.status == "success" for turn in turns),
        "trajectory/tool_error": sum(turn.status == "error" for turn in turns),
        "trajectory/duplicate_call": sum(turn.metadata.get("error_type") == "duplicate_tool_call" for turn in turns),
        "trajectory/invalid_image_id": sum(turn.metadata.get("error_type") == "unknown_image_id" for turn in turns),
    }
