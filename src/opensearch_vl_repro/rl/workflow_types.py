"""Small views over the existing Agent trajectory, not a second runtime schema."""

from __future__ import annotations

from dataclasses import dataclass

from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn


class RLInfrastructureError(RuntimeError):
    """A rollout/judge failure that must abort, never become a model reward."""


@dataclass(frozen=True)
class RLStep:
    index: int
    turn: AgentTurn


@dataclass(frozen=True)
class RLTrajectory:
    agent: AgentTrajectory
    generated_tokens: int | None = None

    @property
    def steps(self) -> tuple[RLStep, ...]:
        return tuple(RLStep(i, turn) for i, turn in enumerate(self.agent.turns))


@dataclass(frozen=True)
class FatalInfo:
    fatal: bool
    start_index: int | None
    reason: str | None
    preserved_prefix_length: int
    threshold: int


@dataclass(frozen=True)
class RewardBreakdown:
    format: float
    accuracy: float
    query: float
    total: float
