"""Deliberately thin integration boundary; no rLLM/verl/vLLM import."""

from __future__ import annotations

from typing import Protocol

from .workflow_types import RLTrajectory, RewardBreakdown


class FrameworkAdapter(Protocol):
    def to_episode(self, trajectory: RLTrajectory, reward: RewardBreakdown) -> object:
        """Map local rollout and reward to a verified framework episode."""
        ...

    def apply_fatal_mask(self, episode: object, *, fatal_start: int | None) -> object:
        """Mask generated suffix after the first fatal-cascade error."""
        ...
