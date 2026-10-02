"""n=2 contract plus lazy official verl RLOO. Clamp only AFTER group statistics."""
import math

from opensearch_vl_repro.rl.reward import clamp_fatal_advantages


def rloo_pair(rewards):
    if len(rewards) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in rewards):
        raise ValueError("finite n=2 rewards required")
    return [float(rewards[0] - rewards[1]), float(rewards[1] - rewards[0])]


def official_rloo(rewards, fatal, *, group_id="group"):
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage
    raw, returns = compute_rloo_outcome_advantage(torch.tensor(rewards, dtype=torch.float64)[:, None],
                                                torch.ones(2, 1, dtype=torch.float64), np.array([group_id, group_id]))
    values = raw[:, 0].tolist()
    if not all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(values, rloo_pair(rewards))):
        raise RuntimeError("official verl RLOO differs from pinned n=2 contract")
    return values, clamp_fatal_advantages(values, fatal)
