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


def assemble_window_rloo(window, run, policy, groups, *, estimator=None):
    """Official verl first, then fatal clamp. Injection is for CPU contract tests only.

    There is deliberately no production fallback when verl is unavailable.
    One outcome row per generation; prompt group IDs prevent cross-prompt baselines.
    """
    from .checkpoint import seal
    from .training_window import validate_training_window
    validate_training_window(window, run, policy, groups)
    import numpy as np
    import torch
    injected = estimator is not None
    if estimator is None:
        from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage
        estimator = compute_rloo_outcome_advantage
    rows = [(g["identity"]["trajectory_group_id"], m) for g in groups
            for m in sorted(g["members"], key=lambda member: member["rollout_index"])]
    rewards = [float(m["reward"]["total"]) for _, m in rows]
    raw, returns = estimator(torch.tensor(rewards, dtype=torch.float64)[:, None],
                             torch.ones(len(rows), 1, dtype=torch.float64),
                             np.array([group_id for group_id, _ in rows]))
    if (raw.shape != (len(rows), 1) or returns.shape != raw.shape
            or not torch.isfinite(raw).all() or not torch.isfinite(returns).all()):
        raise ValueError("invalid official RLOO output")
    values = raw[:, 0].tolist()
    # Sanity check the official result, NEVER use this calculation as a fallback.
    for group in groups:
        indices = [i for i, (gid, _) in enumerate(rows) if gid == group["identity"]["trajectory_group_id"]]
        total = math.fsum(rewards[i] for i in indices)
        for i in indices:
            expected = rewards[i] - (total - rewards[i]) / (len(indices) - 1)
            if not math.isclose(values[i], expected, abs_tol=1e-12):
                raise RuntimeError("official verl RLOO differs from per-prompt leave-one-out contract")
    final = clamp_fatal_advantages(values, [m["fatal"] for _, m in rows])
    if not all(math.isfinite(v) for v in final):
        raise ValueError("nonfinite clamped advantages")
    return seal({"window_sha256": window["window_sha256"],
                 "estimator": "official_verl_rloo", "test_estimator_injected": injected,
                 "status": "zero_signal" if all(v == 0 for v in final) else "signal",
                 "rows": [{"group_id": gid, "member_id": m["member_id"],
                           "rollout_index": m["rollout_index"], "reward": rewards[i],
                           "fatal": m["fatal"], "raw_advantage": values[i], "final_advantage": final[i]}
                          for i, (gid, m) in enumerate(rows)]}, "reward_window_sha256")
