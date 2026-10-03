"""Read-only Gate C handoff evidence; torch is lazy and no CUDA is required.

No loss, tokenization, model forward or optimizer operation lives here.
"""
from __future__ import annotations

import math

# Gate-only strict sanity bound, NOT a training hyperparameter. Small BF16
# kernel/merge drift is allowed, but a difference of 0.1 is already suspect.
MAX_ABS_LOGPROB_DIFF = 0.1
ALIGNMENT_CHECKS = (
    "pre_update_logprobs_computed", "pre_update_logprobs_finite",
    "pre_update_logprob_shape_match", "pre_update_policy_temperature_match",
    "pre_update_policy_ratio_finite", "pre_update_policy_no_initial_clipping",
    "pre_update_policy_logprob_diff_within_bound", "pre_update_policy_alignment_passed",
    "pre_update_policy_token_count_match",
)
STAT_FIELDS = (
    "mean_abs_logprob_diff", "max_abs_logprob_diff", "mean_signed_logprob_diff",
    "mean_importance_ratio", "min_importance_ratio", "max_importance_ratio",
    "initial_clip_fraction",
)


def compare_policy_logprobs(current_log_probs, old_log_probs, response_mask, *,
                            clip_ratio_low, clip_ratio_high,
                            max_abs_logprob_diff=MAX_ABS_LOGPROB_DIFF,
                            expected_masked_token_count=None):
    """Return JSON-safe PASS/FAIL statistics, selecting ONLY response_mask == 1.

    Do not clamp differences/ratios: overflow is a failure, not hidden evidence.
    Failed/undefined statistics are null rather than non-standard JSON NaN/Inf.
    """
    import torch

    if (not 0 < max_abs_logprob_diff <= MAX_ABS_LOGPROB_DIFF
            or not 0 < clip_ratio_low < 1 or not 0 < clip_ratio_high):
        raise ValueError("invalid Gate policy-alignment bounds")
    result = dict.fromkeys(STAT_FIELDS, None)
    result.update(masked_token_count=0, expected_masked_token_count=expected_masked_token_count,
                  ratio_lower_bound=1 - clip_ratio_low, ratio_upper_bound=1 + clip_ratio_high,
                  clip_ratio_low=clip_ratio_low, clip_ratio_high=clip_ratio_high,
                  max_abs_logprob_diff_bound=max_abs_logprob_diff,
                  shape_match=False, token_count_match=False, logprobs_finite=False,
                  ratio_finite=False, finite=False, all_finite=False, passed=False)
    if current_log_probs.shape != old_log_probs.shape or response_mask.shape != old_log_probs.shape:
        result["failure"] = "logprob/mask shape mismatch"
        return result
    result["shape_match"] = True
    if not bool(((response_mask == 0) | (response_mask == 1)).all()):
        result["failure"] = "response mask must be binary"
        return result
    mask = response_mask == 1
    count = int(mask.sum().item())
    result["masked_token_count"] = count
    result["token_count_match"] = count > 0 and (
        expected_masked_token_count is None or count == expected_masked_token_count)
    if not result["token_count_match"]:
        result["failure"] = "empty/inconsistent trainable response token count"
        return result
    # Select first: padding and post-fatal nonfinite values MUST NOT contaminate.
    current = current_log_probs.detach().masked_select(mask).to(dtype=torch.float64)
    old = old_log_probs.detach().masked_select(mask).to(dtype=torch.float64)
    result["logprobs_finite"] = bool(torch.isfinite(current).all() & torch.isfinite(old).all())
    if not result["logprobs_finite"]:
        result["failure"] = "nonfinite masked current/old logprobs"
        return result
    diff = current - old
    ratio = diff.exp()
    result["ratio_finite"] = bool(torch.isfinite(ratio).all())
    if not result["ratio_finite"]:
        result["failure"] = "nonfinite masked importance ratio"
        return result
    result.update(mean_abs_logprob_diff=diff.abs().mean().item(),
                  max_abs_logprob_diff=diff.abs().max().item(),
                  mean_signed_logprob_diff=diff.mean().item(),
                  mean_importance_ratio=ratio.mean().item(),
                  min_importance_ratio=ratio.min().item(), max_importance_ratio=ratio.max().item(),
                  initial_clip_fraction=((ratio < result["ratio_lower_bound"]) |
                                         (ratio > result["ratio_upper_bound"])).double().mean().item())
    result["all_finite"] = result["finite"] = all(math.isfinite(result[k]) for k in STAT_FIELDS)
    result["passed"] = (result["all_finite"] and result["initial_clip_fraction"] == 0.0
                        and result["max_abs_logprob_diff"] < max_abs_logprob_diff)
    if not result["passed"]:
        result["failure"] = "initial clipping or strict logprob-difference bound failed"
    return result


def temperature_matches(temperature, rollout_temperature):
    return (not isinstance(temperature, bool) and not isinstance(rollout_temperature, bool)
            and temperature == rollout_temperature == 0.7)


def alignment_checks(audit):
    """Derive each check independently from numeric/structural evidence."""
    stats_finite = all(isinstance(audit.get(k), (int, float))
                       and not isinstance(audit[k], bool) and math.isfinite(audit[k]) for k in STAT_FIELDS)
    checks = {
        "pre_update_logprobs_computed": audit.get("logprobs_computed") is True,
        "pre_update_logprobs_finite": audit.get("logprobs_finite") is True,
        "pre_update_logprob_shape_match": audit.get("shape_match") is True,
        "pre_update_policy_temperature_match": temperature_matches(
            audit.get("temperature"), audit.get("rollout_temperature")),
        "pre_update_policy_ratio_finite": audit.get("ratio_finite") is True and stats_finite,
        "pre_update_policy_no_initial_clipping": stats_finite and audit["initial_clip_fraction"] == 0.0
            and audit["min_importance_ratio"] >= 0.8 and audit["max_importance_ratio"] <= 1.28,
        "pre_update_policy_logprob_diff_within_bound": stats_finite
            and audit.get("max_abs_logprob_diff_bound") == MAX_ABS_LOGPROB_DIFF
            and 0 <= audit["max_abs_logprob_diff"] < MAX_ABS_LOGPROB_DIFF,
        "pre_update_policy_token_count_match": audit.get("token_count_match") is True
            and type(audit.get("masked_token_count")) is int and audit["masked_token_count"] > 0
            and audit["masked_token_count"] == audit.get("expected_masked_token_count"),
    }
    checks["pre_update_policy_alignment_passed"] = (
        all(checks.values()) and audit.get("all_finite") is True
        and audit.get("clip_ratio_low") == .2 and audit.get("clip_ratio_high") == .28
        and audit.get("ratio_lower_bound") == .8 and audit.get("ratio_upper_bound") == 1.28)
    return checks


def alignment_artifact(per_rank, *, gate_version, identity, trajectory_group_id, policy_fingerprint):
    """Both ranks are required; these are duplicate group rows, not extra rollouts."""
    if len(per_rank) != 2 or {r["rank"] for r in per_rank} != {0, 1}:
        raise ValueError("alignment requires exactly ranks 0 and 1")
    rows = sorted(per_rank, key=lambda r: r["rank"])
    checks = {k: all(alignment_checks(r).get(k) is True for r in rows) for k in ALIGNMENT_CHECKS}
    counts = [r["masked_token_count"] for r in rows]
    if len(set(counts)) != 1:
        checks["pre_update_policy_token_count_match"] = False
        checks["pre_update_policy_alignment_passed"] = False
    result = {**{k: rows[0].get(k) for k in (
        "temperature", "rollout_temperature", "clip_ratio_low", "clip_ratio_high",
        "ratio_lower_bound", "ratio_upper_bound", "max_abs_logprob_diff_bound")},
        "gate_version": gate_version, "identity": identity, "trajectory_group_id": trajectory_group_id,
        "pre_update_policy_fingerprint": policy_fingerprint, "masked_token_count": counts[0],
        "per_rank": rows, "checks": checks, "passed": all(checks.values()),
        "all_finite": all(r.get("all_finite") is True for r in rows),
        "optimizer_step_count": 0, "formal_rl_initialization_allowed": False}
    # Equal row/token counts on both ranks; averaged means describe one group.
    for key in STAT_FIELDS:
        vals = [r.get(key) for r in rows]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            result[key] = None
        elif key.startswith("max_") or key == "initial_clip_fraction":
            result[key] = max(vals)
        elif key.startswith("min_"):
            result[key] = min(vals)
        else:
            result[key] = sum(vals) / len(vals)
    return result


def require_policy_alignment(audit):
    if audit.get("passed") is not True or any(audit.get("checks", {}).get(k) is not True for k in ALIGNMENT_CHECKS):
        failed = [k for k in ALIGNMENT_CHECKS if audit.get("checks", {}).get(k) is not True]
        raise RuntimeError(f"pre-update policy alignment failed before optimizer (0 steps): {failed}")


def formal_alignment_artifact(per_rank, *, world_size, window_sha256):
    """Generic rank-local O/C evidence: token counts need NOT be equal."""
    if (type(world_size) is not int or world_size < 1 or len(per_rank) != world_size
            or any(type(r.get("rank")) is not int for r in per_rank)
            or {r["rank"] for r in per_rank} != set(range(world_size))):
        raise ValueError("complete distinct formal rank alignment required")
    rows = sorted(per_rank, key=lambda r: r["rank"])
    from .old_logprob import ALIGNMENT_META
    if any(r.get("window_sha256") != window_sha256 or any(r.get(k) != v for k, v in ALIGNMENT_META.items()) for r in rows):
        raise ValueError("foreign window or non O/C comparison")
    checks = {k: all(alignment_checks(r)[k] for r in rows) for k in ALIGNMENT_CHECKS}
    counts = [r["masked_token_count"] for r in rows]
    result = dict(window_sha256=window_sha256, world_size=world_size, per_rank=rows,
                  masked_token_count=sum(counts), checks=checks, passed=all(checks.values()), **ALIGNMENT_META)
    for key in STAT_FIELDS:
        values = [r.get(key) for r in rows]
        if not sum(counts) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            result[key] = None
        elif key.startswith("max_"):
            result[key] = max(values)
        elif key.startswith("min_"):
            result[key] = min(values)
        else:
            result[key] = sum(v * n for v, n in zip(values, counts)) / sum(counts)
    return result
