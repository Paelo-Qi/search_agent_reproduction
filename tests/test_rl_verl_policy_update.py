"""Read-only CPU regression of the existing formal actor configuration/alignment.

This requested test module was absent before v4.5.7. No policy loss/update runs.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from opensearch_vl_repro.rl.verl_policy_update import configure_one_update, audit_pre_update_policy


@dataclass
class Config:
    ppo_mini_batch_size: int = 0
    ppo_micro_batch_size_per_gpu: int = 0
    ppo_epochs: int = 0
    shuffle: bool = True
    clip_ratio_low: float = 0.
    clip_ratio_high: float = 0.
    entropy_coeff: float = 1.
    use_kl_loss: bool = True
    use_rollout_log_probs: bool = False
    loss_agg_mode: str = ""


def test_formal_runtime_config_still_true_one_epoch_no_new_diagnostic_config():
    actor = SimpleNamespace(config=Config())
    configure_one_update(actor, 3, dict(clip_ratio_low=.2, clip_ratio_high=.28))
    assert actor.config.use_rollout_log_probs is True
    assert actor.config.ppo_mini_batch_size == 3 and actor.config.ppo_epochs == 1
    assert actor.config.ppo_micro_batch_size_per_gpu == 1
    assert (actor.config.clip_ratio_low, actor.config.clip_ratio_high) == (.2, .28)
    with pytest.raises(ValueError):
        configure_one_update(actor, 0, {})


def test_logprobs_first_entropy_second_preserves_formal_alignment_return_contract():
    calls = []
    old = torch.tensor([[-.1, -.2]])
    def compute(data, calculate_entropy):
        assert not torch.is_grad_enabled() and calculate_entropy is False
        calls.append(data)
        return old.clone(), None  # pinned verl returns (log_probs, entropy)
    actor = SimpleNamespace(compute_log_prob=compute)
    data = SimpleNamespace(batch=dict(old_log_probs=old, response_mask=torch.ones_like(old)), meta_info=dict(temperature=.7))
    result = audit_pre_update_policy(actor, data, dict(clip_ratio_low=.2, clip_ratio_high=.28, vllm=dict(temperature=.7)), expected_masked_token_count=2)
    assert len(calls) == 1 and result["passed"] and result["mean_importance_ratio"] == 1
