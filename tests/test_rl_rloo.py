import math
import sys
from types import ModuleType

import pytest

from opensearch_vl_repro.rl.rloo import official_rloo, rloo_pair
from opensearch_vl_repro.rl.reward import clamp_fatal_advantages


def test_n2_leave_one_out_uses_full_group_before_fatal_clamp():
    assert rloo_pair([.9, .1]) == pytest.approx([.8, -.8])
    assert clamp_fatal_advantages(rloo_pair([.9, .1]), [False, True]) == pytest.approx([.8, 0])
    assert clamp_fatal_advantages(rloo_pair([.9, .1]), [True, False]) == pytest.approx([.8, -.8])
    assert rloo_pair([.5, .5]) == [0, 0]


@pytest.mark.parametrize("value", [[], [1], [1, 2, 3], [True, .1], [math.nan, .1]])
def test_bad_group_rewards_fail(value):
    with pytest.raises(ValueError): rloo_pair(value)


def test_official_rloo_adapter_delegates_and_checks_contract(monkeypatch):
    import torch
    calls = []
    module = ModuleType("verl.trainer.ppo.core_algos")
    def implementation(rewards, response_mask, index):
        calls.append((rewards.clone(), response_mask.clone(), index.copy()))
        out = torch.tensor([[.8], [-.8]], dtype=torch.float64)
        return out, out
    module.compute_rloo_outcome_advantage = implementation
    monkeypatch.setitem(sys.modules, module.__name__, module)
    raw, final = official_rloo([.9, .1], [False, True])
    assert raw == pytest.approx([.8, -.8]) and final == pytest.approx([.8, 0])
    assert len(calls) == 1 and calls[0][2].tolist() == ["group", "group"]
    module.compute_rloo_outcome_advantage = lambda *args: (torch.zeros(2, 1), torch.zeros(2, 1))
    with pytest.raises(RuntimeError, match="differs"): official_rloo([.9, .1], [False, False])
