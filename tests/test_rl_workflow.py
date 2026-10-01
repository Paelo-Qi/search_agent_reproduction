import importlib.util

from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn
from opensearch_vl_repro.rl.framework_adapter import FrameworkAdapter
from opensearch_vl_repro.rl.metrics import METRIC_NAMES, aggregate_group, trajectory_metrics
from opensearch_vl_repro.rl.workflow_types import FatalInfo, RLTrajectory, RewardBreakdown


def test_rl_view_reuses_agent_turn_and_imports_without_trainer():
    turn = AgentTurn("answer", None, None, "success")
    agent = AgentTrajectory("s", "synthetic", [turn], "answer", "success", ["img_1"])
    view = RLTrajectory(agent, generated_tokens=3)
    assert view.steps[0].turn is turn
    assert trajectory_metrics(agent, generated_tokens=3)["trajectory/generated_tokens"] == 3
    assert "fatal/preserved_prefix_length" in METRIC_NAMES
    assert FrameworkAdapter is not None


def test_group_metrics_zero_variance_and_fatal_rate():
    breakdown = RewardBreakdown(1, 1, 0, .8)
    normal = FatalInfo(False, None, None, 2, 3)
    fatal = FatalInfo(True, 1, "consecutive_model_tool_errors", 1, 3)
    result = aggregate_group([breakdown, breakdown], [normal, fatal])
    assert result.zero_variance and result.fatal_rate == .5
