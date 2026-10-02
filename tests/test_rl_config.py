from pathlib import Path

import pytest

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.rl.config import load_rl_config
from opensearch_vl_repro.rl.quality_audit import QUALITY_AUDIT_VERSION, QUALITY_ELIGIBILITY_RULE


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,rollout_n", [("rl_smoke", 2), ("rl_main", 4)])
def test_configs_load_without_framework(name, rollout_n):
    config = load_rl_config(ROOT / "configs" / f"{name}.yaml")
    assert config["rollout_n"] == rollout_n
    assert config["tool"]["resolved_runtime_protocol"] == RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    assert config["data"]["quality_audit_dir"] == "data/rl_quality_audit"
    assert config["data"]["quality_audit_version"] == QUALITY_AUDIT_VERSION
    assert config["data"]["quality_eligibility_rule"] == QUALITY_ELIGIBILITY_RULE
    assert "fsdp" not in config and "vllm" not in config


def test_wrong_protocol_fails_closed(tmp_path):
    source = (ROOT / "configs/rl_smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "rl.yaml"
    path.write_text(source.replace("runtime_protocol: current", "runtime_protocol: legacy"), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        load_rl_config(path)


def test_no_separate_pilot_config():
    assert not (ROOT / "configs/rl_pilot.yaml").exists()


def test_invalid_reward_fails_closed(tmp_path):
    source = (ROOT / "configs/rl_smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "rl.yaml"
    path.write_text(source.replace("accuracy_weight: 0.8", "accuracy_weight: 0.7"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rl_config(path)


def test_wrong_quality_audit_contract_fails_closed(tmp_path):
    source = (ROOT / "configs/rl_smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "rl.yaml"
    path.write_text(source.replace("quality_eligibility_rule: status == ok",
                                   "quality_eligibility_rule: status != exclude"), encoding="utf-8")
    with pytest.raises(ValueError, match="quality audit"):
        load_rl_config(path)
