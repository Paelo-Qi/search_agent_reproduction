"""Small provenance guards for the v3 Agent runtime and historical repair CLIs."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from opensearch_vl_repro.agent.reliability import (AGENT_BEHAVIOR_VERSION,
                                                   CACHE_SCHEMA_VERSION,
                                                   LAYOUT_BEHAVIOR_VERSION,
                                                   SEARCH_BEHAVIOR_VERSION)
from opensearch_vl_repro.sft_repair_training import run_repair


ROOT = Path(__file__).resolve().parents[1]
REPAIR_ERROR = r"legacy v2 targeted repair only supports image_search\.url; .*v3 image_search\.image_id"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev50_default_uses_formal_base_eval_config():
    path = ROOT / "scripts/run_tool_protocol_dev.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults = [keyword.value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--config"
                for keyword in node.keywords if keyword.arg == "default"]
    assert len(defaults) == 1
    assert ast.unparse(defaults[0]) == "ROOT / 'configs/eval_base_300.yaml'"
    config = yaml.safe_load((ROOT / "configs/eval_base_300.yaml").read_text(encoding="utf-8"))
    assert config["generation"]["max_new_tokens"] == 512
    assert config["runtime"]["max_agent_turns"] == 16


def test_image_id_v3_agent_search_v4_and_unchanged_other_versions():
    assert AGENT_BEHAVIOR_VERSION == 5
    assert (CACHE_SCHEMA_VERSION, SEARCH_BEHAVIOR_VERSION, LAYOUT_BEHAVIOR_VERSION) == (1, 4, 1)


def test_legacy_repair_training_fails_before_loading_config(tmp_path):
    with pytest.raises(RuntimeError, match=REPAIR_ERROR):
        run_repair(tmp_path / "missing.yaml")


@pytest.mark.parametrize("mode", [None, "argument_only", "full_tool_call", "r3"])
def test_legacy_repair_prepare_fails_before_touching_artifacts(tmp_path, monkeypatch, mode):
    script = _load_script("prepare_sft_repair.py")
    artifact = tmp_path / "repair.json"
    artifact.write_bytes(b"historical-v2-repair")
    arguments = ["prepare_sft_repair.py", "--output-root", str(tmp_path)]
    if mode is not None:
        arguments += ["--mode", mode]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(RuntimeError, match=REPAIR_ERROR):
        script.main()
    assert artifact.read_bytes() == b"historical-v2-repair"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["repair.json"]


def test_legacy_repair_mask_audit_fails_before_touching_artifacts(tmp_path, monkeypatch):
    script = _load_script("audit_sft_repair_masks.py")
    artifact = tmp_path / "historical-checkpoint.bin"
    artifact.write_bytes(b"historical-v2-checkpoint")
    monkeypatch.setattr(sys, "argv", ["audit_sft_repair_masks.py", "--config",
                                      str(tmp_path / "missing.yaml"), "--report-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match=REPAIR_ERROR):
        script.main()
    assert artifact.read_bytes() == b"historical-v2-checkpoint"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["historical-checkpoint.bin"]
