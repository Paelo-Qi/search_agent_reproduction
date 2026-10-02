"""Gate B CPU contracts: mocks here cannot publish a real GPU PASS."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.rl import rollout_gate
from opensearch_vl_repro.rl.actor_gate import atomic_json
from opensearch_vl_repro.rl.data import question_sha256
from opensearch_vl_repro.rl.rollout_gate import (
    GATE_B_CHECKS, VLLMStaticBackend, create_gate_tool_registry, diagnostic_crop_probe,
    gate_passed, load_rollout_gate_config, roundtrip_checks, run_rollout_gate, select_probe_image,
)
from opensearch_vl_repro.rl.workflow_adapter import RLWorkflowAdapter

ROOT = Path(__file__).resolve().parents[1]


def test_gate_config_is_inference_only_and_formal_rl_stays_original():
    config = load_rollout_gate_config(ROOT / "configs/rl_gate_b.yaml")
    assert config["vllm"] == {"tensor_parallel_size": 1, "max_model_len": 8192, "max_new_tokens": 256,
                              "temperature": 0.0, "gpu_memory_utilization": .6}
    assert config["agent"]["max_turns"] == 4
    formal = yaml.safe_load((ROOT / "configs/rl_main.yaml").read_text(encoding="utf-8"))
    assert formal["model"]["sft_adapter"] == "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    assert formal["algorithm"] == {"advantage_estimator": "rloo", "fatal_consecutive_errors": 3}
    assert formal["reward"] == {"accuracy_weight": .8, "query_weight": .2, "format_multiplicative": True}
    assert formal["rollout_n"] == 4 and formal["data"]["main_count"] == 400


def test_gate_pass_requires_literal_all_checks():
    checks = dict.fromkeys(GATE_B_CHECKS, True)
    assert gate_passed(checks)
    for name in checks:
        for wrong in (False, None, 1):
            assert not gate_passed({**checks, name: wrong})
    assert not gate_passed({})


def test_source_first_image_only_hash_check_no_mutation(tmp_path):
    image = Image.new("RGB", (9, 7), "blue")
    image.save(tmp_path / "first.png")
    row = {"source_sample_id": "rl_000002", "prompt_id": "rl_000002", "question": "What?",
           "question_hash": question_sha256("What?"), "image_relpaths": ["first.png", "not_opened.png"],
           "image_hashes": [image_sha256(image), "b" * 64]}
    before = copy.deepcopy(row)
    selected, loaded = select_probe_image([row], 0, tmp_path)
    assert selected is row and loaded.size == image.size and row == before
    with pytest.raises(ValueError, match="hash"):
        select_probe_image([{**row, "image_hashes": ["0" * 64, "b" * 64]}], 0, tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        select_probe_image([{**row, "image_relpaths": ["../escape.png", "x"]}], 0, tmp_path)


@pytest.mark.parametrize("size", [(1, 1), (2, 3), (29, 57), (2000, 999)])
def test_crop_probe_uses_actual_dimensions_and_bounds(size):
    question, args = diagnostic_crop_probe(Image.new("RGB", size), .5)
    assert 0 <= args["x"] < size[0] and 0 <= args["y"] < size[1]
    assert 0 < args["width"] <= size[0] - args["x"]
    assert 0 < args["height"] <= size[1] - args["y"]
    assert args["image"] == "img_1" and "diagnostic" in question and "exactly once" in question


def test_roundtrip_checks_require_real_second_image_not_text_only():
    adapter = RLWorkflowAdapter(create_gate_tool_registry())
    adapter.initialize_episode(question="fixture", images=[Image.new("RGB", (8, 6), "blue")], sample_id="s")
    parsed = adapter.handle_model_output('<tool_call>{"name":"crop","arguments":{"image":"img_1","x":2,"y":1,"width":4,"height":3}}</tool_call>')
    adapter.apply_tool_calls(parsed)
    adapter.handle_model_output("A blue crop.")
    trajectory = adapter.finalize_episode(termination="env_done")
    receipts = [{"succeeded": True}, {"succeeded": True, "actual_pil_inputs": True,
                  "multimodal_image_count": 2, "images": [{}, {"sha256": trajectory.images[1]["sha256"], "size": [4, 3]}]}]
    _, expected = diagnostic_crop_probe(Image.new("RGB", (8, 6)), .5)
    assert all(roundtrip_checks(trajectory, receipts, expected_crop_args=expected).values())
    for changed in ({"actual_pil_inputs": False}, {"images": [{}]}, {"succeeded": False}):
        assert not all(roundtrip_checks(trajectory, [receipts[0], {**receipts[1], **changed}], expected_crop_args=expected).values())
    direct = RLWorkflowAdapter(create_gate_tool_registry())
    direct.initialize_episode(question="fixture", images=[Image.new("RGB", (8, 6))], sample_id="s")
    direct.handle_model_output("Direct answer")
    checks = roundtrip_checks(direct.finalize_episode(termination="env_done"), [{"succeeded": True}], expected_crop_args=expected)
    assert checks["trajectory_success"] and not checks["model_generated_crop"]


@pytest.mark.parametrize("changed", [{}, {"x": 3}, {"y": 2}, {"width": 3}, {"height": 2},
                                   {"image": "img_2"}, {"extra_argument": 1}])
def test_diagnostic_crop_arguments_must_match_exactly(changed):
    _, expected = diagnostic_crop_probe(Image.new("RGB", (8, 6)), .5)
    turn = SimpleNamespace(tool_call={"name": "crop", "arguments": {**expected, **changed}},
                           status="success", metadata={})
    trajectory = SimpleNamespace(turns=[turn], images=[], final_answer="fixture", status="success")
    checks = roundtrip_checks(trajectory, [], expected_crop_args=expected)
    assert checks["model_generated_crop"] is (not changed)
    assert checks["crop_img_1_success"] is (not changed)


@pytest.mark.parametrize("second_image", ["img_1", "img_2"])
def test_two_successful_crops_fail_even_when_first_matches(second_image):
    _, expected = diagnostic_crop_probe(Image.new("RGB", (8, 6)), .5)
    turns = [SimpleNamespace(tool_call={"name": "crop", "arguments": expected}, status="success", metadata={}),
             SimpleNamespace(tool_call={"name": "crop", "arguments": {**expected, "image": second_image, "width": 1}},
                             status="success", metadata={})]
    trajectory = SimpleNamespace(turns=turns, images=[], final_answer="fixture", status="success")
    checks = roundtrip_checks(trajectory, [], expected_crop_args=expected)
    assert checks["model_generated_crop"] is False and checks["crop_img_1_success"] is False


@pytest.fixture
def publication_gate(tmp_path, monkeypatch):
    """CPU-only fake infrastructure; exercise the real runner's publication path."""
    source = tmp_path / "source"
    base = tmp_path / "base"
    source.mkdir(); base.mkdir()
    image = Image.new("RGB", (8, 6), "blue")
    image.save(source / "probe.png")
    row = {"source_sample_id": "rl_000002", "prompt_id": "rl_000002", "question": "fixture",
           "question_hash": question_sha256("fixture"), "image_relpaths": ["probe.png"],
           "image_hashes": [image_sha256(image)]}
    args = SimpleNamespace(output_dir=tmp_path / "output", report_dir=tmp_path / "reports",
        gate_config=ROOT / "configs/rl_gate_b.yaml", tensor_parallel_size=None,
        actor_adapter=tmp_path / "actor/adapter", actor_gate_manifest=None,
        source_root=source, data=tmp_path / "input/smoke20.json", config=ROOT / "configs/rl_main.yaml",
        base_model=rollout_gate.BASE_MODEL, base_revision=rollout_gate.BASE_REVISION,
        sample_index=0, base_model_path=base, seed=20260506)
    versions = {"torch": "2.8.0", "transformers": "4.57.1", "peft": "0.21.1", "vllm": "0.11.0", "rllm": "0.2.1"}
    monkeypatch.setattr(rollout_gate.importlib.metadata, "version", lambda name: versions.get(name, "fixture"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True, set_device=lambda _: None,
        reset_peak_memory_stats=lambda _: None, is_initialized=lambda: False,
        max_memory_allocated=lambda _: 0, max_memory_reserved=lambda _: 0)))
    monkeypatch.setattr("opensearch_vl_repro.rl.config.load_rl_config", lambda _: {
        "model": {"sft_config": "configs/sft_main_imageid_v3.yaml", "sft_adapter": "unused_fixture"},
        "data": {"quality_audit_dir": "data/rl_quality_audit"}})
    monkeypatch.setattr("opensearch_vl_repro.sft_train_plan.load_main_config", lambda *a, **k: {})
    monkeypatch.setattr(rollout_gate, "validate_actor_adapter", lambda **k: {"actor_adapter_fingerprint": "a" * 64})
    monkeypatch.setattr(rollout_gate, "load_smoke_records", lambda *a: ([row], {"manifest_sha256": "d" * 64}))
    monkeypatch.setattr(rollout_gate, "merge_actor_adapter", lambda **k: {
        "identity": {"merged_checkpoint_fingerprint": "f" * 64}, "fresh_hf_forward_finite": True,
        "no_active_peft": True, "merge_hf_destroyed": True, "reload_hf_destroyed": True})
    class Backend:
        def __init__(self, **kwargs):
            self.receipts = []
            self.closed = False
        def close(self):
            self.closed = True
    backend = Backend()
    monkeypatch.setattr(rollout_gate, "VLLMStaticBackend", lambda **k: backend)
    def workflow_binding(*, adapter, **kwargs):
        class Workflow:
            async def run_with_termination_handling(self, task, uid):
                _, expected = diagnostic_crop_probe(task["images"][0], .5)
                text = '<tool_call>' + json.dumps({"name": "crop", "arguments": expected}) + '</tool_call>'
                adapter.apply_tool_calls(adapter.handle_model_output(text))
                adapter.handle_model_output("CPU fixture final answer")
                derived = adapter.state().context.image_registry.get("img_2")
                backend.receipts = [{"succeeded": True}, {"succeeded": True, "actual_pil_inputs": True,
                    "multimodal_image_count": 2, "images": [{}, {"sha256": image_sha256(derived), "size": list(derived.size)}]}]
                return SimpleNamespace(id="cpu-fixture", termination_reason=SimpleNamespace(value="env_done"),
                                       info={}, to_dict=lambda: {"cpu_fixture_only": True})
        return Workflow(), {"cpu_fixture_only": True}
    monkeypatch.setattr(rollout_gate, "build_rllm_workflow", workflow_binding)
    return args, backend


def test_final_report_precedes_manifest_and_manifest_is_last_pass_artifact(publication_gate, monkeypatch):
    args, backend = publication_gate
    writes = []
    manifest = args.output_dir / "gate_manifest.json"
    report = args.report_dir / "gate_b_report.json"
    def writer(path, value):
        if value.get("passed") is True:
            assert backend.closed
            assert not manifest.exists()
            if path == manifest:
                assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True
        atomic_json(path, value)
        writes.append((path, value.get("passed")))
    monkeypatch.setattr(rollout_gate, "atomic_json", writer)
    assert run_rollout_gate(args, ROOT) == 0
    assert writes[-2:] == [(report, True), (manifest, True)]
    assert json.loads(manifest.read_text(encoding="utf-8"))["passed"] is True


@pytest.mark.parametrize("failure", ["final_report", "manifest_before_commit", "manifest_after_commit", "pass_stdout"])
def test_publication_failure_exits_with_false_report_and_no_pass_manifest(publication_gate, monkeypatch, failure):
    args, backend = publication_gate
    manifest = args.output_dir / "gate_manifest.json"
    report = args.report_dir / "gate_b_report.json"
    def writer(path, value):
        if value.get("passed") is True:
            assert backend.closed
            if path == report and failure == "final_report":
                assert not manifest.exists()
                raise OSError("injected publication failure")
            if path == manifest and failure.startswith("manifest_"):
                assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True
                if failure == "manifest_after_commit":
                    atomic_json(path, value)
                raise OSError("injected publication failure")
        atomic_json(path, value)
    def printer(text, **kwargs):
        if failure == "pass_stdout" and text.startswith("Gate B PASS"):
            assert json.loads(manifest.read_text(encoding="utf-8"))["passed"] is True
            raise OSError("injected publication failure")
    monkeypatch.setattr(rollout_gate, "atomic_json", writer)
    monkeypatch.setattr(rollout_gate, "print", printer, raising=False)
    with pytest.raises(OSError, match="injected publication failure"):
        run_rollout_gate(args, ROOT)
    assert not manifest.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False


def test_vllm_bridge_receives_exact_pil_objects_including_real_crop():
    adapter = RLWorkflowAdapter(create_gate_tool_registry())
    initial = Image.new("RGB", (8, 6), "green")
    adapter.initialize_episode(question="fixture", images=[initial], sample_id="s")
    adapter.apply_tool_calls(adapter.handle_model_output('<tool_call>{"name":"crop","arguments":{"image":"img_1","x":0,"y":0,"width":4,"height":3}}</tool_call>'))
    derived = adapter.state().context.image_registry.get("img_2")
    calls = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert isinstance(messages[0]["content"], str)
            assert kwargs == {"tools": adapter.tool_registry.declarations_for_model(),
                              "tokenize": False, "add_generation_prompt": True}
            return "CPU mock prompt"
        def __call__(self, **kwargs):
            assert kwargs == {"text": ["CPU mock prompt"], "images": [[initial, derived]],
                              "return_tensors": "pt", "truncation": False}
            assert kwargs["images"][0][0] is initial and kwargs["images"][0][1] is derived
            return {"input_ids": SimpleNamespace(shape=(1, 300))}
    def generate(prompts, **kwargs):
        calls.append(prompts)
        return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(text="answer", token_ids=[2], finish_reason="stop")])]
    backend = VLLMStaticBackend.__new__(VLLMStaticBackend)
    backend.processor, backend.llm = Processor(), SimpleNamespace(generate=generate)
    backend.receipts, backend.sampling, backend.max_images = [], object(), 5
    backend.settings = {"max_model_len": 8192, "max_new_tokens": 256}
    result = backend.generate(messages=adapter.build_next_messages(), tools=adapter.tool_registry.declarations_for_model())
    assert result["text"] == "answer"
    actual = calls[0][0]["multi_modal_data"]["image"]
    assert actual[0] is initial and actual[1] is derived
    assert backend.receipts[0]["succeeded"] is True
    assert backend.receipts[0]["images"][1]["sha256"] == image_sha256(derived)
    backend.settings["max_model_len"] = 301
    with pytest.raises(RuntimeError, match="no truncation"):
        backend.generate(messages=adapter.build_next_messages(), tools=adapter.tool_registry.declarations_for_model())


def test_offline_failure_report_remains_false_and_nonzero(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("gate_cli", ROOT / "scripts/validate_rl_rollout_roundtrip.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["--source-root", str(tmp_path / "source"),
        "--actor-adapter", str(tmp_path / "actor/adapter"), "--output-dir", str(tmp_path / "output"),
        "--report-dir", str(tmp_path / "reports")])
    monkeypatch.setattr("opensearch_vl_repro.rl.rollout_gate.importlib.metadata.version", lambda _: (_ for _ in ()).throw(RuntimeError("CPU deliberately missing software")))
    with pytest.raises(RuntimeError, match="missing software"):
        run_rollout_gate(args, ROOT)
    report = json.loads((args.report_dir / "gate_b_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["stage"] == "software"
    assert not (args.output_dir / "gate_manifest.json").exists()


def test_import_and_cli_help_on_windows_without_heavy_dependencies():
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','peft','verl','ray','vllm','rllm'}:
            raise AssertionError('heavy top-level import: ' + fullname)
sys.meta_path.insert(0, Block())
from opensearch_vl_repro.rl import rollout_gate, rollout_sync, workflow_adapter
import runpy
sys.argv=['validate_rl_rollout_roundtrip.py', '--help']
runpy.run_path('scripts/validate_rl_rollout_roundtrip.py', run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gate_never_calls_eval_loop_dynamic_lora_or_training():
    for file in ("rollout_gate.py", "rollout_sync.py", "workflow_adapter.py"):
        source = (ROOT / "src/opensearch_vl_repro/rl" / file).read_text(encoding="utf-8")
        for prohibited in ("AgentRuntime.run(", "LoRARequest(", "enable_lora=", "update_policy(", ".backward(",
                           "optimizer.step(", "ray.init(", "from opensearch_vl_repro.rl.reward"):
            assert prohibited not in source
