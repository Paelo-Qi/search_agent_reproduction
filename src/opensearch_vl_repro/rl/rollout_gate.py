"""Gate B: static merged vLLM + upstream rLLM loop + real local crop.

No training, reward, provider, group rollout cache or accuracy evaluation.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import importlib.metadata
import os
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from opensearch_vl_repro.agent.local_visual_tools import LOCAL_VISUAL_BACKENDS
from opensearch_vl_repro.agent.reliability import image_sha256, redact_secrets
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS, RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry, ToolResult
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json, load_smoke_records, validate_output_paths
from opensearch_vl_repro.rl.data import question_sha256, safe_image_relpath
from opensearch_vl_repro.rl.rollout_sync import (
    GATE_B_VERSION, merge_actor_adapter, prepare_qwen_vl_processor_inputs, validate_actor_adapter,
)
from opensearch_vl_repro.rl.workflow_adapter import RLWorkflowAdapter, build_rllm_workflow

GATE_B_CHECKS = (
    "software_versions", "base_identity", "actor_adapter_verified", "actor_provenance_verified",
    "source_image_verified", "merged_checkpoint_created", "merged_files_valid",
    "fresh_hf_reload_finite", "no_active_peft", "merged_fingerprint_recorded",
    "hf_models_destroyed", "fresh_vllm_loaded", "initial_image_accepted", "real_rllm_workflow_used",
    "model_generated_crop", "crop_img_1_success", "img_2_registered", "img_2_parent_img_1",
    "derived_image_sha_recorded", "second_generation_received_derived_image",
    "second_generation_succeeded", "final_answer_present", "trajectory_success",
    "no_tool_failures", "no_external_provider", "vllm_destroyed", "artifacts_written",
)


def load_rollout_gate_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or set(config) != {"gate_version", "dtype", "merge_method", "vllm", "agent", "probe"}
            or config["gate_version"] != GATE_B_VERSION or config["dtype"] != "bfloat16"
            or config["merge_method"] != "peft.merge_and_unload"):
        raise ValueError("invalid Gate B version/dtype/static merge config")
    inference = config["vllm"]
    if (not isinstance(inference, dict) or set(inference) != {"tensor_parallel_size", "max_model_len", "max_new_tokens", "temperature", "gpu_memory_utilization"}
            or type(inference["tensor_parallel_size"]) is not int or inference["tensor_parallel_size"] < 1
            or type(inference["max_model_len"]) is not int or not 1024 <= inference["max_model_len"] <= 8192
            or type(inference["max_new_tokens"]) is not int or not 128 <= inference["max_new_tokens"] <= 256
            or inference["temperature"] != 0 or not 0 < inference["gpu_memory_utilization"] <= .9):
        raise ValueError("invalid Gate-only vLLM settings")
    if (set(config["agent"]) != {"max_turns"} or type(config["agent"]["max_turns"]) is not int
            or not 2 <= config["agent"]["max_turns"] <= 6
            or set(config["probe"]) != {"crop_fraction"}
            or not .1 <= config["probe"]["crop_fraction"] <= .9):
        raise ValueError("invalid Gate B max turns/crop fraction")
    return config


def validate_rollout_versions(versions: dict[str, str]) -> None:
    for name, expected in {"torch": "2.8.", "vllm": "0.11."}.items():
        if not versions.get(name, "").split("+")[0].startswith(expected):
            raise ValueError(f"Gate B requires {name} {expected}x, got {versions.get(name)}")
    for name, expected in {"transformers": "4.57.1", "peft": "0.21.1", "rllm": "0.2.1"}.items():
        if versions.get(name, "").split("+")[0] != expected:
            raise ValueError(f"Gate B requires {name}=={expected}, got {versions.get(name)}")


def create_gate_tool_registry() -> ToolRegistry:
    def disabled(arguments, context):
        return ToolResult("error", "<observation>Remote tools are disabled for Gate B.</observation>",
                          "gate_tool_disabled", {"provider_called": False})
    registry = ToolRegistry()
    for declaration in TOOL_DECLARATIONS:
        registry.register(RegisteredTool(declaration, LOCAL_VISUAL_BACKENDS.get(declaration.name, disabled)))
    return registry


def diagnostic_crop_probe(image: Image.Image, fraction: float) -> tuple[str, dict[str, Any]]:
    width, height = image.size
    if min(width, height) < 1 or not .1 <= fraction <= .9:
        raise ValueError("invalid image size/crop fraction")
    crop_width, crop_height = max(1, int(width * fraction)), max(1, int(height * fraction))
    arguments = {"image": "img_1", "x": (width - crop_width) // 2,
                 "y": (height - crop_height) // 2, "width": crop_width, "height": crop_height}
    question = (
        "This is a diagnostic tool-protocol probe, not a factual benchmark. "
        "Before answering, you must call the crop tool exactly once on registered img_1 "
        f"with x={arguments['x']}, y={arguments['y']}, width={crop_width}, height={crop_height}. "
        "Do not answer before the tool observation. After receiving the registered cropped image "
        "and inspecting its actual pixels, provide a short final answer describing one visible "
        "detail from that crop. Do not call remote/search tools."
    )
    return question, arguments


def select_probe_image(records: list[dict[str, Any]], index: int, root: Path) -> tuple[dict[str, Any], Image.Image]:
    if type(index) is not int or not 0 <= index < len(records):
        raise IndexError("Gate sample index out of bounds")
    row = records[index]
    identity = row.get("source_sample_id")
    paths, hashes = row.get("image_relpaths"), row.get("image_hashes")
    if (not isinstance(identity, str) or re.fullmatch(r"rl_\d{6}", identity) is None
            or row.get("prompt_id") != identity or "trajectory_group_id" in row
            or not isinstance(paths, list) or not paths or not isinstance(hashes, list) or len(paths) != len(hashes)
            or row.get("question_hash") != question_sha256(row["question"])):
        raise ValueError("invalid frozen schema-v3 probe record")
    root = root.resolve()
    path = (root / safe_image_relpath(paths[0])).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError("source probe image missing/escaping root")
    with Image.open(path) as loaded:
        image = loaded.convert("RGB").copy()
    if image_sha256(image) != hashes[0]:
        raise ValueError("source image hash differs from frozen smoke record")
    return row, image


class VLLMStaticBackend:
    """Synchronous local vLLM compatibility bridge; does not own a turn loop."""
    def __init__(self, *, checkpoint: Path, sft_config: dict[str, Any], gate: dict[str, Any], seed: int):
        from vllm import LLM, SamplingParams
        from opensearch_vl_repro.model import load_processor
        import torch
        config = copy.deepcopy(sft_config)
        config["model"]["name_or_path"] = str(checkpoint)
        self.processor = load_processor(config, local_files_only=True)
        self.receipts: list[dict[str, Any]] = []
        self.settings = gate["vllm"]
        self.max_images = gate["agent"]["max_turns"] + 1
        if torch.cuda.device_count() < self.settings["tensor_parallel_size"]:
            raise RuntimeError("visible GPU count is below Gate tensor_parallel_size")
        self.llm = LLM(model=str(checkpoint), tokenizer=str(checkpoint), dtype="bfloat16",
                       tensor_parallel_size=self.settings["tensor_parallel_size"],
                       max_model_len=self.settings["max_model_len"], max_num_seqs=1,
                       max_num_batched_tokens=self.settings["max_model_len"],
                       gpu_memory_utilization=self.settings["gpu_memory_utilization"],
                       limit_mm_per_prompt={"image": self.max_images},
                       mm_processor_kwargs={"max_pixels": sft_config["model"]["image_max_pixels"]},
                       distributed_executor_backend="mp", enforce_eager=True, trust_remote_code=False, seed=seed)
        self.sampling = SamplingParams(temperature=0.0, max_tokens=self.settings["max_new_tokens"])

    def generate(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        prompt, images, batch = prepare_qwen_vl_processor_inputs(
            self.processor, messages, tools, max_images=self.max_images)
        length = int(batch["input_ids"].shape[-1])
        del batch
        if length + self.settings["max_new_tokens"] > self.settings["max_model_len"]:
            raise RuntimeError("real multimodal probe does not fit Gate max_model_len; no truncation allowed")
        receipt = {"generation_index": len(self.receipts) + 1, "multimodal_image_count": len(images),
                   "actual_pil_inputs": True, "images": [{"sha256": image_sha256(image), "size": list(image.size)} for image in images],
                   "processor_token_length": length, "succeeded": False}
        self.receipts.append(receipt)
        # These exact PIL objects, not paths/URLs/base64 strings, cross generate.
        output = self.llm.generate([{"prompt": prompt, "multi_modal_data": {"image": images}}],
                                   sampling_params=self.sampling, use_tqdm=False)
        if len(output) != 1 or len(output[0].outputs) != 1:
            raise RuntimeError("unexpected vLLM generation output count")
        result = output[0].outputs[0]
        if not isinstance(result.text, str) or not result.text.strip() or not result.token_ids:
            raise RuntimeError("empty vLLM model output")
        receipt["succeeded"] = True
        receipt["completion_token_count"] = len(result.token_ids)
        return {"text": result.text, "prompt_ids": list(output[0].prompt_token_ids),
                "completion_ids": list(result.token_ids), "finish_reason": result.finish_reason}

    def close(self) -> None:
        if self.llm is not None:
            # Verified vLLM 0.11 V1 EngineCoreClient API, no dynamic adapter path.
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()


def roundtrip_checks(trajectory: Any, receipts: list[dict[str, Any]], *,
                     expected_crop_args: dict[str, Any]) -> dict[str, bool]:
    crops = [turn for turn in trajectory.turns if turn.tool_call and turn.tool_call["name"] == "crop"
             and turn.status == "success"]
    exact_crop = (len(crops) == 1 and crops[0].tool_call["arguments"].get("image") == "img_1"
                  and crops[0].tool_call["arguments"] == expected_crop_args)
    derived = next((entry for entry in trajectory.images if entry["image_id"] == "img_2" and entry["kind"] == "derived"), None)
    received = (len(receipts) >= 2 and derived is not None and receipts[1].get("actual_pil_inputs") is True
                and receipts[1].get("multimodal_image_count", 0) >= 2
                and any(item.get("sha256") == derived["sha256"] and item.get("size") == derived["size"]
                        for item in receipts[1].get("images", [])[1:]))
    return {"initial_image_accepted": bool(receipts) and receipts[0].get("succeeded") is True,
            "model_generated_crop": exact_crop, "crop_img_1_success": exact_crop,
            "img_2_registered": derived is not None,
            "img_2_parent_img_1": derived is not None and derived["parent_id"] == "img_1" and derived["metadata"].get("producing_tool") == "crop",
            "derived_image_sha_recorded": derived is not None and re.fullmatch(r"[0-9a-f]{64}", derived["sha256"]) is not None,
            "second_generation_received_derived_image": bool(received),
            "second_generation_succeeded": len(receipts) >= 2 and receipts[1].get("succeeded") is True,
            "final_answer_present": bool(trajectory.final_answer and trajectory.final_answer.strip()),
            "trajectory_success": trajectory.status == "success",
            "no_tool_failures": all(turn.status == "success" for turn in trajectory.turns),
            "no_external_provider": all(not turn.metadata.get("provider_called", False) for turn in trajectory.turns)}


def gate_passed(checks: dict[str, Any]) -> bool:
    return all(checks.get(name) is True for name in GATE_B_CHECKS)


def run_rollout_gate(args: Any, root: Path) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    output, reports = args.output_dir.resolve(), args.report_dir.resolve()
    config = load_rollout_gate_config(args.gate_config)
    if args.tensor_parallel_size is not None:
        if args.tensor_parallel_size < 1:
            raise ValueError("tensor parallel size must be positive")
        config["vllm"]["tensor_parallel_size"] = args.tensor_parallel_size
    protected = [args.actor_adapter.parent, args.source_root, root / "data", root / "configs",
                 root / "outputs/sft_main_imageid_v3", args.data.parent]
    if args.base_model_path:
        protected.append(args.base_model_path)
    if args.actor_gate_manifest:
        protected.append(args.actor_gate_manifest.parent)
    validate_output_paths(output, reports, protected)
    output.mkdir(parents=True, exist_ok=False)
    reports.mkdir(parents=True, exist_ok=False)
    report_path = reports / "gate_b_report.json"
    report: dict[str, Any] = {"passed": False, "gate_version": GATE_B_VERSION, "stage": "initialized",
                              "checks": dict.fromkeys(GATE_B_CHECKS, False)}
    atomic_json(report_path, report)
    backend, adapter, workflow = None, None, None
    stage, started = "initialized", time.monotonic()

    def enter(name: str):
        nonlocal stage
        stage = name
        report["stage"] = name
        atomic_json(report_path, redact_secrets(report))

    try:
        enter("software")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "vllm", "rllm")}
        versions["python"] = platform.python_version()
        try:
            versions["flash_attn"] = importlib.metadata.version("flash-attn")
        except importlib.metadata.PackageNotFoundError:
            versions["flash_attn"] = "not-installed"
        validate_rollout_versions(versions)
        report["software_versions"] = versions
        report["checks"]["software_versions"] = True
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Gate B requires real CUDA/BF16; CPU tests are not a PASS")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        if (args.base_model, args.base_revision) != (BASE_MODEL, BASE_REVISION):
            raise ValueError("Gate B model/revision must match pinned Qwen identity")
        from opensearch_vl_repro.rl.config import load_rl_config
        from opensearch_vl_repro.sft_train_plan import load_main_config
        rl = load_rl_config(args.config)
        sft = load_main_config(root / rl["model"]["sft_config"], base_eval_config=root / "configs/eval_base_300.yaml")
        report["checks"]["base_identity"] = True
        enter("actor_provenance")
        actor = validate_actor_adapter(adapter=args.actor_adapter, gate_manifest=args.actor_gate_manifest,
                                       rl_config=rl, sft_config=sft, source_sft_adapter=root / rl["model"]["sft_adapter"])
        report.update(actor)
        report["checks"].update(actor_adapter_verified=True, actor_provenance_verified=True)
        enter("source_probe")
        rl["data"]["quality_audit_dir"] = str(root / rl["data"]["quality_audit_dir"])
        records, manifest = load_smoke_records(args.data, rl)
        row, image = select_probe_image(records, args.sample_index, args.source_root)
        question, crop_arguments = diagnostic_crop_probe(image, config["probe"]["crop_fraction"])
        report.update(sample_id=row["source_sample_id"], source_image_sha256=image_sha256(image),
                      source_image_size=list(image.size), gate_selected_source_image_index=0,
                      source_data_manifest_sha256=manifest["manifest_sha256"], crop_probe_arguments=crop_arguments,
                      base_model=BASE_MODEL, base_revision=BASE_REVISION, vllm_config=config["vllm"])
        report["checks"]["source_image_verified"] = True
        adapter = RLWorkflowAdapter(create_gate_tool_registry())
        adapter.initialize_episode(question=question, images=[image], sample_id=row["source_sample_id"])
        base = args.base_model_path
        if base is None:
            from huggingface_hub import snapshot_download
            base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
        if not base.is_dir():
            raise FileNotFoundError("offline base snapshot missing")
        enter("static_merge_and_fresh_hf_reload")
        merge = merge_actor_adapter(base_snapshot=base.resolve(), adapter=args.actor_adapter.resolve(), actor=actor,
                                    sft_config=sft, output=output / "merged_model", versions=versions,
                                    validation_messages=adapter.build_next_messages(), tools=adapter.tool_registry.declarations_for_model(),
                                    on_stage=enter)
        report["merged_checkpoint_fingerprint"] = merge["identity"]["merged_checkpoint_fingerprint"]
        report["checks"].update(merged_checkpoint_created=True, merged_files_valid=True,
                                fresh_hf_reload_finite=merge["fresh_hf_forward_finite"], no_active_peft=merge["no_active_peft"],
                                merged_fingerprint_recorded=True,
                                hf_models_destroyed=merge["merge_hf_destroyed"] and merge["reload_hf_destroyed"])
        enter("fresh_vllm_init")
        backend = VLLMStaticBackend(checkpoint=output / "merged_model", sft_config=sft, gate=config, seed=args.seed)
        report["checks"]["fresh_vllm_loaded"] = True
        enter("rllm_owned_roundtrip")
        with ThreadPoolExecutor(max_workers=1) as executor:
            workflow, provenance = build_rllm_workflow(adapter=adapter, backend=backend, executor=executor,
                                                       max_turns=config["agent"]["max_turns"])
            report["rllm_components_used"] = provenance
            report["checks"]["real_rllm_workflow_used"] = True
            uid = f"gate-b:{row['source_sample_id']}:{report['merged_checkpoint_fingerprint'][:16]}"
            episode = asyncio.run(workflow.run_with_termination_handling(
                task={"sample_id": row["source_sample_id"], "question": question, "images": [image]}, uid=uid))
        termination = episode.termination_reason.value if episode.termination_reason else "unknown"
        trajectory = adapter.finalize_episode(termination=termination)
        if termination != "env_done" and trajectory.status == "success":
            trajectory.status, trajectory.error = "workflow_error", f"rLLM termination={termination}"
        report["checks"].update(roundtrip_checks(trajectory, backend.receipts, expected_crop_args=crop_arguments))
        report.update(rllm_episode_identity=episode.id, rllm_termination=termination,
                      derived_image_chain=trajectory.images, generation_receipts=backend.receipts,
                      second_generation_received_derived_image=report["checks"]["second_generation_received_derived_image"],
                      final_answer_present=bool(trajectory.final_answer), termination_status=trajectory.status,
                      tool_count=sum(turn.tool_call is not None for turn in trajectory.turns),
                      tool_calls=[turn.tool_call for turn in trajectory.turns if turn.tool_call is not None],
                      model_turn_count=len(adapter.state().assistant_outputs))
        artifact = {**trajectory.to_dict(), "question": question, "assistant_outputs": adapter.state().assistant_outputs,
                    "rllm_episode_identity": episode.id, "rllm_termination": termination,
                    "rllm_episode": episode.to_dict(), "merged_checkpoint_fingerprint": report["merged_checkpoint_fingerprint"],
                    "actor_adapter_fingerprint": actor["actor_adapter_fingerprint"],
                    "tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION, "generation_receipts": backend.receipts}
        atomic_json(reports / "trajectory.json", redact_secrets(artifact))
        report["checks"]["artifacts_written"] = True
        enter("vllm_shutdown")
        backend.close()
        backend, workflow = None, None
        report["checks"]["vllm_destroyed"] = True
        if not gate_passed(report["checks"]):
            failed = [name for name in GATE_B_CHECKS if report["checks"][name] is not True]
            raise RuntimeError(f"Gate B required checks failed: {failed}")
        report.update(passed=True, stage="complete", elapsed_seconds=time.monotonic() - started,
                      peak_cuda_memory={"allocated_bytes": torch.cuda.max_memory_allocated(0),
                                        "reserved_bytes": torch.cuda.max_memory_reserved(0)},
                      runtime_image_protocol_version=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION)
        atomic_json(report_path, redact_secrets(report))
        # The authoritative PASS marker is the last persistent success artifact.
        atomic_json(output / "gate_manifest.json", redact_secrets(report))
        print(f"Gate B PASS: {report_path}", flush=True)
        return 0
    except Exception as exc:
        report.update(passed=False, stage=stage, elapsed_seconds=time.monotonic() - started,
                      error={"type": type(exc).__name__, "message": str(exc)})
        # Output belongs to this fresh invocation. Revoke any partially published
        # PASS marker before failure-report I/O, which itself might also fail.
        (output / "gate_manifest.json").unlink(missing_ok=True)
        if adapter is not None:
            report["partial_trajectory"] = adapter.finalize_episode(termination="error").to_dict()
            report["error"]["origin"] = adapter.state().error_origin or stage
        if backend is not None:
            report["generation_receipts"] = backend.receipts
        if "episode" in locals():
            report["rllm_error"] = episode.info.get("error")
        if "torch" in locals() and torch.cuda.is_initialized():
            report["peak_cuda_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(0),
                                          "reserved_bytes": torch.cuda.max_memory_reserved(0)}
        atomic_json(report_path, redact_secrets(report))
        print(f"Gate B FAIL stage={stage}: {type(exc).__name__}: {redact_secrets(str(exc))}", flush=True)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as cleanup_error:
                report["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
                report["passed"] = False
                atomic_json(report_path, redact_secrets(report))
        gc.collect()
