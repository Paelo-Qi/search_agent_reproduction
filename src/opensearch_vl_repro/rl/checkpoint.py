"""RL lineage built from the existing verified SFT adapter identity."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.inference.adapter import adapter_identity
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_train_plan import STAGE_NAMES


@dataclass(frozen=True)
class RLLineage:
    schema_version: int
    base_model: str
    base_revision: str
    sft_adapter_fingerprint: str
    sft_adapter_config_fingerprint: str
    sft_checkpoint_metadata_fingerprint: str
    sft_stage: str
    sft_lineage: tuple[str, ...]
    runtime_tool_protocol_version: str
    rl_run_id: str
    rl_adapter_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sft_lineage": list(self.sft_lineage)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RLLineage":
        result = cls(**{**value, "sft_lineage": tuple(value["sft_lineage"])})
        result.validate()
        return result

    def validate(self) -> None:
        if (self.schema_version != 1 or not self.base_model or not self.base_revision
                or not self.rl_run_id
                or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in (
                    self.sft_adapter_fingerprint, self.sft_adapter_config_fingerprint,
                    self.sft_checkpoint_metadata_fingerprint))
                or self.runtime_tool_protocol_version != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
                or self.sft_stage not in STAGE_NAMES
                or not self.sft_lineage or self.sft_lineage[-1] != self.sft_stage):
            raise ValueError("invalid RL/SFT checkpoint lineage")
        if self.rl_adapter_fingerprint is not None and re.fullmatch(r"[0-9a-f]{64}", self.rl_adapter_fingerprint) is None:
            raise ValueError("invalid RL adapter fingerprint")


def validate_sft_overlap_scope(lineage: RLLineage, sft_shards: list[str]) -> None:
    """The overlap audit must cover exactly the shards seen by this adapter."""
    lineage.validate()
    if sft_shards != list(lineage.sft_lineage):
        raise ValueError("SFT overlap shard scope differs from RL initialization adapter lineage")


def build_rl_run_manifest(lineage: RLLineage, *, config: dict[str, Any],
                          data_manifest_sha256: str | None) -> dict[str, Any]:
    lineage.validate()
    if data_manifest_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", data_manifest_sha256) is None:
        raise ValueError("invalid RL data manifest fingerprint")
    payload = {"schema_version": 1, "lineage": lineage.to_dict(),
               "config_sha256": canonical_json_sha256(config),
               "data_manifest_sha256": data_manifest_sha256}
    payload["run_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def build_rl_lineage(*, config: dict[str, Any], sft_config: dict[str, Any],
                     adapter_path: str | Path, run_id: str) -> RLLineage:
    model = sft_config["model"]
    lora = sft_config["lora"]
    if model["freeze_vision_tower"] is not True or model["freeze_multimodal_projector"] is not True:
        raise ValueError("RL requires SFT frozen vision tower and projector")
    if config["model"]["continue_from_sft_adapter"] is not True:
        raise ValueError("RL cannot start from an empty LoRA")
    identity = adapter_identity(adapter_path, base_model=model["name_or_path"],
                                base_revision=model["revision"])
    adapter_config = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    if (adapter_config.get("peft_type") != "LORA"
            or adapter_config.get("r") != lora["rank"]
            or adapter_config.get("lora_alpha") != lora["alpha"]
            or float(adapter_config.get("lora_dropout", -1)) != float(lora["dropout"])
            or set(adapter_config.get("target_modules", [])) != set(lora["target_modules"])):
        raise ValueError("RL LoRA parameters differ from SFT adapter")
    metadata = json.loads((Path(adapter_path).parent / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("stage_complete") is not True:
        raise ValueError("RL requires a complete SFT stage checkpoint")
    stage = identity["training_cumulative_stage"]
    if stage != config["model"]["sft_stage"]:
        raise ValueError("RL SFT stage differs from plan")
    lineage = identity["source_checkpoint_lineage"]
    result = RLLineage(1, model["name_or_path"], model["revision"],
                       identity["adapter_fingerprint"], identity["adapter_config_fingerprint"],
                       identity["checkpoint_metadata_fingerprint"], stage, tuple(lineage),
                       RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION, run_id)
    result.validate()
    return result
