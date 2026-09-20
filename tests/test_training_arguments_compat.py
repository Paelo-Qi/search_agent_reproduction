from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen from transformers v5.17.0 TrainingArguments. Keeping this scoped to
# the arguments used by this project makes any future addition require an
# explicit compatibility review instead of failing only when a GPU run starts.
SUPPORTED_PROJECT_ARGUMENTS_V5_17 = {
    "output_dir",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "lr_scheduler_type",
    "logging_steps",
    "save_strategy",
    "bf16",
    "tf32",
    "gradient_checkpointing",
    "dataloader_num_workers",
    "remove_unused_columns",
    "report_to",
    "ddp_find_unused_parameters",
    "seed",
    "data_seed",
    "logging_nan_inf_filter",
}


def training_arguments_keywords() -> set[str]:
    source_path = PROJECT_ROOT / "src" / "opensearch_vl_repro" / "training.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrainingArguments"
    ]
    assert len(calls) == 1
    assert all(keyword.arg is not None for keyword in calls[0].keywords)
    return {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}


def test_training_arguments_match_transformers_5_17_signature() -> None:
    keywords = training_arguments_keywords()
    assert keywords <= SUPPORTED_PROJECT_ARGUMENTS_V5_17
    assert {"overwrite_output_dir", "warmup_ratio", "logging_dir"}.isdisjoint(keywords)
    assert {"output_dir", "warmup_steps", "save_strategy"} <= keywords


def test_training_scale_and_manual_save_contract_remain_explicit() -> None:
    source = (
        PROJECT_ROOT / "src" / "opensearch_vl_repro" / "training.py"
    ).read_text(encoding="utf-8")
    assert 'save_strategy="no"' in source
    assert "unwrapped.save_pretrained(adapter_dir" in source
