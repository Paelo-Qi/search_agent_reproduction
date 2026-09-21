from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from opensearch_vl_repro.inference.config import load_inference_config
from opensearch_vl_repro.inference.eval_reader import read_eval_sample
from opensearch_vl_repro.inference.model_loader import load_inference_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def packed_image() -> str:
    image = Image.new("RGB", (5, 4), color=(20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return json.dumps(
        [{"filename": "fixture.png", "data": base64.b64encode(buffer.getvalue()).decode()}]
    )


def test_inference_config_is_pinned_and_deterministic() -> None:
    config = load_inference_config(PROJECT_ROOT / "configs" / "eval_4b.yaml")
    assert config.model_name_or_path == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.revision == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert config.dtype == "bfloat16"
    assert config.device == "cuda:0"
    assert config.do_sample is False
    assert config.temperature == 0
    assert config.max_agent_turns > 0
    assert config.generation_kwargs() == {"max_new_tokens": 256, "do_sample": False}
    assert config.data_path == (PROJECT_ROOT / "data" / "eval" / "combined_eval_300.parquet")


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    __version__ = "fake-torch"
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"
    cuda = FakeCuda()
    version = SimpleNamespace(cuda=None)


class FakeTransformers:
    __version__ = "5.17.0"


class FakeProcessorClass:
    kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.kwargs = {"args": args, **kwargs}
        return object()


class FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.grad_enabled = True

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, enabled: bool):
        self.grad_enabled = enabled
        return self


class FakeModelClass:
    kwargs = None
    instance = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.kwargs = {"args": args, **kwargs}
        cls.instance = FakeModel()
        return cls.instance


def test_4b_loader_path_uses_eval_no_grad_and_transformers_5_dtype() -> None:
    config = replace(
        load_inference_config(PROJECT_ROOT / "configs" / "eval_4b.yaml"),
        device="cpu",
    )
    bundle = load_inference_bundle(
        config,
        model_class=FakeModelClass,
        processor_class=FakeProcessorClass,
        torch_module=FakeTorch,
        transformers_module=FakeTransformers,
    )
    assert FakeModelClass.kwargs["dtype"] == "bf16"
    assert "torch_dtype" not in FakeModelClass.kwargs
    assert FakeModelClass.kwargs["device_map"] == "cpu"
    assert FakeModelClass.kwargs["revision"] == config.revision
    assert FakeProcessorClass.kwargs["revision"] == config.revision
    assert bundle.model.training is False
    assert bundle.model.grad_enabled is False
    assert bundle.environment["transformers_version"] == "5.17.0"
    assert bundle.environment["model"] == "Qwen/Qwen3-VL-4B-Instruct"


def test_eval_reader_extracts_question_and_image_without_modifying_input(tmp_path: Path) -> None:
    path = tmp_path / "eval.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "id": "fixture-1",
                "benchmark": "simplevqa",
                "question": "What is shown?",
                "image_packed": packed_image(),
            }
        ]
    )
    pq.write_table(table, path)
    before = path.read_bytes()
    sample = read_eval_sample(path, 0)
    assert sample.sample_id == "fixture-1"
    assert sample.benchmark == "simplevqa"
    assert sample.question == "What is shown?"
    assert len(sample.images) == 1
    assert sample.images[0].size == (5, 4)
    assert path.read_bytes() == before
    with pytest.raises(IndexError, match="out of range"):
        read_eval_sample(path, 1)


def test_actual_frozen_combined_eval_sample_is_readable() -> None:
    path = PROJECT_ROOT / "data" / "eval" / "combined_eval_300.parquet"
    if not path.is_file():
        pytest.skip("frozen combined_eval_300.parquet is not materialized in this checkout")
    before = path.stat().st_mtime_ns
    sample = read_eval_sample(path, 0)
    assert sample.sample_id
    assert sample.benchmark in {"simplevqa", "mmsearch", "vdr_bench"}
    assert sample.question.strip()
    assert sample.images and all(image.width > 0 and image.height > 0 for image in sample.images)
    assert path.stat().st_mtime_ns == before
