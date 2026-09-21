from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class EvalSample:
    sample_id: str
    benchmark: str
    question: str
    images: list[Image.Image]


def _packed_payloads(value: Any) -> list[bytes]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return [bytes(value)]
    if not isinstance(value, list) or not value:
        raise ValueError("image_packed must contain at least one image")
    payloads: list[bytes] = []
    for item in value:
        data = item.get("data", item.get("bytes")) if isinstance(item, dict) else item
        if isinstance(data, str):
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            payloads.append(base64.b64decode("".join(data.split()), validate=True))
        elif isinstance(data, (bytes, bytearray, memoryview)):
            payloads.append(bytes(data))
        else:
            raise ValueError(f"unsupported packed image payload: {type(data).__name__}")
    return payloads


def read_eval_sample(path: str | Path, index: int) -> EvalSample:
    import pyarrow.parquet as pq

    parquet_path = Path(path).expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"frozen evaluation parquet is missing: {parquet_path}")
    parquet = pq.ParquetFile(parquet_path)
    if index < 0 or index >= parquet.metadata.num_rows:
        raise IndexError(
            f"evaluation index {index} is out of range for {parquet.metadata.num_rows} rows"
        )
    required = {"id", "benchmark", "question", "image_packed"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"evaluation parquet is missing fields: {missing}")
    # The frozen file contains only 300 rows. Reading these four columns keeps
    # the source immutable while avoiding unrelated answer/caption columns.
    row = parquet.read(columns=sorted(required)).slice(index, 1).to_pylist()[0]
    images: list[Image.Image] = []
    for payload in _packed_payloads(row["image_packed"]):
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            images.append(image.convert("RGB").copy())
    if not images:
        raise ValueError(f"evaluation sample {row['id']} contains no decodable images")
    return EvalSample(
        sample_id=str(row["id"]),
        benchmark=str(row["benchmark"]),
        question=str(row["question"]),
        images=images,
    )

