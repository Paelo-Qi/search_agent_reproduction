"""Remove the known evaluation question wrapper from model-facing text only."""

from __future__ import annotations

import re


_EVAL_IMAGE_QUESTION = re.compile(
    r"\A\s*image_id\s*:\s*"
    r"[^\s/\\:]+\.(?:jpg|jpeg|png|webp|bmp|gif|tif|tiff)"
    r"\s+Question\s*:\s*(?P<question>\S(?:.|\n)*?)\s*\Z",
    re.IGNORECASE,
)


def normalize_model_question(question: str) -> str:
    """Strip only `image_id: <filename> Question: <text>`; preserve other text."""
    match = _EVAL_IMAGE_QUESTION.fullmatch(question)
    return match.group("question") if match else question
