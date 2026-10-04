"""Read-only source validation and policy-task whitelist shared by Gate C/S3."""
from pathlib import Path


def model_task(row, images):
    if "trajectory_group_id" in row or row.get("prompt_id") != row.get("source_sample_id"):
        raise ValueError("prepared data must contain source identity, not runtime group identity")
    return {"sample_id": row["source_sample_id"], "question": row["question"], "images": images}


def load_source_images(row, root):
    from PIL import Image
    from opensearch_vl_repro.agent.reliability import image_sha256
    from .data import question_sha256, safe_image_relpath
    root = Path(root).resolve()
    if row.get("question_hash") != question_sha256(row["question"]):
        raise ValueError("frozen question identity mismatch")
    if len(row["image_relpaths"]) != len(row["image_hashes"]) or not row["image_relpaths"]:
        raise ValueError("source image identities missing")
    images = []
    for name, digest in zip(row["image_relpaths"], row["image_hashes"], strict=True):
        path = (root / safe_image_relpath(name)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError("source image missing/escaping root")
        with Image.open(path) as loaded:
            image = loaded.convert("RGB").copy()
        if image_sha256(image) != digest:
            raise ValueError("source image fingerprint mismatch")
        images.append(image)
    return images
