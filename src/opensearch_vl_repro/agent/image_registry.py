from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageEntry:
    image_id: str
    value: Any
    kind: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageRegistry:
    """Per-trajectory mapping from deterministic img_n IDs to image values."""

    def __init__(self) -> None:
        self._entries: dict[str, ImageEntry] = {}
        self._next_index = 1

    @staticmethod
    def _validate_value(value: Any) -> Any:
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"image path does not exist: {path}")
            return path
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - Pillow is a project dependency.
            Image = None
        if Image is not None and isinstance(value, Image.Image):
            return value
        raise TypeError("image value must be a PIL.Image.Image or an existing local path")

    def _register(
        self,
        value: Any,
        *,
        kind: str,
        parent_id: str | None,
        image_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> str:
        normalized = self._validate_value(value)
        expected_id = f"img_{self._next_index}"
        assigned_id = image_id or expected_id
        if assigned_id in self._entries:
            raise ValueError(f"image ID already exists: {assigned_id}")
        if assigned_id != expected_id:
            raise ValueError(
                f"image IDs must be monotonically assigned; expected {expected_id}, got {assigned_id}"
            )
        if parent_id is not None and parent_id not in self._entries:
            raise KeyError(f"unknown parent image ID: {parent_id}")
        self._entries[assigned_id] = ImageEntry(
            image_id=assigned_id,
            value=normalized,
            kind=kind,
            parent_id=parent_id,
            metadata=dict(metadata or {}),
        )
        self._next_index += 1
        return assigned_id

    def register_initial_image(
        self,
        value: Any,
        *,
        image_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._register(
            value,
            kind="initial",
            parent_id=None,
            image_id=image_id,
            metadata=metadata,
        )

    def register_derived_image(
        self,
        value: Any,
        *,
        parent_id: str,
        image_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._register(
            value,
            kind="derived",
            parent_id=parent_id,
            image_id=image_id,
            metadata=metadata,
        )

    def get(self, image_id: str) -> Any:
        try:
            return self._entries[image_id].value
        except KeyError as exc:
            raise KeyError(f"unknown image ID: {image_id}") from exc

    def get_entry(self, image_id: str) -> ImageEntry:
        try:
            return self._entries[image_id]
        except KeyError as exc:
            raise KeyError(f"unknown image ID: {image_id}") from exc

    def exists(self, image_id: str) -> bool:
        return image_id in self._entries

    def list_images(self) -> tuple[ImageEntry, ...]:
        return tuple(self._entries.values())

