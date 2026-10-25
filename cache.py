"""Content/recipe-addressed artifact cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atmos_quality.layouts import Layout
from atmos_quality.model import Asset, RenderOptions


def _hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class Cache:
    root: Path

    def __post_init__(self) -> None:
        if self.root is None:
            raise ValueError("cache root is not configured")
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, key: str, suffix: str) -> Path:
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}{suffix}"

    def render_key(
        self,
        backend: str,
        backend_version: str,
        asset: Asset,
        layout: Layout,
        options: RenderOptions,
    ) -> str:
        return _hash(
            "render",
            backend,
            backend_version,
            asset.digest,
            asset.path,
            layout.name,
            options.cache_parts(),
        )

    def build_key(self, *parts: object) -> str:
        return _hash("build", *parts)


def read_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"invalid metadata: {path}")
    return value


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temporary.replace(path)
