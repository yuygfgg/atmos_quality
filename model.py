"""Core data model for assets, renders, pairs and observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from atmos_quality import audio
from atmos_quality.layouts import Layout


def infer_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".atmos", ".damf"}:
        return "damf"
    if suffix in {".ec3", ".eac3"}:
        return "ec3"
    if suffix in {".mlp", ".thd", ".truehd"}:
        return "truehd"
    if suffix in {".wav", ".wave"}:
        return "wav"
    if suffix in {".f32", ".raw", ".pcm"}:
        return "raw"
    raise ValueError(f"cannot infer asset kind from suffix: {path.suffix}")


@dataclass(frozen=True)
class Asset:
    path: Path
    kind: str
    digest: str = ""
    sample_rate: int | None = None
    channels: int | None = None
    objects: int | None = None

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        object.__setattr__(self, "path", path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not self.digest:
            object.__setattr__(self, "digest", audio.fingerprint(path))

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        kind: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        objects: int | None = None,
    ) -> Asset:
        resolved = Path(path).expanduser().resolve()
        return cls(
            path=resolved,
            kind=kind or infer_kind(resolved),
            sample_rate=sample_rate,
            channels=channels,
            objects=objects,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "digest": self.digest,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "objects": self.objects,
        }


@dataclass(frozen=True)
class RenderOptions:
    max_seconds: float = 0.0
    gain_db: float = 0.0
    drc_suppress: bool = True
    limiter: bool = False
    drop_delay: bool = True
    update_rate: int = 256
    extra: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_seconds": self.max_seconds,
            "gain_db": self.gain_db,
            "drc_suppress": self.drc_suppress,
            "limiter": self.limiter,
            "drop_delay": self.drop_delay,
            "update_rate": self.update_rate,
            "extra": dict(self.extra),
        }

    def cache_parts(self) -> tuple[str, ...]:
        return (
            f"max={self.max_seconds:.6f}",
            f"gain={self.gain_db:.6f}",
            f"drc={int(self.drc_suppress)}",
            f"lim={int(self.limiter)}",
            f"delay={int(self.drop_delay)}",
            f"rate={self.update_rate}",
            repr(tuple(sorted(self.extra))),
        )


@dataclass(frozen=True)
class Render:
    path: Path
    format: str
    labels: tuple[str, ...]
    sample_rate: int
    samples: int
    backend: str
    asset: Asset
    layout: Layout
    options: RenderOptions
    cache_key: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return len(self.labels)

    def open(self) -> np.ndarray:
        if self.format == "f32":
            return audio.load_f32(self.path, self.channels)
        if self.format == "wav":
            rate, data = audio.load_wav(self.path)
            if rate != self.sample_rate:
                raise ValueError(
                    f"sample rate mismatch for {self.path}: {rate} != {self.sample_rate}"
                )
            return data
        raise ValueError(f"unknown render format: {self.format}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "labels": list(self.labels),
            "sample_rate": self.sample_rate,
            "samples": self.samples,
            "duration_s": self.samples / self.sample_rate if self.sample_rate else None,
            "backend": self.backend,
            "asset": self.asset.to_dict(),
            "layout": self.layout.name,
            "options": self.options.to_dict(),
            "cache_key": self.cache_key,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Alignment:
    lag: int
    gain: float
    method: str
    confidence: float

    @property
    def gain_db(self) -> float:
        return float(20.0 * np.log10(max(self.gain, 1e-30)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lag_samples": self.lag,
            "gain": self.gain,
            "gain_db": self.gain_db,
            "method": self.method,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Pair:
    reference: Render
    candidate: Render
    ref: np.ndarray
    cand: np.ndarray
    labels: tuple[str, ...]
    sample_rate: int
    alignment: Alignment
    variant: str = "master"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        return int(self.ref.shape[0])

    @property
    def channels(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "layout": self.reference.layout.name,
            "reference": self.reference.asset.to_dict(),
            "candidate": self.candidate.asset.to_dict(),
            "labels": list(self.labels),
            "sample_rate": self.sample_rate,
            "samples": self.samples,
            "duration_s": self.samples / self.sample_rate if self.sample_rate else None,
            "alignment": self.alignment.to_dict(),
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Observation:
    metric: str
    value: float | None
    unit: str = ""
    scope: str = "aggregate"
    channel: str | None = None
    variant: str = ""
    segment: tuple[int, int] | None = None
    note: str = ""
    layout: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "scope": self.scope,
            "channel": self.channel,
            "variant": self.variant,
            "segment": list(self.segment) if self.segment else None,
            "note": self.note,
            "layout": self.layout,
        }
