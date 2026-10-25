"""Object-distribution metrics for DAMF source/cluster masters."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atmos_quality.metrics.base import MetricContext, MetricUnavailable, register
from atmos_quality.model import Asset, Observation


@dataclass(frozen=True)
class SpatialPair:
    source: Asset
    candidate: Asset
    frames: int
    source_objects: int
    candidate_objects: int
    chamfer: np.ndarray
    source_to_candidate: np.ndarray
    candidate_to_source: np.ndarray
    source_height: np.ndarray
    candidate_height: np.ndarray
    source_height_energy: np.ndarray | None
    candidate_height_energy: np.ndarray | None
    source_active: np.ndarray
    candidate_active: np.ndarray


def _load_tracks(asset: Asset, hop: int):
    try:
        from atmos_cluster.damf.caf import read_caf_info
        from atmos_cluster.damf.frames import sample_timelines
        from atmos_cluster.damf.manifest import first_presentation, load_manifest
        from atmos_cluster.damf.metadata import load_events
        from atmos_cluster.damf.timeline import build_sources
    except ImportError as exc:
        raise MetricUnavailable(
            "atmos_cluster is required for spatial metrics; install the workspace package"
        ) from exc

    manifest = load_manifest(asset.path)
    presentation = first_presentation(manifest)
    directory = asset.path.parent
    _, events = load_events(directory / presentation["metadata"])
    audio = directory / presentation["audio"]
    caf = read_caf_info(audio)
    sources = build_sources(presentation, events)
    total_frames = int(caf.frames)
    if total_frames <= 0:
        raise MetricUnavailable(f"no audio frames found in {audio}")
    n_frames = int(math.ceil(total_frames / hop))
    tracks = sample_timelines(sources.sources, n_frames, hop, total_frames)
    return tracks, sources.n_sources, sources.channels, audio, total_frames


def _frame_channel_power(
    audio: Path,
    channels: list[int],
    n_frames: int,
    hop: int,
    workdir: Path,
) -> np.ndarray:
    from atmos_cluster.audio import decode_caf_to_f32
    from atmos_cluster.damf.caf import read_caf_info

    caf = read_caf_info(audio)
    raw = workdir / f"{audio.name}.power.f32"
    try:
        decode_caf_to_f32(audio, raw)
        samples = np.memmap(raw, dtype="<f4", mode="r", shape=(caf.frames, caf.channels))
        power = np.zeros((n_frames, len(channels)), dtype=np.float64)
        for frame in range(n_frames):
            start = frame * hop
            end = min(start + hop, caf.frames)
            if end <= start:
                break
            block = np.asarray(samples[start:end][:, channels], dtype=np.float64)
            power[frame] = np.square(block).mean(axis=0)
        del samples
        return power
    finally:
        raw.unlink(missing_ok=True)


def _energy_height(tracks, power: np.ndarray) -> np.ndarray:
    count = min(tracks.n_frames, power.shape[0])
    height = np.full(count, np.nan, dtype=np.float64)
    for frame in range(count):
        active = tracks.active[frame]
        weights = power[frame]
        total = float(weights[active].sum())
        if active.any() and total > 0.0:
            height[frame] = float(
                np.sum(weights[active] * tracks.positions[frame, active, 2]) / total
            )
    return height


def _weighted_height(positions: np.ndarray, weights: np.ndarray) -> float:
    if len(positions) == 0 or weights.sum() <= 0.0:
        return float("nan")
    return float(np.sum(weights * (positions[:, 2] > 0.25)) / weights.sum())


def build_spatial_pair(
    source: Asset,
    candidate: Asset,
    *,
    hop: int = 1536,
    workdir: Path | None = None,
) -> SpatialPair:
    if source.kind != "damf" or candidate.kind != "damf":
        raise MetricUnavailable("spatial metrics require two DAMF masters")
    source_tracks, source_objects, source_channels, source_audio, _ = _load_tracks(source, hop)
    candidate_tracks, candidate_objects, candidate_channels, candidate_audio, _ = _load_tracks(
        candidate, hop
    )
    frames = min(source_tracks.n_frames, candidate_tracks.n_frames)
    if frames <= 0:
        raise MetricUnavailable("no overlapping metadata frames")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="atmos_quality_spatial_")
        work = Path(temporary.name)
    else:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)

    chamfer: list[float] = []
    source_to_candidate: list[float] = []
    candidate_to_source: list[float] = []
    source_height: list[float] = []
    candidate_height: list[float] = []
    source_active: list[int] = []
    candidate_active: list[int] = []

    try:
        for frame in range(frames):
            source_mask = source_tracks.active[frame]
            candidate_mask = candidate_tracks.active[frame]
            source_positions = source_tracks.positions[frame][source_mask]
            candidate_positions = candidate_tracks.positions[frame][candidate_mask]
            if len(source_positions) == 0 or len(candidate_positions) == 0:
                continue

            source_weights = np.maximum(source_tracks.importance[frame][source_mask], 0.0)
            candidate_weights = np.maximum(candidate_tracks.importance[frame][candidate_mask], 0.0)
            if source_weights.sum() <= 0.0:
                source_weights = np.ones(len(source_positions), dtype=np.float64)
            if candidate_weights.sum() <= 0.0:
                candidate_weights = np.ones(len(candidate_positions), dtype=np.float64)

            distances = np.linalg.norm(
                source_positions[:, None, :] - candidate_positions[None, :, :], axis=2
            )
            s2c = float(np.sum(distances.min(axis=1) * source_weights) / source_weights.sum())
            c2s = float(np.sum(distances.min(axis=0) * candidate_weights) / candidate_weights.sum())
            source_to_candidate.append(s2c)
            candidate_to_source.append(c2s)
            chamfer.append(0.5 * (s2c + c2s))
            source_height.append(_weighted_height(source_positions, source_weights))
            candidate_height.append(_weighted_height(candidate_positions, candidate_weights))
            source_active.append(int(source_mask.sum()))
            candidate_active.append(int(candidate_mask.sum()))

        if not chamfer:
            raise MetricUnavailable("no frames with active objects in both masters")

        source_height_energy: np.ndarray | None = None
        candidate_height_energy: np.ndarray | None = None
        try:
            source_power = _frame_channel_power(
                source_audio, source_channels, source_tracks.n_frames, hop, work
            )
            if candidate_audio == source_audio and candidate_channels == source_channels:
                candidate_power = source_power
            else:
                candidate_power = _frame_channel_power(
                    candidate_audio, candidate_channels, candidate_tracks.n_frames, hop, work
                )
            source_height_energy = _energy_height(source_tracks, source_power)
            candidate_height_energy = _energy_height(candidate_tracks, candidate_power)
        except Exception:
            source_height_energy = None
            candidate_height_energy = None

        return SpatialPair(
            source=source,
            candidate=candidate,
            frames=len(chamfer),
            source_objects=source_objects,
            candidate_objects=candidate_objects,
            chamfer=np.asarray(chamfer, dtype=np.float64),
            source_to_candidate=np.asarray(source_to_candidate, dtype=np.float64),
            candidate_to_source=np.asarray(candidate_to_source, dtype=np.float64),
            source_height=np.asarray(source_height, dtype=np.float64),
            candidate_height=np.asarray(candidate_height, dtype=np.float64),
            source_height_energy=source_height_energy,
            candidate_height_energy=candidate_height_energy,
            source_active=np.asarray(source_active, dtype=np.float64),
            candidate_active=np.asarray(candidate_active, dtype=np.float64),
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def _require_spatial(context: MetricContext) -> SpatialPair:
    if context.spatial is None:
        raise MetricUnavailable("no spatial pair")
    return context.spatial


@register("spatial.chamfer")
def spatial_chamfer(context: MetricContext) -> list[Observation]:
    spatial = _require_spatial(context)
    values = spatial.chamfer
    return [
        Observation(
            "spatial.chamfer.mean",
            float(np.mean(values)),
            unit="room",
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.chamfer.p95",
            float(np.percentile(values, 95)),
            unit="room",
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.chamfer.max",
            float(np.max(values)),
            unit="room",
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.chamfer.source_to_candidate",
            float(np.mean(spatial.source_to_candidate)),
            unit="room",
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.chamfer.candidate_to_source",
            float(np.mean(spatial.candidate_to_source)),
            unit="room",
            scope="aggregate",
            variant=context.variant,
        ),
    ]


@register("spatial.height")
def spatial_height(context: MetricContext) -> list[Observation]:
    spatial = _require_spatial(context)
    source = float(np.nanmean(spatial.source_height))
    candidate = float(np.nanmean(spatial.candidate_height))
    observations = [
        Observation(
            "spatial.height.source",
            source,
            scope="aggregate",
            variant=context.variant,
            note="metadata importance weights; not comparable across masters",
        ),
        Observation(
            "spatial.height.candidate",
            candidate,
            scope="aggregate",
            variant=context.variant,
            note="metadata importance weights; not comparable across masters",
        ),
        Observation(
            "spatial.height.delta",
            candidate - source,
            scope="aggregate",
            variant=context.variant,
            note="metadata importance weights; not comparable across masters",
        ),
    ]
    if spatial.source_height_energy is not None and spatial.candidate_height_energy is not None:
        source_energy = float(np.nanmean(spatial.source_height_energy))
        candidate_energy = float(np.nanmean(spatial.candidate_height_energy))
        observations.extend(
            [
                Observation(
                    "spatial.height.energy.source",
                    source_energy,
                    scope="aggregate",
                    variant=context.variant,
                    note="audio-power weighted",
                ),
                Observation(
                    "spatial.height.energy.candidate",
                    candidate_energy,
                    scope="aggregate",
                    variant=context.variant,
                    note="audio-power weighted",
                ),
                Observation(
                    "spatial.height.energy.delta",
                    candidate_energy - source_energy,
                    scope="aggregate",
                    variant=context.variant,
                    note="audio-power weighted",
                ),
            ]
        )
    return observations


@register("spatial.occupancy")
def spatial_occupancy(context: MetricContext) -> list[Observation]:
    spatial = _require_spatial(context)
    return [
        Observation(
            "spatial.objects.source",
            float(spatial.source_objects),
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.objects.candidate",
            float(spatial.candidate_objects),
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.active.source",
            float(np.mean(spatial.source_active)),
            scope="aggregate",
            variant=context.variant,
        ),
        Observation(
            "spatial.active.candidate",
            float(np.mean(spatial.candidate_active)),
            scope="aggregate",
            variant=context.variant,
        ),
    ]
