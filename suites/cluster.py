"""Cluster suite: source DAMF master vs clustered DAMF master."""

from __future__ import annotations

import tempfile
from pathlib import Path

from atmos_quality.cache import Cache
from atmos_quality.config import cache_dir, output_dir
from atmos_quality.layouts import get_layout
from atmos_quality.metrics.base import MetricContext, evaluate_metrics
from atmos_quality.metrics.spatial import build_spatial_pair
from atmos_quality.model import Asset, Observation, Pair, Render, RenderOptions
from atmos_quality.report import RunResult, write_run
from atmos_quality.suites.common import (
    choose_backend,
    default_audio_metrics,
    evaluate_audio,
    render_asset,
    tag_metadata,
)

SPATIAL_METRICS = ["spatial.chamfer", "spatial.height", "spatial.occupancy"]


def run_cluster(
    source: str | Path,
    candidate: str | Path,
    *,
    layouts: tuple[str, ...] = ("7.1.4",),
    metrics: list[str] | None = None,
    out_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    workdir: str | Path | None = None,
    max_seconds: float = 0.0,
    force: bool = False,
    align: bool = True,
    include_visqol: bool = True,
    include_spauq: bool = True,
) -> RunResult:
    if not layouts:
        raise ValueError("at least one layout is required")
    source_asset = Asset.from_path(source)
    candidate_asset = Asset.from_path(candidate)
    if source_asset.kind != "damf" or candidate_asset.kind != "damf":
        raise ValueError("cluster suite requires two .atmos/.damf masters")

    cache = Cache(cache_dir(cache_root))
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="atmos_quality_cluster_")
        work = Path(temporary.name)
    else:
        work = Path(workdir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)

    selected = metrics or (
        default_audio_metrics(
            get_layout(layouts[0]),
            include_visqol=include_visqol,
            include_spauq=include_spauq,
        )
        + SPATIAL_METRICS
    )
    audio_metrics = [name for name in selected if not name.startswith("spatial.")]
    spatial_metrics = [name for name in selected if name.startswith("spatial.")]

    renders: dict[str, Render] = {}
    pairs: list[Pair] = []
    observations: list[Observation] = []
    skipped: list[str] = []
    errors: list[str] = []
    options = RenderOptions(max_seconds=max_seconds)
    spatial_pair = None
    extra: dict[str, object] = {"layouts": list(layouts)}

    try:
        if spatial_metrics:
            try:
                spatial_pair = build_spatial_pair(
                    source_asset, candidate_asset, workdir=work / "spatial"
                )
                context = MetricContext(
                    spatial=spatial_pair,
                    workdir=work / "metadata",
                    variant="source",
                    config={},
                )
                result = evaluate_metrics(spatial_metrics, context)
                observations.extend(tag_metadata(result.observations))
                skipped.extend(f"metadata: {item}" for item in result.skipped)
                errors.extend(f"metadata: {item}" for item in result.errors)
                extra["source_objects"] = spatial_pair.source_objects
                extra["candidate_objects"] = spatial_pair.candidate_objects
            except Exception as exc:
                skipped.append(f"spatial: {exc}")

        for layout_name in layouts:
            layout = get_layout(layout_name)
            try:
                backend = choose_backend([source_asset, candidate_asset], layout)
            except ValueError as exc:
                skipped.append(f"{layout.name}: {exc}")
                continue

            try:
                source_render = render_asset(
                    source_asset, layout, options, cache, backend, force=force
                )
                candidate_render = render_asset(
                    candidate_asset, layout, options, cache, backend, force=force
                )
            except Exception as exc:
                errors.append(f"{layout.name}: render failed: {exc}")
                continue

            renders[source_render.cache_key] = source_render
            renders[candidate_render.cache_key] = candidate_render

            if not audio_metrics:
                continue
            try:
                pair, result = evaluate_audio(
                    source_render,
                    candidate_render,
                    layout,
                    audio_metrics,
                    work,
                    variant="source",
                    align=align,
                    max_lag=4800,
                )
            except Exception as exc:
                errors.append(f"{layout.name}: pair failed: {exc}")
                continue
            pairs.append(pair)
            observations.extend(result.observations)
            skipped.extend(f"{layout.name}: {item}" for item in result.skipped)
            errors.extend(f"{layout.name}: {item}" for item in result.errors)

        if not pairs and not observations:
            raise RuntimeError("no comparable pair or spatial metric was produced")

        return write_run(
            suite="cluster",
            out_root=output_dir(out_root),
            assets={"source": source_asset, "candidate": candidate_asset},
            renders=list(renders.values()),
            pairs=pairs,
            observations=observations,
            skipped=skipped,
            errors=errors,
            extra=extra,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
