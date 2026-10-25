"""Shared suite helpers: backend choice, rendering and pair evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from atmos_quality.align import make_pair
from atmos_quality.backends.base import RenderBackend
from atmos_quality.backends.cavern import CavernBackend
from atmos_quality.backends.dolby_gst import DolbyGstBackend
from atmos_quality.cache import Cache
from atmos_quality.layouts import Layout
from atmos_quality.metrics.base import MetricContext, MetricRun, evaluate_metrics
from atmos_quality.model import Asset, Observation, Pair, Render, RenderOptions


def choose_backend(assets: list[Asset], layout: Layout) -> RenderBackend:
    kinds = {asset.kind for asset in assets}
    if layout.name == "binaural":
        backend = DolbyGstBackend()
        if all(backend.supports(asset, layout) for asset in assets):
            return backend
        raise ValueError("binaural rendering requires E-AC-3 or TrueHD assets")
    if kinds <= {"ec3", "truehd"}:
        backend = DolbyGstBackend()
        if all(backend.supports(asset, layout) for asset in assets):
            return backend
        raise ValueError(f"Dolby backend cannot render layout {layout.name}")
    if kinds == {"damf"}:
        backend = CavernBackend()
        if all(backend.supports(asset, layout) for asset in assets):
            return backend
        raise ValueError(f"Cavern backend cannot render layout {layout.name}")
    raise ValueError(
        "one comparison cannot mix DAMF with encoded assets; "
        "render the DAMF master through the same backend first"
    )


def render_asset(
    asset: Asset,
    layout: Layout,
    options: RenderOptions,
    cache: Cache,
    backend: RenderBackend,
    *,
    force: bool,
) -> Render:
    return backend.render(asset, layout, options, cache, force=force)


def evaluate_audio(
    reference: Render,
    candidate: Render,
    layout: Layout,
    metrics: list[str],
    workdir: Path,
    *,
    variant: str,
    align: bool = True,
    start_s: float = 30.0,
    duration_s: float = 10.0,
    max_lag: int = 20000,
    level_match: str = "global",
) -> tuple[Pair, MetricRun]:
    pair = make_pair(
        reference,
        candidate,
        level_match=level_match,
        align=align,
        start_s=start_s,
        duration_s=duration_s,
        max_lag=max_lag,
        variant=variant,
    )
    metric_workdir = workdir / layout.name / variant
    metric_workdir.mkdir(parents=True, exist_ok=True)
    context = MetricContext(
        pair=pair,
        workdir=metric_workdir,
        variant=variant,
        config={"layout": layout.name},
    )
    result = evaluate_metrics(metrics, context)
    result.observations = [
        replace(observation, layout=layout.name) for observation in result.observations
    ]
    return pair, result


def default_audio_metrics(
    layout: Layout,
    *,
    include_visqol: bool,
    include_spauq: bool,
) -> list[str]:
    metrics = ["signal.corr", "signal.err_db", "signal.rms"]
    if include_visqol:
        metrics.append("perceptual.visqol")
    if include_spauq and layout.channels >= 6:
        metrics.append("perceptual.spauq")
    return metrics


def reference_metrics(layout: Layout | None = None) -> list[str]:
    metrics = ["signal.corr", "signal.err_db", "signal.rms"]
    if layout is not None and layout.channels >= 6:
        metrics.append("perceptual.spauq")
    return metrics


def tag_metadata(observations: list[Observation]) -> list[Observation]:
    return [replace(observation, layout="metadata") for observation in observations]
