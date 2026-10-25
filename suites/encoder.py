"""Encoder suite: encoded E-AC-3/TrueHD against a master and optional DME reference."""

from __future__ import annotations

import tempfile
from pathlib import Path

from atmos_quality.cache import Cache
from atmos_quality.config import cache_dir, output_dir
from atmos_quality.layouts import get_layout
from atmos_quality.model import Asset, Observation, Pair, Render, RenderOptions
from atmos_quality.report import RunResult, write_run
from atmos_quality.suites.common import (
    choose_backend,
    default_audio_metrics,
    evaluate_audio,
    reference_metrics,
    render_asset,
)


def run_encoder(
    master: str | Path,
    candidate: str | Path,
    *,
    reference: str | Path | None = None,
    layouts: tuple[str, ...] = ("7.1.4", "binaural"),
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
    master_asset = Asset.from_path(master)
    candidate_asset = Asset.from_path(candidate)
    if master_asset.kind == "damf":
        raise ValueError(
            "encoder suite requires a TrueHD/E-AC-3 ground truth; "
            "use `aq cluster` for DAMF source/cluster comparisons"
        )
    reference_asset = Asset.from_path(reference) if reference is not None else None
    assets = {"master": master_asset, "candidate": candidate_asset}
    if reference_asset is not None:
        assets["reference"] = reference_asset

    cache = Cache(cache_dir(cache_root))
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="atmos_quality_encoder_")
        work = Path(temporary.name)
    else:
        work = Path(workdir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)

    renders: dict[str, Render] = {}
    pairs: list[Pair] = []
    observations: list[Observation] = []
    skipped: list[str] = []
    errors: list[str] = []
    options = RenderOptions(max_seconds=max_seconds)

    try:
        for layout_name in layouts:
            layout = get_layout(layout_name)
            try:
                backend = choose_backend([master_asset, candidate_asset], layout)
            except ValueError as exc:
                skipped.append(f"{layout.name}: {exc}")
                continue

            try:
                master_render = render_asset(
                    master_asset, layout, options, cache, backend, force=force
                )
                candidate_render = render_asset(
                    candidate_asset, layout, options, cache, backend, force=force
                )
            except Exception as exc:
                errors.append(f"{layout.name}: render failed: {exc}")
                continue

            renders[master_render.cache_key] = master_render
            renders[candidate_render.cache_key] = candidate_render

            selected = metrics or default_audio_metrics(
                layout,
                include_visqol=include_visqol,
                include_spauq=include_spauq,
            )
            try:
                pair, result = evaluate_audio(
                    master_render,
                    candidate_render,
                    layout,
                    selected,
                    work,
                    variant="master",
                    align=align,
                    max_lag=20000,
                )
            except Exception as exc:
                errors.append(f"{layout.name}: pair failed: {exc}")
                continue
            pairs.append(pair)
            observations.extend(result.observations)
            skipped.extend(f"{layout.name}: {item}" for item in result.skipped)
            errors.extend(f"{layout.name}: {item}" for item in result.errors)

            if reference_asset is not None:
                try:
                    reference_backend = choose_backend([reference_asset, candidate_asset], layout)
                    if reference_backend.name != backend.name:
                        raise ValueError(
                            "reference backend "
                            f"{reference_backend.name} != candidate backend {backend.name}"
                        )
                    reference_render = render_asset(
                        reference_asset, layout, options, cache, reference_backend, force=force
                    )
                    renders[reference_render.cache_key] = reference_render
                    ref_pair, ref_result = evaluate_audio(
                        reference_render,
                        candidate_render,
                        layout,
                        reference_metrics(layout),
                        work,
                        variant="reference",
                        align=align,
                        max_lag=20000,
                    )
                    pairs.append(ref_pair)
                    observations.extend(ref_result.observations)
                    skipped.extend(
                        f"{layout.name}/reference: {item}" for item in ref_result.skipped
                    )
                    errors.extend(f"{layout.name}/reference: {item}" for item in ref_result.errors)
                except Exception as exc:
                    errors.append(f"{layout.name}/reference: {exc}")

        if not pairs:
            raise RuntimeError("no comparable pair was produced")

        return write_run(
            suite="encoder",
            out_root=output_dir(out_root),
            assets=assets,
            renders=list(renders.values()),
            pairs=pairs,
            observations=observations,
            skipped=skipped,
            errors=errors,
            extra={"layouts": list(layouts)},
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
