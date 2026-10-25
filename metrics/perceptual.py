"""Optional perceptual metrics: ViSQOL v3 and SPAUQ."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from atmos_quality.audio import write_wav_mono
from atmos_quality.metrics.base import MetricContext, MetricUnavailable, register, require_pair
from atmos_quality.model import Observation


@register("perceptual.visqol")
def visqol(context: MetricContext) -> list[Observation]:
    pair = require_pair(context)
    if pair.samples < pair.sample_rate // 2:
        raise MetricUnavailable("audio is shorter than 0.5 s")
    try:
        from visqol import VisqolApi
    except ImportError as exc:
        raise MetricUnavailable(
            "visqol is not installed; pip install 'visqol-python[accel,fftw]'"
        ) from exc

    root = Path(context.workdir or tempfile.mkdtemp(prefix="aq_visqol_")) / "visqol"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, Path, str]] = []
    for index, label in enumerate(pair.labels):
        ref_path = root / f"{index:02d}_{label}_ref.wav"
        deg_path = root / f"{index:02d}_{label}_deg.wav"
        reference = np.asarray(pair.ref[:, index], dtype=np.float32)
        candidate = np.asarray(pair.cand[:, index], dtype=np.float32) * pair.alignment.gain
        write_wav_mono(ref_path, reference, pair.sample_rate)
        write_wav_mono(deg_path, candidate, pair.sample_rate)
        paths.append((ref_path, deg_path, label))

    api = VisqolApi()
    api.create(mode="audio")
    # Sequential by default: parallel=True uses multiprocessing spawn, which
    # is unsafe when the package is called from an embedded interpreter.
    results = api.measure_batch(
        [(str(ref), str(deg)) for ref, deg, _ in paths],
        max_workers=1,
        parallel=False,
    )

    observations: list[Observation] = []
    mos: list[float] = []
    vnsim: list[float] = []
    failures = 0
    for (_, _, label), result in zip(paths, results, strict=False):
        if isinstance(result, Exception):
            failures += 1
            observations.append(
                Observation(
                    "perceptual.visqol.moslqo",
                    None,
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                    note=f"visqol error: {result}",
                )
            )
            continue
        mos_value = float(result.moslqo)
        vnsim_value = float(result.vnsim)
        mos.append(mos_value)
        vnsim.append(vnsim_value)
        observations.extend(
            [
                Observation(
                    "perceptual.visqol.moslqo",
                    mos_value,
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                ),
                Observation(
                    "perceptual.visqol.vnsim",
                    vnsim_value,
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                ),
            ]
        )
    if not mos:
        raise RuntimeError(f"all ViSQOL channels failed ({failures})")
    observations.extend(
        [
            Observation(
                "perceptual.visqol.moslqo.mean",
                float(np.mean(mos)),
                scope="aggregate",
                variant=context.variant,
            ),
            Observation(
                "perceptual.visqol.moslqo.min",
                float(np.min(mos)),
                scope="aggregate",
                variant=context.variant,
            ),
        ]
    )
    return observations


@register("perceptual.spauq")
def spauq(context: MetricContext) -> list[Observation]:
    pair = require_pair(context)
    if pair.channels < 2:
        raise MetricUnavailable("SPAUQ requires at least two channels")
    try:
        from spauq.core.metrics import spauq_eval
    except Exception as exc:
        raise MetricUnavailable(
            "spauq is not installed (GPL-3.0, Python <3.11); "
            "install from github.com/kwatcharasupat/spauq"
        ) from exc

    chunk_seconds = float(context.config.get("spauq_chunk_seconds", 10.0))
    chunk_samples = max(int(chunk_seconds * pair.sample_rate), pair.sample_rate // 2)
    ssr_values: list[float] = []
    srr_values: list[float] = []
    failures = 0
    total_chunks = 0
    for start in range(0, pair.samples, chunk_samples):
        stop = min(start + chunk_samples, pair.samples)
        if stop - start < pair.sample_rate // 4:
            continue
        total_chunks += 1
        reference = np.asarray(pair.ref[start:stop], dtype=np.float32).T
        candidate = np.asarray(pair.cand[start:stop], dtype=np.float32)
        if pair.alignment.gain != 1.0:
            candidate = candidate * pair.alignment.gain
        try:
            result = spauq_eval(
                reference=reference,
                estimate=candidate.T,
                fs=pair.sample_rate,
                align_mode=None,
                scale_mode=None,
                verbose=False,
            )
        except Exception:
            failures += 1
            continue
        ssr = result.get("SSR")
        srr = result.get("SRR")
        if ssr is not None:
            ssr_values.append(float(np.median(np.asarray(ssr, dtype=np.float64))))
        if srr is not None:
            srr_values.append(float(np.median(np.asarray(srr, dtype=np.float64))))

    if not ssr_values and not srr_values:
        raise RuntimeError(f"all SPAUQ chunks failed ({failures}/{total_chunks})")
    note = f"{len(ssr_values)} chunks"
    if failures:
        note += f", {failures} failed"
    observations = [
        Observation(
            "perceptual.spauq.ssr",
            float(np.median(ssr_values)),
            unit="dB",
            scope="aggregate",
            variant=context.variant,
            note=note,
        )
    ]
    if srr_values:
        observations.append(
            Observation(
                "perceptual.spauq.srr",
                float(np.median(srr_values)),
                unit="dB",
                scope="aggregate",
                variant=context.variant,
                note=note,
            )
        )
    return observations
