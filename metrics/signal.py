"""Waveform-domain metrics: correlation, error and level."""

from __future__ import annotations

import numpy as np

from atmos_quality.audio import corr, rms_db
from atmos_quality.metrics.base import MetricContext, register, require_pair
from atmos_quality.model import Observation


def _channel(pair, index: int, gain: bool) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(pair.ref[:, index], dtype=np.float64)
    candidate = np.asarray(pair.cand[:, index], dtype=np.float64)
    if gain:
        candidate = candidate * pair.alignment.gain
    return reference, candidate


@register("signal.corr")
def signal_corr(context: MetricContext) -> list[Observation]:
    pair = require_pair(context)
    observations: list[Observation] = []
    values: list[float] = []
    for index, label in enumerate(pair.labels):
        reference, candidate = _channel(pair, index, gain=False)
        if rms_db(reference) <= -100.0 or rms_db(candidate) <= -100.0:
            observations.append(
                Observation(
                    "signal.corr",
                    None,
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                    note="silent channel",
                )
            )
            continue
        value = corr(reference, candidate)
        values.append(value)
        observations.append(
            Observation(
                "signal.corr",
                value,
                scope="channel",
                channel=label,
                variant=context.variant,
            )
        )
    if values:
        observations.append(
            Observation(
                "signal.corr.mean",
                float(np.mean(values)),
                scope="aggregate",
                variant=context.variant,
            )
        )
        observations.append(
            Observation(
                "signal.corr.min",
                float(np.min(values)),
                scope="aggregate",
                variant=context.variant,
            )
        )
    return observations


@register("signal.err_db")
def signal_err_db(context: MetricContext) -> list[Observation]:
    pair = require_pair(context)
    observations: list[Observation] = []
    values: list[float] = []
    for index, label in enumerate(pair.labels):
        reference, candidate = _channel(pair, index, gain=True)
        reference_energy = float(np.dot(reference, reference))
        if reference_energy <= 1e-30:
            observations.append(
                Observation(
                    "signal.err_db",
                    None,
                    unit="dB",
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                    note="silent reference",
                )
            )
            continue
        error = candidate - reference
        value = float(10.0 * np.log10(float(np.dot(error, error)) / reference_energy + 1e-30))
        values.append(value)
        observations.append(
            Observation(
                "signal.err_db",
                value,
                unit="dB",
                scope="channel",
                channel=label,
                variant=context.variant,
            )
        )
    if values:
        observations.append(
            Observation(
                "signal.err_db.mean",
                float(np.mean(values)),
                unit="dB",
                scope="aggregate",
                variant=context.variant,
            )
        )
        observations.append(
            Observation(
                "signal.err_db.max",
                float(np.max(values)),
                unit="dB",
                scope="aggregate",
                variant=context.variant,
            )
        )
    return observations


@register("signal.rms")
def signal_rms(context: MetricContext) -> list[Observation]:
    pair = require_pair(context)
    observations: list[Observation] = []
    deltas: list[float] = []
    for index, label in enumerate(pair.labels):
        reference, candidate = _channel(pair, index, gain=False)
        ref_db = rms_db(reference)
        cand_db = rms_db(candidate)
        delta = cand_db - ref_db
        deltas.append(delta)
        observations.extend(
            [
                Observation(
                    "signal.rms.reference",
                    ref_db,
                    unit="dBFS",
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                ),
                Observation(
                    "signal.rms.candidate",
                    cand_db,
                    unit="dBFS",
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                ),
                Observation(
                    "signal.rms.delta",
                    delta,
                    unit="dB",
                    scope="channel",
                    channel=label,
                    variant=context.variant,
                ),
            ]
        )
    if deltas:
        observations.append(
            Observation(
                "signal.rms.delta.mean",
                float(np.mean(deltas)),
                unit="dB",
                scope="aggregate",
                variant=context.variant,
            )
        )
    return observations
