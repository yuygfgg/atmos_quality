"""Shared lag/trim/gain alignment for all metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from atmos_quality.layouts import reorder_channels
from atmos_quality.model import Alignment, Pair, Render


@dataclass(frozen=True)
class LagResult:
    lag: int
    confidence: float


def _alignment_channels(labels: tuple[str, ...]) -> list[int]:
    indices = [i for i, label in enumerate(labels) if label != "LFE"]
    return indices or list(range(len(labels)))


def find_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: tuple[str, ...],
    sample_rate: int,
    *,
    start_s: float = 30.0,
    duration_s: float = 10.0,
    max_lag: int = 20000,
) -> LagResult:
    """Find the candidate delay in samples.

    Positive ``lag`` means the candidate is delayed: ``candidate[lag:]``
    aligns with ``reference[:]``.
    """
    channels = _alignment_channels(labels)
    length = min(len(reference), len(candidate))
    if length == 0:
        raise ValueError("cannot align empty signals")
    start = min(int(start_s * sample_rate), max(length - 1, 0))
    stop = min(start + int(duration_s * sample_rate), length)
    if stop - start < sample_rate // 4:
        start = 0
        stop = min(length, int(20.0 * sample_rate))
    if stop <= start:
        raise ValueError("signal is too short to align")

    a = np.asarray(reference[start:stop][:, channels], dtype=np.float64).mean(axis=1)
    a = a - a.mean()
    begin = max(0, start - max_lag)
    end = min(len(candidate), stop + max_lag)
    b = np.asarray(candidate[begin:end][:, channels], dtype=np.float64).mean(axis=1)
    b = b - b.mean()
    if len(b) < len(a):
        raise ValueError("candidate search window is shorter than the reference window")

    correlation = signal.correlate(a, b, mode="full", method="fft")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    peak = int(lags[int(np.argmax(np.abs(correlation)))])
    lag = begin - start - peak

    aligned_reference, aligned_candidate = trim_pair(reference, candidate, lag)
    segment_start = min(start, max(len(aligned_reference) - 1, 0))
    segment_stop = min(segment_start + (stop - start), len(aligned_reference))
    x = np.asarray(aligned_reference[segment_start:segment_stop][:, channels], dtype=np.float64)
    y = np.asarray(aligned_candidate[segment_start:segment_stop][:, channels], dtype=np.float64)
    x = x.mean(axis=1)
    y = y.mean(axis=1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y))) + 1e-30
    confidence = float(np.dot(x, y) / denominator)
    return LagResult(lag=lag, confidence=confidence)


def trim_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        candidate = candidate[lag:]
    else:
        reference = reference[-lag:]
    length = min(len(reference), len(candidate))
    if length <= 0:
        raise ValueError(f"alignment produced no overlap (lag={lag})")
    return reference[:length], candidate[:length]


def global_gain(reference: np.ndarray, candidate: np.ndarray, labels: tuple[str, ...]) -> float:
    channels = _alignment_channels(labels)
    ref = np.asarray(reference[:, channels], dtype=np.float64).ravel()
    cand = np.asarray(candidate[:, channels], dtype=np.float64).ravel()
    denominator = float(np.dot(cand, cand))
    if denominator <= 1e-30:
        return 1.0
    return float(np.dot(ref, cand) / denominator)


def make_pair(
    reference: Render,
    candidate: Render,
    *,
    level_match: str = "global",
    align: bool = True,
    start_s: float = 30.0,
    duration_s: float = 10.0,
    max_lag: int = 20000,
    variant: str = "master",
) -> Pair:
    if reference.sample_rate != candidate.sample_rate:
        raise ValueError(
            f"sample-rate mismatch: {reference.sample_rate} != {candidate.sample_rate}"
        )

    ref_data = reference.open()
    cand_data = candidate.open()
    labels = reference.labels
    if tuple(candidate.labels) != labels:
        cand_data = reorder_channels(cand_data, candidate.labels, labels)

    if align:
        lag_result = find_lag(
            ref_data,
            cand_data,
            labels,
            reference.sample_rate,
            start_s=start_s,
            duration_s=duration_s,
            max_lag=max_lag,
        )
        lag = lag_result.lag
        confidence = lag_result.confidence
        method = "fft-crosscorr"
    else:
        lag = 0
        confidence = 1.0
        method = "none"

    ref_trimmed, cand_trimmed = trim_pair(ref_data, cand_data, lag)
    if level_match == "global":
        gain = global_gain(ref_trimmed, cand_trimmed, labels)
    elif level_match == "none":
        gain = 1.0
    else:
        raise ValueError(f"unknown level_match mode: {level_match}")

    alignment = Alignment(
        lag=lag,
        gain=gain,
        method=method,
        confidence=confidence,
    )
    return Pair(
        reference=reference,
        candidate=candidate,
        ref=ref_trimmed,
        cand=cand_trimmed,
        labels=labels,
        sample_rate=reference.sample_rate,
        alignment=alignment,
        variant=variant,
    )
