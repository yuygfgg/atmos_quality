"""Metric registry and built-in metrics."""

from __future__ import annotations

from atmos_quality.metrics import perceptual, signal, spatial
from atmos_quality.metrics.base import (
    METRICS,
    MetricContext,
    MetricRun,
    MetricUnavailable,
    evaluate_metrics,
    get_metric,
    list_metrics,
)

__all__ = [
    "METRICS",
    "MetricContext",
    "MetricRun",
    "MetricUnavailable",
    "evaluate_metrics",
    "get_metric",
    "list_metrics",
    "perceptual",
    "signal",
    "spatial",
]
