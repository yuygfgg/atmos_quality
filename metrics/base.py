"""Metric protocol, registry and failure isolation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atmos_quality.model import Observation, Pair


class MetricUnavailable(RuntimeError):
    """An optional metric or its dependency is not available."""


@dataclass
class MetricContext:
    pair: Pair | None = None
    spatial: Any | None = None
    workdir: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    variant: str = ""


@dataclass
class MetricRun:
    observations: list[Observation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


MetricFunction = Callable[[MetricContext], list[Observation]]
METRICS: dict[str, MetricFunction] = {}


def register(name: str) -> Callable[[MetricFunction], MetricFunction]:
    def decorator(function: MetricFunction) -> MetricFunction:
        if name in METRICS:
            raise ValueError(f"duplicate metric: {name}")
        METRICS[name] = function
        return function

    return decorator


def get_metric(name: str) -> MetricFunction:
    try:
        return METRICS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(METRICS))
        raise KeyError(f"unknown metric {name!r}; available: {valid}") from exc


def list_metrics() -> list[str]:
    return sorted(METRICS)


def evaluate_metrics(names: list[str], context: MetricContext) -> MetricRun:
    run = MetricRun()
    for name in names:
        try:
            run.observations.extend(get_metric(name)(context))
        except MetricUnavailable as exc:
            run.skipped.append(f"{name}: {exc}")
        except Exception as exc:  # keep one metric failure from aborting a run
            run.errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return run


def require_pair(context: MetricContext) -> Pair:
    if context.pair is None:
        raise MetricUnavailable("no aligned audio pair")
    return context.pair
