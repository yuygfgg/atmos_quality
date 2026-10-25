"""Stable JSON/CSV/Markdown reports and an append-only run index."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from atmos_quality import __version__
from atmos_quality.config import PROJECT_ROOT, tool_status
from atmos_quality.model import Asset, Observation, Pair, Render

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    report_json: Path
    report_md: Path
    metrics_csv: Path
    verdict: dict[str, Any]
    observations: list[Observation]
    skipped: list[str]
    errors: list[str]


def _short_hash(*parts: object) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:8]


def _git_sha(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def provenance() -> dict[str, Any]:
    tools = tool_status()
    return {
        "package": "atmos_quality",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "git_encoder": _git_sha(PROJECT_ROOT / "dolby-atmos-encoder"),
        "tools": tools,
    }


def _verdict(observations: Iterable[Observation], errors: list[str]) -> dict[str, Any]:
    reasons: list[str] = []
    if errors:
        reasons.append(f"{len(errors)} metric error(s)")
    correlations = [
        observation.value
        for observation in observations
        if observation.metric == "signal.corr.mean" and observation.value is not None
    ]
    status = "pass"
    if correlations:
        worst = min(correlations)
        if worst < 0.90:
            status = "fail"
            reasons.append(f"mean correlation {worst:.4f} < 0.90")
        elif worst < 0.95:
            status = "warn"
            reasons.append(f"mean correlation {worst:.4f} < 0.95")
    elif not errors:
        status = "warn"
        reasons.append("no signal.corr.mean observation")
    if errors and status == "pass":
        status = "fail"
    return {"status": status, "reasons": reasons}


def _markdown(
    suite: str,
    run_id: str,
    created_at: str,
    assets: dict[str, Asset],
    renders: list[Render],
    pairs: list[Pair],
    observations: list[Observation],
    skipped: list[str],
    errors: list[str],
    verdict: dict[str, Any],
    extra: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# Quality run: {suite}",
        "",
        f"- run: `{run_id}`",
        f"- created: {created_at}",
        f"- verdict: **{verdict['status']}**",
    ]
    if verdict["reasons"]:
        lines.append("- reasons: " + "; ".join(verdict["reasons"]))
    if extra:
        for key, value in extra.items():
            lines.append(f"- {key}: {value}")
    lines.append("")

    lines += [
        "## Assets",
        "",
        "| role | kind | path | objects | channels |",
        "|---|---|---|---:|---:|",
    ]
    for role, asset in assets.items():
        objects = asset.objects if asset.objects is not None else ""
        channels = asset.channels if asset.channels is not None else ""
        lines.append(f"| {role} | {asset.kind} | `{asset.path}` | {objects} | {channels} |")
    lines.append("")

    lines += [
        "## Renders",
        "",
        "| asset | backend | layout | labels | duration |",
        "|---|---|---|---|---:|",
    ]
    for render in renders:
        duration = render.samples / render.sample_rate if render.sample_rate else 0.0
        lines.append(
            f"| `{render.asset.path.name}` | {render.backend} | {render.layout.name} | "
            f"{','.join(render.labels)} | {duration:.2f}s |"
        )
    lines.append("")

    lines += [
        "## Pairs",
        "",
        "| variant | layout | lag | gain | confidence | samples |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair.variant} | {pair.reference.layout.name} | {pair.alignment.lag} | "
            f"{pair.alignment.gain_db:+.2f} dB | {pair.alignment.confidence:.4f} | {pair.samples} |"
        )
    lines.append("")

    lines += [
        "## Metrics",
        "",
        "| variant | layout | metric | channel | value | unit |",
        "|---|---|---|---|---:|---|",
    ]
    for observation in sorted(
        observations,
        key=lambda item: (item.variant, item.layout, item.metric, item.channel or ""),
    ):
        value = "" if observation.value is None else f"{observation.value:.6f}"
        note = f" ({observation.note})" if observation.note else ""
        lines.append(
            f"| {observation.variant} | {observation.layout} | {observation.metric} | "
            f"{observation.channel or ''} | {value}{note} | {observation.unit} |"
        )
    lines.append("")

    if skipped:
        lines += ["## Skipped", ""]
        lines += [f"- {item}" for item in skipped]
        lines.append("")
    if errors:
        lines += ["## Errors", ""]
        lines += [f"- {item}" for item in errors]
        lines.append("")
    return "\n".join(lines) + "\n"


def write_run(
    *,
    suite: str,
    out_root: Path,
    assets: dict[str, Asset],
    renders: list[Render],
    pairs: list[Pair],
    observations: list[Observation],
    skipped: list[str],
    errors: list[str],
    extra: dict[str, Any] | None = None,
) -> RunResult:
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    created_at = datetime.now(timezone.utc).isoformat()
    identity = [
        suite,
        *[f"{role}:{asset.digest}" for role, asset in assets.items()],
        *[
            f"{pair.variant}:{pair.reference.cache_key}:{pair.candidate.cache_key}"
            for pair in pairs
        ],
    ]
    run_id = f"{suite}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{_short_hash(*identity)}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    verdict = _verdict(observations, errors)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "created_at": created_at,
        "provenance": provenance(),
        "assets": {role: asset.to_dict() for role, asset in assets.items()},
        "renders": [render.to_dict() for render in renders],
        "pairs": [pair.to_dict() for pair in pairs],
        "observations": [observation.to_dict() for observation in observations],
        "skipped": skipped,
        "errors": errors,
        "verdict": verdict,
        "extra": extra,
    }

    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    metrics_csv = run_dir / "metrics.csv"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(
        _markdown(
            suite,
            run_id,
            created_at,
            assets,
            renders,
            pairs,
            observations,
            skipped,
            errors,
            verdict,
            extra,
        ),
        encoding="utf-8",
    )
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric", "scope", "channel", "variant", "layout", "value", "unit", "note"]
        )
        for observation in observations:
            writer.writerow(
                [
                    observation.metric,
                    observation.scope,
                    observation.channel or "",
                    observation.variant,
                    observation.layout,
                    "" if observation.value is None else f"{observation.value:.9g}",
                    observation.unit,
                    observation.note,
                ]
            )

    summary = {
        "run_id": run_id,
        "suite": suite,
        "created_at": created_at,
        "status": verdict["status"],
        "assets": {role: str(asset.path) for role, asset in assets.items()},
        "metrics": {
            f"{observation.variant}/{observation.layout}/{observation.metric}": observation.value
            for observation in observations
            if observation.scope == "aggregate"
        },
        "report_json": str(report_json),
        "report_md": str(report_md),
    }
    with (out_root / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        report_json=report_json,
        report_md=report_md,
        metrics_csv=metrics_csv,
        verdict=verdict,
        observations=observations,
        skipped=skipped,
        errors=errors,
    )
