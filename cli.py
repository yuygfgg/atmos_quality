"""Command-line interface for the unified quality package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from atmos_quality import __version__
from atmos_quality.audio import write_wav
from atmos_quality.backends.base import BackendUnavailable
from atmos_quality.cache import Cache
from atmos_quality.config import (
    DEFAULT_CACHE,
    DEFAULT_GST_LAUNCH,
    DEFAULT_GST_PLUGIN_PATH,
    DEFAULT_OUT,
    PACKAGE_DIR,
    cache_dir,
    env_status,
    tool_status,
)
from atmos_quality.layouts import get_layout
from atmos_quality.model import Asset, RenderOptions
from atmos_quality.suites import run_cluster, run_encoder
from atmos_quality.suites.common import choose_backend


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_run(result) -> None:
    print(f"run     : {result.run_id}")
    print(f"verdict : {result.verdict['status']}")
    for reason in result.verdict["reasons"]:
        print(f"  {reason}")
    print(f"json    : {result.report_json}")
    print(f"report  : {result.report_md}")
    print(f"csv     : {result.metrics_csv}")
    aggregates = [
        observation
        for observation in result.observations
        if observation.scope == "aggregate" and observation.value is not None
    ]
    if aggregates:
        print()
        print("aggregate metrics")
        for observation in sorted(
            aggregates, key=lambda item: (item.variant, item.layout, item.metric)
        ):
            unit = f" {observation.unit}" if observation.unit else ""
            print(
                f"  {observation.variant:<10} {observation.layout:<10} "
                f"{observation.metric:<36} {observation.value:+.6f}{unit}"
            )
    if result.skipped:
        print()
        print(f"skipped: {len(result.skipped)} (see report)")
    if result.errors:
        print()
        print(f"errors: {len(result.errors)} (see report)")


def _doctor() -> int:
    print(f"atmos_quality {__version__}")
    print(f"package       : {PACKAGE_DIR}")
    status = env_status()
    print("config")
    print(f"  env files     : {', '.join(status['files']) or 'none'}")
    print(f"  env dir       : {status['dir']}")
    print(f"  python-dotenv : {'ok' if status['dotenv_available'] else 'MISSING'}")
    print(f"  cache         : {DEFAULT_CACHE or 'MISSING'}")
    print(f"  output        : {DEFAULT_OUT or 'MISSING'}")
    print("tools")
    for name, value in tool_status().items():
        print(f"  {name:<14}: {value or 'MISSING'}")
    print("optional Python modules")
    for name in ("visqol", "spauq", "atmos_cluster", "soundfile"):
        print(f"  {name:<14}: {'ok' if importlib.util.find_spec(name) else 'MISSING'}")
    print("Dolby GStreamer")
    launch_status = "ok" if DEFAULT_GST_LAUNCH and DEFAULT_GST_LAUNCH.is_file() else "MISSING"
    print(f"  launch        : {DEFAULT_GST_LAUNCH or 'MISSING'} ({launch_status})")
    plugin_status = (
        "ok" if DEFAULT_GST_PLUGIN_PATH and DEFAULT_GST_PLUGIN_PATH.exists() else "MISSING"
    )
    print(f"  plugins       : {DEFAULT_GST_PLUGIN_PATH or 'MISSING'} ({plugin_status})")
    print("Cavern")
    source = PACKAGE_DIR / "cavern_render"
    source_status = "ok" if (source / "CavernRender.csproj").is_file() else "MISSING"
    print(f"  source        : {source} ({source_status})")
    print(f"  dotnet        : {tool_status().get('dotnet') or 'MISSING'}")
    return 0


def _render(args: argparse.Namespace) -> int:
    asset = Asset.from_path(args.input)
    layout = get_layout(args.layout)
    if args.backend == "dolby":
        from atmos_quality.backends.dolby_gst import DolbyGstBackend

        backend = DolbyGstBackend()
    elif args.backend == "cavern":
        from atmos_quality.backends.cavern import CavernBackend

        backend = CavernBackend()
    else:
        backend = choose_backend([asset], layout)
    if not backend.supports(asset, layout):
        raise ValueError(f"{backend.name} cannot render {asset.kind} to {layout.name}")

    cache = Cache(cache_dir(args.cache))
    options = RenderOptions(max_seconds=args.max_seconds)
    render = backend.render(asset, layout, options, cache, force=args.force)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".wav", ".wave"}:
        write_wav(output, render.open(), render.sample_rate)
    else:
        shutil.copyfile(render.path, output)
    sidecar = Path(str(output) + ".json")
    payload = render.to_dict()
    payload["output"] = str(output)
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"render  : {output}")
    print(f"sidecar : {sidecar}")
    print(f"backend : {render.backend}")
    print(f"layout  : {render.layout.name} ({','.join(render.labels)})")
    print(f"samples : {render.samples} ({render.samples / render.sample_rate:.3f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aq",
        description=(
            "Unified objective/perceptual quality measurement for "
            "Dolby Atmos masters and streams."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="show tool and optional-metric availability")

    render = sub.add_parser("render", help="render one asset to a speaker layout")
    render.add_argument("input")
    render.add_argument("--layout", default="7.1.4")
    render.add_argument("--out", required=True)
    render.add_argument("--backend", choices=("auto", "dolby", "cavern"), default="auto")
    render.add_argument("--cache", default=None, help="cache directory (default: AQ_CACHE)")
    render.add_argument("--max-seconds", type=float, default=0.0)
    render.add_argument("--force", action="store_true")

    encoder = sub.add_parser(
        "encoder", help="score an encoded stream against a master/DME reference"
    )
    encoder.add_argument("master", help="ground-truth .mlp/.ec3 master; use `aq cluster` for DAMF")
    encoder.add_argument("candidate", help="encoded .ec3/.mlp candidate")
    encoder.add_argument("--reference", help="optional DME/other reference .ec3")
    encoder.add_argument("--layouts", default="7.1.4,binaural")
    encoder.add_argument("--metrics", help="comma-separated metric names (default: suite metrics)")
    encoder.add_argument("--out", default=None, help="report directory (default: AQ_OUT)")
    encoder.add_argument("--cache", default=None, help="cache directory (default: AQ_CACHE)")
    encoder.add_argument("--workdir")
    encoder.add_argument("--max-seconds", type=float, default=0.0)
    encoder.add_argument("--no-align", action="store_true")
    encoder.add_argument("--no-visqol", action="store_true")
    encoder.add_argument("--no-spauq", action="store_true")
    encoder.add_argument("--force", action="store_true")

    cluster = sub.add_parser("cluster", help="score a clustered DAMF master against its source")
    cluster.add_argument("source")
    cluster.add_argument("candidate")
    cluster.add_argument("--layouts", default="7.1.4")
    cluster.add_argument("--metrics", help="comma-separated metric names (default: suite metrics)")
    cluster.add_argument("--out", default=None, help="report directory (default: AQ_OUT)")
    cluster.add_argument("--cache", default=None, help="cache directory (default: AQ_CACHE)")
    cluster.add_argument("--workdir")
    cluster.add_argument("--max-seconds", type=float, default=0.0)
    cluster.add_argument("--no-align", action="store_true")
    cluster.add_argument("--no-visqol", action="store_true")
    cluster.add_argument("--no-spauq", action="store_true")
    cluster.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "render":
            return _render(args)
        if args.command == "encoder":
            result = run_encoder(
                args.master,
                args.candidate,
                reference=args.reference,
                layouts=tuple(_csv(args.layouts)),
                metrics=_csv(args.metrics) or None,
                out_root=args.out,
                cache_root=args.cache,
                workdir=args.workdir,
                max_seconds=args.max_seconds,
                force=args.force,
                align=not args.no_align,
                include_visqol=not args.no_visqol,
                include_spauq=not args.no_spauq,
            )
            _print_run(result)
            return 0
        if args.command == "cluster":
            result = run_cluster(
                args.source,
                args.candidate,
                layouts=tuple(_csv(args.layouts)),
                metrics=_csv(args.metrics) or None,
                out_root=args.out,
                cache_root=args.cache,
                workdir=args.workdir,
                max_seconds=args.max_seconds,
                force=args.force,
                align=not args.no_align,
                include_visqol=not args.no_visqol,
                include_spauq=not args.no_spauq,
            )
            _print_run(result)
            return 0
    except (BackendUnavailable, FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2
