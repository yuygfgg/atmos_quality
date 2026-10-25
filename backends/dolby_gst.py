"""Official Dolby GStreamer backend for E-AC-3 JOC and TrueHD."""

from __future__ import annotations

from pathlib import Path

from atmos_quality.backends.base import BackendUnavailable, RenderBackend, run_command
from atmos_quality.cache import Cache, read_meta, write_meta
from atmos_quality.config import (
    DEFAULT_GST_LAUNCH,
    DEFAULT_GST_PLUGIN_PATH,
    DEFAULT_GST_ROOT,
    gst_environment,
)
from atmos_quality.layouts import Layout
from atmos_quality.model import Asset, Render, RenderOptions


class DolbyGstBackend(RenderBackend):
    name = "dolby-gst"
    version = "1"

    def supports(self, asset: Asset, layout: Layout) -> bool:
        if asset.kind not in {"ec3", "truehd"}:
            return False
        return layout.name == "binaural" or layout.dolby_id is not None

    def preflight(self) -> None:
        try:
            gst_environment()
        except RuntimeError as exc:
            raise BackendUnavailable(str(exc)) from exc
        if DEFAULT_GST_LAUNCH is None:
            raise BackendUnavailable("AQ_GST_LAUNCH is not configured; set it in .env")
        if not DEFAULT_GST_LAUNCH.is_file():
            raise BackendUnavailable(f"Dolby GStreamer not found: {DEFAULT_GST_LAUNCH}")
        if DEFAULT_GST_PLUGIN_PATH is None:
            raise BackendUnavailable("AQ_GST_PLUGIN_PATH is not configured; set it in .env")
        if not DEFAULT_GST_PLUGIN_PATH.exists():
            raise BackendUnavailable(f"GStreamer plugin path not found: {DEFAULT_GST_PLUGIN_PATH}")
        if DEFAULT_GST_ROOT is not None and not DEFAULT_GST_ROOT.exists():
            raise BackendUnavailable(f"Dolby GStreamer root not found: {DEFAULT_GST_ROOT}")

    def render(
        self,
        asset: Asset,
        layout: Layout,
        options: RenderOptions,
        cache: Cache,
        *,
        force: bool = False,
    ) -> Render:
        self.preflight()
        if not self.supports(asset, layout):
            raise ValueError(f"{self.name} does not support {asset.kind} -> {layout.name}")

        key = cache.render_key(self.name, self.version, asset, layout, options)
        output = cache.path("renders/dolby-gst", key, ".f32")
        meta_path = output.with_suffix(".json")

        if not force and output.is_file() and meta_path.is_file():
            meta = read_meta(meta_path)
            return self._make_render(asset, layout, options, key, output, meta)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.unlink(missing_ok=True)
        command = self._command(asset, layout, options, temporary)
        run_command(command, env=gst_environment(), timeout=1800)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"GStreamer produced no output: {temporary}")

        if options.max_seconds > 0.0:
            limit = int(round(options.max_seconds * 48000)) * layout.channels * 4
            if temporary.stat().st_size > limit:
                with temporary.open("r+b") as handle:
                    handle.truncate(limit)

        temporary.replace(output)
        samples = output.stat().st_size // (4 * layout.channels)
        meta = {
            "backend": self.name,
            "backend_version": self.version,
            "asset": asset.to_dict(),
            "layout": layout.name,
            "labels": list(layout.labels),
            "sample_rate": 48000,
            "samples": samples,
            "options": options.to_dict(),
            "command": command,
            "gst_root": str(DEFAULT_GST_ROOT) if DEFAULT_GST_ROOT else None,
        }
        write_meta(meta_path, meta)
        return self._make_render(asset, layout, options, key, output, meta)

    def _make_render(
        self,
        asset: Asset,
        layout: Layout,
        options: RenderOptions,
        key: str,
        output: Path,
        meta: dict,
    ) -> Render:
        return Render(
            path=output,
            format="f32",
            labels=layout.labels,
            sample_rate=int(meta["sample_rate"]),
            samples=int(meta["samples"]),
            backend=self.name,
            asset=asset,
            layout=layout,
            options=options,
            cache_key=key,
            meta=meta,
        )

    def _command(
        self,
        asset: Asset,
        layout: Layout,
        options: RenderOptions,
        output: Path,
    ) -> list[str]:
        command = [str(DEFAULT_GST_LAUNCH), "-q", "filesrc", f"location={asset.path}"]
        if asset.kind == "ec3":
            command += ["!", "dlbac3parse"]
        elif asset.kind == "truehd":
            command += ["!", "dlbtruehdparse", "align-major-sync=false"]
        else:
            raise ValueError(f"unsupported asset kind: {asset.kind}")

        if layout.name == "binaural":
            if asset.kind == "ec3":
                command += [
                    "enable-metadata=true",
                    "!",
                    "dlbac3dec",
                    "out-ch-config=21",
                    f"drc-suppress={'true' if options.drc_suppress else 'false'}",
                ]
            else:
                command += [
                    "!",
                    "dlbtruehddec",
                    "out-ch-config=21",
                    "presentation=16",
                    f"drc-mode={'disabled' if options.drc_suppress else 'follow'}",
                ]
            command += [
                "!",
                "oamdcapsfeature",
                "float=true",
                "!",
                "dlbdapvr",
                "!",
                "audioconvert",
                "!",
                "audio/x-raw,format=F32LE,layout=interleaved",
                "!",
                "filesink",
                f"location={output}",
            ]
            return command

        if layout.dolby_id is None:
            raise ValueError(f"layout {layout.name} has no Dolby out-ch-config id")
        command += ["!", "dlbaudiodecbin"]
        if asset.kind == "truehd":
            command += ["truehddec-presentation=16"]
        command += [
            f"out-ch-config={layout.dolby_id}",
            f"oar-limiter-enable={'true' if options.limiter else 'false'}",
            f"oar-drop-delay={'true' if options.drop_delay else 'false'}",
        ]
        if asset.kind == "ec3":
            command += [f"ac3dec-drc-suppress={'true' if options.drc_suppress else 'false'}"]
        else:
            command += [f"truehddec-drc-mode={'disabled' if options.drc_suppress else 'follow'}"]
        command += [
            "!",
            "audioconvert",
            "!",
            f"audio/x-raw,format=F32LE,layout=interleaved,channels={layout.channels}",
            "!",
            "filesink",
            f"location={output}",
        ]
        return command
