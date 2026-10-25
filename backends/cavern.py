"""Cavern backend: arbitrary-object DAMF masters to arbitrary speaker layouts."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from atmos_quality.backends.base import BackendUnavailable, RenderBackend, run_command
from atmos_quality.cache import Cache, read_meta, write_meta
from atmos_quality.config import PACKAGE_DIR
from atmos_quality.layouts import Layout
from atmos_quality.model import Asset, Render, RenderOptions

CAVERN_SOURCE = PACKAGE_DIR / "cavern_render"


class CavernBackend(RenderBackend):
    name = "cavern"
    version = "2.1.0"

    def supports(self, asset: Asset, layout: Layout) -> bool:
        return asset.kind == "damf" and layout.cavern_spec is not None

    def preflight(self) -> None:
        if shutil.which("dotnet") is None:
            raise BackendUnavailable("dotnet is not installed or not in PATH")
        if not (CAVERN_SOURCE / "CavernRender.csproj").is_file():
            raise BackendUnavailable(f"Cavern source not found: {CAVERN_SOURCE}")

    def render(
        self,
        asset: Asset,
        layout: Layout,
        options: RenderOptions,
        cache: Cache,
        *,
        force: bool = False,
    ) -> Render:
        if not self.supports(asset, layout):
            raise ValueError(f"{self.name} does not support {asset.kind} -> {layout.name}")

        key = cache.render_key(self.name, self.version, asset, layout, options)
        output = cache.path("renders/cavern", key, ".f32")
        meta_path = output.with_suffix(".json")
        if not force and output.is_file() and meta_path.is_file():
            meta = read_meta(meta_path)
            return self._make_render(asset, layout, options, key, output, meta)

        prefix = self._build(cache)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.unlink(missing_ok=True)
        sidecar = Path(str(temporary) + ".json")
        sidecar.unlink(missing_ok=True)

        command = [
            *prefix,
            str(asset.path),
            str(temporary),
            layout.cavern_spec or layout.name,
            "--raw",
        ]
        if options.max_seconds > 0.0:
            command += ["--max-seconds", f"{options.max_seconds:.6f}"]
        if options.gain_db:
            command += ["--gain-db", f"{options.gain_db:.6f}"]
        command += ["--update-rate", str(options.update_rate)]
        run_command(command, timeout=1800)

        if not temporary.is_file() or not sidecar.is_file():
            raise RuntimeError(f"Cavern produced no output: {temporary}")
        meta = read_meta(sidecar)
        source_labels = tuple(meta.get("channels") or ())
        if not source_labels:
            raise RuntimeError(f"Cavern sidecar has no channel labels: {sidecar}")

        if source_labels != layout.labels:
            self._reorder_file(temporary, source_labels, layout.labels, int(meta["samples"]))
            meta["source_labels"] = list(source_labels)
        meta["channels"] = list(layout.labels)

        sidecar.unlink(missing_ok=True)
        temporary.replace(output)
        meta.update(
            {
                "backend": self.name,
                "backend_version": self.version,
                "asset": asset.to_dict(),
                "layout": layout.name,
                "labels": list(layout.labels),
                "options": options.to_dict(),
                "command": command,
            }
        )
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

    def _build(self, cache: Cache) -> list[str]:
        self.preflight()
        csproj = CAVERN_SOURCE / "CavernRender.csproj"
        program = CAVERN_SOURCE / "Program.cs"
        key = cache.build_key(self.version, csproj.read_bytes(), program.read_bytes())
        build_root = cache.root / "cavern_build"
        out_dir = build_root / key
        exe = out_dir / "cavern-render"
        dll = out_dir / "cavern-render.dll"
        if exe.is_file():
            return [str(exe)]
        if dll.is_file():
            return ["dotnet", str(dll)]

        source_dir = build_root / f"{key}_src"
        if source_dir.exists():
            shutil.rmtree(source_dir)
        shutil.copytree(
            CAVERN_SOURCE,
            source_dir,
            ignore=shutil.ignore_patterns("bin", "obj", ".gitignore"),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "dotnet",
                "build",
                "-c",
                "Release",
                "-o",
                str(out_dir),
                str(source_dir / "CavernRender.csproj"),
            ],
            cwd=source_dir,
            timeout=600,
        )
        if exe.is_file():
            return [str(exe)]
        if dll.is_file():
            return ["dotnet", str(dll)]
        raise RuntimeError(f"Cavern build did not produce an executable in {out_dir}")

    @staticmethod
    def _reorder_file(
        path: Path,
        source_labels: tuple[str, ...],
        target_labels: tuple[str, ...],
        samples: int,
    ) -> None:
        index = {label: i for i, label in enumerate(source_labels)}
        missing = [label for label in target_labels if label not in index]
        if missing:
            raise RuntimeError(f"Cavern output is missing channels {missing}; got {source_labels}")
        order = np.asarray([index[label] for label in target_labels], dtype=np.intp)
        data = np.memmap(path, dtype="<f4", mode="r", shape=(samples, len(source_labels)))
        temporary = path.with_name(path.name + ".ordered")
        with temporary.open("wb") as handle:
            for start in range(0, samples, 1 << 20):
                stop = min(start + (1 << 20), samples)
                block = np.asarray(data[start:stop][:, order], dtype="<f4")
                handle.write(np.ascontiguousarray(block).tobytes())
        del data
        temporary.replace(path)
