"""Render backend protocol and shared subprocess helper."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

from atmos_quality.cache import Cache
from atmos_quality.layouts import Layout
from atmos_quality.model import Asset, Render, RenderOptions


class BackendUnavailable(RuntimeError):
    """The external renderer or its dependencies are not available."""


class RenderBackend(ABC):
    name: str
    version: str

    @abstractmethod
    def supports(self, asset: Asset, layout: Layout) -> bool:
        raise NotImplementedError

    @abstractmethod
    def render(
        self,
        asset: Asset,
        layout: Layout,
        options: RenderOptions,
        cache: Cache,
        *,
        force: bool = False,
    ) -> Render:
        raise NotImplementedError

    def preflight(self) -> None:
        return None


def run_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise BackendUnavailable(f"command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {' '.join(command[:4])}") from exc
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command[:6])}\n{tail}")
    return result
