"""Tool discovery, paths and environment for the quality package.

Machine-specific paths are loaded from ``.env`` (see ``.env.example``) rather
than being compiled into this module. ``AQ_ENV_FILE`` can point at an explicit
env file; otherwise the nearest ``.env`` from the current directory and the
workspace root are searched. Shell environment variables take precedence over
``.env``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    load_dotenv = None

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _candidate_env_files() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("AQ_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            candidates.append(candidate)
            break
    candidates.append(PROJECT_ROOT / ".env")
    return candidates


def _load_env_files() -> tuple[Path, ...]:
    loaded: list[Path] = []
    seen: set[Path] = set()
    for candidate in _candidate_env_files():
        path = candidate.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if load_dotenv is not None:
            load_dotenv(path, override=False)
        loaded.append(path)
    return tuple(loaded)


ENV_FILES = _load_env_files()
ENV_DIR = ENV_FILES[0].parent if ENV_FILES else Path.cwd().resolve()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ENV_DIR / path
    return path.resolve()


def _derive(root: Path | None, relative: str) -> Path | None:
    return root / relative if root is not None else None


DEFAULT_GST_ROOT = _env_path("AQ_GST_ROOT")
DEFAULT_GST_PLUGIN_PATH = _env_path("AQ_GST_PLUGIN_PATH")
DEFAULT_GST_LAUNCH = _env_path("AQ_GST_LAUNCH") or _derive(
    DEFAULT_GST_ROOT, "Resources/bin/gst-launch-1.0"
)
DEFAULT_GST_SCANNER = _env_path("AQ_GST_SCANNER") or _derive(
    DEFAULT_GST_ROOT, "Resources/libexec/gst-plugin-scanner"
)
DEFAULT_GST_SYSTEM_PATH = _env_path("AQ_GST_SYSTEM_PATH") or _derive(
    DEFAULT_GST_ROOT, "Resources/lib/gstreamer-1.0"
)
DEFAULT_DDPMETA = _env_path("AQ_DDPMETA")
DEFAULT_CACHE = _env_path("AQ_CACHE")
DEFAULT_OUT = _env_path("AQ_OUT")


def cache_dir(value: str | Path | None = None) -> Path:
    path = Path(value).expanduser() if value not in (None, "") else DEFAULT_CACHE
    if path is None:
        raise RuntimeError("AQ_CACHE is not configured; set it in .env or pass --cache")
    return path.resolve()


def output_dir(value: str | Path | None = None) -> Path:
    path = Path(value).expanduser() if value not in (None, "") else DEFAULT_OUT
    if path is None:
        raise RuntimeError("AQ_OUT is not configured; set it in .env or pass --out")
    return path.resolve()


def gst_environment() -> dict[str, str]:
    missing = [
        name
        for name, value in (
            ("AQ_GST_PLUGIN_PATH", DEFAULT_GST_PLUGIN_PATH),
            ("AQ_GST_SCANNER", DEFAULT_GST_SCANNER),
            ("AQ_GST_SYSTEM_PATH", DEFAULT_GST_SYSTEM_PATH),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(
            "Dolby GStreamer is not configured; set "
            + ", ".join(missing)
            + " in .env (or the shell environment)"
        )
    env = os.environ.copy()
    env["GST_PLUGIN_SCANNER"] = str(DEFAULT_GST_SCANNER)
    env["GST_PLUGIN_PATH"] = str(DEFAULT_GST_PLUGIN_PATH)
    env["GST_PLUGIN_SYSTEM_PATH_1_0"] = str(DEFAULT_GST_SYSTEM_PATH)
    return env


def env_status() -> dict[str, object]:
    return {
        "files": [str(path) for path in ENV_FILES],
        "dir": str(ENV_DIR),
        "dotenv_available": load_dotenv is not None,
    }


def tool_status() -> dict[str, str | None]:
    def executable(name: str) -> str | None:
        return shutil.which(name)

    def existing(path: Path | None) -> str | None:
        return str(path) if path is not None and path.exists() else None

    return {
        "ffmpeg": executable("ffmpeg"),
        "dotnet": executable("dotnet"),
        "gst_launch": existing(DEFAULT_GST_LAUNCH),
        "gst_plugin_path": existing(DEFAULT_GST_PLUGIN_PATH),
        "ddpmeta": existing(DEFAULT_DDPMETA),
    }
