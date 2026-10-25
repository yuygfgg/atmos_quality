"""Audio I/O and small signal helpers used by the quality package."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

HASH_LIMIT = 64 << 20


def related_asset_files(path: Path) -> list[Path]:
    """Return the files that make up an asset for fingerprinting."""
    files = [path]
    if path.suffix.lower() in {".atmos", ".damf"}:
        for suffix in (".metadata", ".audio"):
            candidate = Path(str(path) + suffix)
            if candidate.is_file():
                files.append(candidate)
    return files


def fingerprint(path: Path) -> str:
    """Stable fingerprint for cache keys.

    Small files are hashed by content. Large masters use path/size/mtime and
    a content hash of their small metadata siblings.
    """
    digest = hashlib.sha1()
    for item in related_asset_files(path):
        stat = item.stat()
        digest.update(str(item.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        if stat.st_size <= HASH_LIMIT:
            digest.update(hashlib.sha1(item.read_bytes()).digest())
    return digest.hexdigest()


def _to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.float32 or data.dtype == np.float64:
        return data.astype(np.float32, copy=False)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    raise TypeError(f"unsupported sample type: {data.dtype}")


def load_f32(path: Path, channels: int) -> np.memmap:
    path = Path(path)
    size = path.stat().st_size
    if channels <= 0:
        raise ValueError("channels must be positive")
    if size % (4 * channels):
        raise ValueError(
            f"{path}: {size} bytes is not a whole number of {channels}-channel f32 frames"
        )
    samples = size // (4 * channels)
    return np.memmap(path, dtype="<f4", mode="r", shape=(samples, channels))


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    if data.ndim == 1:
        data = data[:, None]
    return int(rate), _to_float(data)


def load_audio(path: Path, *, channels: int | None = None) -> tuple[int, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in {".f32", ".raw", ".pcm"}:
        if channels is None:
            raise ValueError(f"{path}: channels are required for raw f32 input")
        data = load_f32(path, channels)
        return 48000, data
    if suffix in {".wav", ".wave"}:
        return load_wav(path)
    raise ValueError(f"unsupported audio container: {path.suffix}")


def write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    if data.ndim == 1:
        data = data[:, None]
    wavfile.write(path, sample_rate, np.asarray(data, dtype=np.float32))


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    write_wav(path, np.asarray(samples, dtype=np.float32), sample_rate)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y))) + 1e-30
    return float(np.dot(x, y) / denominator)


def rms_db(x: np.ndarray) -> float:
    value = float(np.mean(np.square(np.asarray(x, dtype=np.float64))))
    return 10.0 * np.log10(value + 1e-30)
