"""Canonical speaker layouts and channel-label reordering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    """A named speaker layout.

    ``labels`` is the canonical channel order used by the quality package.
    ``dolby_id`` is the Dolby GStreamer ``out-ch-config`` enum value.
    ``cavern_spec`` is the layout name accepted by ``cavern-render``.
    """

    name: str
    labels: tuple[str, ...]
    dolby_id: int | None = None
    cavern_spec: str | None = None

    @property
    def channels(self) -> int:
        return len(self.labels)


def _layout(
    name: str,
    labels: Iterable[str],
    *,
    dolby_id: int | None = None,
    cavern_spec: str | None = None,
) -> Layout:
    return Layout(name=name, labels=tuple(labels), dolby_id=dolby_id, cavern_spec=cavern_spec)


LAYOUTS: dict[str, Layout] = {
    "2.0": _layout("2.0", ("L", "R"), dolby_id=0, cavern_spec="2.0"),
    "3.1": _layout("3.1", ("L", "R", "C", "LFE"), dolby_id=3),
    "5.1": _layout("5.1", ("L", "R", "C", "LFE", "Ls", "Rs"), dolby_id=7, cavern_spec="5.1"),
    "7.1": _layout(
        "7.1",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"),
        dolby_id=11,
        cavern_spec="7.1",
    ),
    "9.1": _layout(
        "9.1",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Lw", "Rw"),
        dolby_id=12,
    ),
    "5.1.2": _layout(
        "5.1.2",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Ltm", "Rtm"),
        dolby_id=13,
    ),
    "5.1.4": _layout(
        "5.1.4",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Ltf", "Rtf", "Ltr", "Rtr"),
        dolby_id=14,
        cavern_spec="5.1.4",
    ),
    "7.1.2": _layout(
        "7.1.2",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltm", "Rtm"),
        dolby_id=15,
    ),
    "7.1.4": _layout(
        "7.1.4",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf", "Ltr", "Rtr"),
        dolby_id=16,
        cavern_spec="7.1.4",
    ),
    "7.1.6": _layout(
        "7.1.6",
        (
            "L",
            "R",
            "C",
            "LFE",
            "Ls",
            "Rs",
            "Lrs",
            "Rrs",
            "Ltf",
            "Rtf",
            "Ltm",
            "Rtm",
            "Ltr",
            "Rtr",
        ),
        dolby_id=17,
    ),
    "9.1.2": _layout(
        "9.1.2",
        ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Lw", "Rw", "Ltm", "Rtm"),
        dolby_id=18,
    ),
    "9.1.4": _layout(
        "9.1.4",
        (
            "L",
            "R",
            "C",
            "LFE",
            "Ls",
            "Rs",
            "Lrs",
            "Rrs",
            "Lw",
            "Rw",
            "Ltf",
            "Rtf",
            "Ltr",
            "Rtr",
        ),
        dolby_id=19,
        cavern_spec="9.1.4",
    ),
    "9.1.6": _layout(
        "9.1.6",
        (
            "L",
            "R",
            "C",
            "LFE",
            "Ls",
            "Rs",
            "Lrs",
            "Rrs",
            "Lw",
            "Rw",
            "Ltf",
            "Rtf",
            "Ltm",
            "Rtm",
            "Ltr",
            "Rtr",
        ),
        dolby_id=20,
    ),
    "binaural": _layout("binaural", ("L", "R")),
}

# Cavern's WAVE 9.1.6 / 7.1.6 layouts use top-centre channels instead of
# top-middle L/R, so they are kept as separate, honest layouts.
LAYOUTS["9.1.6-cavern"] = _layout(
    "9.1.6-cavern",
    (
        "L",
        "R",
        "C",
        "LFE",
        "Lrs",
        "Rrs",
        "Ls",
        "Rs",
        "Lw",
        "Rw",
        "Ltf",
        "Ctf",
        "Rtf",
        "Ltr",
        "Ctr",
        "Rtr",
    ),
    cavern_spec="9.1.6",
)
LAYOUTS["7.1.6-cavern"] = _layout(
    "7.1.6-cavern",
    ("L", "R", "C", "LFE", "Lrs", "Rrs", "Ls", "Rs", "Ltf", "Ctf", "Rtf", "Ltr", "Ctr", "Rtr"),
    cavern_spec="7.1.6",
)

ALIASES = {
    "stereo": "2.0",
    "2": "2.0",
    "5": "5.1",
    "51": "5.1",
    "71": "7.1",
    "714": "7.1.4",
    "7.1.4-dolby": "7.1.4",
    "914": "9.1.4",
    "916": "9.1.6",
    "916-cavern": "9.1.6-cavern",
    "714-cavern": "7.1.4",
}


def get_layout(name: str) -> Layout:
    key = name.strip()
    key = ALIASES.get(key.lower(), key)
    try:
        return LAYOUTS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(LAYOUTS))
        raise KeyError(f"unknown layout {name!r}; valid layouts: {valid}") from exc


def channel_indices(labels: tuple[str, ...], selected: Iterable[str] | None = None) -> list[int]:
    if selected is None:
        return list(range(len(labels)))
    index = {label: i for i, label in enumerate(labels)}
    missing = [label for label in selected if label not in index]
    if missing:
        raise KeyError(f"channels not present in layout: {missing}; layout labels: {labels}")
    return [index[label] for label in selected]


def reorder_channels(
    data: np.ndarray,
    source_labels: Iterable[str],
    target_labels: Iterable[str],
) -> np.ndarray:
    """Reorder ``data`` (samples, channels) from source to target labels."""
    source = tuple(source_labels)
    target = tuple(target_labels)
    if source == target:
        return data
    index = {label: i for i, label in enumerate(source)}
    missing = [label for label in target if label not in index]
    if missing:
        raise KeyError(f"cannot reorder channels; missing {missing} in {source}; target {target}")
    if len(set(source)) != len(source):
        raise ValueError(f"source channel labels contain duplicates: {source}")
    order = [index[label] for label in target]
    return data[:, order]
