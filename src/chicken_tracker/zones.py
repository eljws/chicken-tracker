"""Named polygon zones (feeder, waterer, nest box, ...) in pixel coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

Zones = dict[str, np.ndarray]


def load_zones(path: str | Path | None) -> Zones:
    """Load ``{"feeder": [[x, y], ...], ...}`` from a JSON file."""
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    zones: Zones = {}
    for name, pts in raw.items():
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
            raise ValueError(f"Zone {name!r} must be a list of at least 3 [x, y] points")
        zones[name] = arr
    return zones


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting test."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)
    px, py = poly[:, 0], poly[:, 1]
    n = len(poly)
    for i in range(n):
        x1, y1 = px[i], py[i]
        x2, y2 = px[(i + 1) % n], py[(i + 1) % n]
        crosses = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < x_at)
    return inside
