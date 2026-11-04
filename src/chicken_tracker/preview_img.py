"""Small picture helpers for the progress dashboard."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _tile(path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    t = np.full((size, size, 3), 30, np.uint8)
    if img is None:
        return t
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    t[y:y + img.shape[0], x:x + img.shape[1]] = img
    return t


def crop_grid(paths: list, size: int = 96, cols: int = 10) -> np.ndarray | None:
    tiles = [_tile(p, size) for p in paths]
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.full((size, size, 3), 30, np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def hen_rows(rows: dict[str, list], size: int = 96, per_row: int = 10, label_w: int = 190) -> np.ndarray | None:
    """One row per hen: her name, then crops the recogniser is sure are her."""
    out = []
    for name, paths in rows.items():
        tiles = [_tile(p, size) for p in paths[:per_row]]
        tiles += [np.full((size, size, 3), 30, np.uint8)] * (per_row - len(tiles))
        label = np.full((size, label_w, 3), 30, np.uint8)
        cv2.putText(label, name.split(" (")[0], (8, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if " (" in name:
            cv2.putText(label, "(" + name.split(" (")[1], (8, size // 2 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (170, 170, 170), 1)
        out.append(np.hstack([label] + tiles))
    return np.vstack(out) if out else None
