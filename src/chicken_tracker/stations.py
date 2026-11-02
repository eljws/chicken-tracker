"""Feeder / water stations and per-chicken visits to them.

Stations come from feeder/water boxes labelled in the dataset (``review --objects``).
The camera is static, so labelled positions apply to the whole video. (One camera is
assumed for now; with several cameras, keep one dataset per camera.)

A chicken is *at* a station while her box touches the station box widened by a margin.
Visits merge gaps of up to ``gap_s`` and must last at least ``min_visit_s``, so a chicken
walking past doesn't count as eating or drinking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labeling import CLASSES, LabelStore

STATION_CLASSES = {k: v for k, v in CLASSES.items() if k != 0}  # 1 feeder, 2 water


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def stations_from_labels(dataset: Path, min_iou: float = 0.3) -> list[dict]:
    """Group labelled feeder/water boxes by position; each group is one station (median box)."""
    store = LabelStore(dataset)
    groups: list[dict] = []
    for it in store.items.values():
        if it.get("obj_status") != "done":
            continue
        for *box, cls in it.get("objects", []):
            g = next((g for g in groups if g["cls"] == cls and _iou(g["boxes"][0], box) > min_iou), None)
            if g is None:
                groups.append({"cls": cls, "boxes": [box]})
            else:
                g["boxes"].append(box)
    out, count = [], {}
    for g in sorted(groups, key=lambda g: (g["cls"], -len(g["boxes"]))):
        kind = STATION_CLASSES[g["cls"]]
        count[kind] = count.get(kind, 0) + 1
        out.append({"id": f"{kind} {count[kind]}", "type": kind,
                    "box": [round(float(v), 1) for v in np.median(np.array(g["boxes"]), axis=0)],
                    "labelled_in_frames": len(g["boxes"])})
    return out


def save_stations(run_dir: Path, stations: list[dict]) -> Path:
    p = Path(run_dir) / "stations.json"
    p.write_text(json.dumps(stations, indent=1))
    return p


def load_stations(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "stations.json"
    return json.loads(p.read_text()) if p.exists() else []


def at_station(det: pd.DataFrame, box, margin: float) -> np.ndarray:
    """Per row: does the chicken's box touch the station box widened by `margin` px?"""
    x1, y1, x2, y2 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
    return ((det.x2.to_numpy() >= x1) & (det.x1.to_numpy() <= x2) &
            (det.y2.to_numpy() >= y1) & (det.y1.to_numpy() <= y2))


def station_visits(named: pd.DataFrame, stations: list[dict], fps: float, margin: float = 30.0,
                   gap_s: float = 2.0, min_visit_s: float = 2.0) -> pd.DataFrame:
    """One row per (chicken, station, visit). `named` = detections_named.csv."""
    cols = ["chicken", "station", "type", "start_s", "end_s", "duration_s"]
    rows = []
    for st in stations:
        hit = named[at_station(named, st["box"], margin)]
        for chicken, g in hit.groupby("chicken"):
            t = np.sort(g.time_s.to_numpy())
            breaks = np.flatnonzero(np.diff(t) > gap_s) + 1
            for seg in np.split(t, breaks):
                dur = seg[-1] - seg[0] + 1 / fps
                if dur >= min_visit_s:
                    rows.append((chicken, st["id"], st["type"], seg[0], seg[-1] + 1 / fps, dur))
    return pd.DataFrame(rows, columns=cols).sort_values(["chicken", "start_s"]) if rows else pd.DataFrame(columns=cols)


def write_station_analytics(run_dir: Path, fps: float, **kw) -> dict | None:
    """station_visits.csv + per-type visit counts/times merged into chickens.csv."""
    run_dir = Path(run_dir)
    stations = load_stations(run_dir)
    named_p, chickens_p = run_dir / "detections_named.csv", run_dir / "chickens.csv"
    if not stations or not named_p.exists() or not chickens_p.exists():
        return None
    visits = station_visits(pd.read_csv(named_p), stations, fps, **kw)
    visits.to_csv(run_dir / "station_visits.csv", index=False, float_format="%.2f")
    summary = pd.read_csv(chickens_p)
    summary = summary.drop(columns=[c for c in summary.columns
                                    if c.endswith(("_visits", "_time_s")) and c.split("_")[0] in
                                    set(STATION_CLASSES.values())])
    for kind in sorted({s["type"] for s in stations}):
        v = visits[visits.type == kind].groupby("chicken")
        summary[f"{kind}_visits"] = summary.chicken.map(v.size()).fillna(0).astype(int)
        summary[f"{kind}_time_s"] = summary.chicken.map(v.duration_s.sum()).fillna(0.0)
    summary.to_csv(chickens_p, index=False, float_format="%.2f")
    return {"stations": len(stations), "visits": len(visits)}
