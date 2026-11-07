"""How big a hen's box should be: by where she stands (distance) and by which hen she is.

Learned from your confirmed pieces across every clip (one fixed camera, one ground calibration):

    log(box area) = a + b * log(pixels per metre of ground under her feet) + hen size

* distance: the ground calibration (calibration.json, patio slabs) says how many pixels a metre
  of ground covers at each spot; boxes shrink as that shrinks (b near 2 = plain perspective).
* hen size: each hen's own factor against the average hen (Monster's box ~1.3x, Ginger's ~0.8x).

Only clean daytime boxes count: confirmed, not in infrared, not flagged partial/loose, no other box overlapping, not
cut off by the picture edge. Saved as <output>/size_calibration.json; the naming page's checks
then judge a named hen's box against *her* usual size at that spot, and set-aside pieces
against the average hen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FILE = "size_calibration.json"
EDGE_PX = 5          # boxes touching the picture edge are cut off: not used
EVERY_N = 5          # every 5th frame is plenty (neighbouring frames are near-identical)
BIN_M = 0.5          # distance bins for the chart
GRID_PX = (160, 128)  # location correction cells (x, y) - the ground isn't one flat plane (lawn vs patio)
GRID_MIN = 30         # boxes a cell needs before its correction is trusted
MIN_BIN = 40         # boxes needed for a bin to be drawn


def pixels_per_metre(H: np.ndarray, u, v, e: float = 2.0) -> np.ndarray:
    """Local scale of the ground at image points (u, v): sqrt of px^2 per m^2."""
    u, v = np.asarray(u, float), np.asarray(v, float)

    def g(uu, vv):
        p = H @ np.vstack([uu, vv, np.ones_like(uu)])
        return p[0] / p[2], p[1] / p[2]
    x0, y0 = g(u, v)
    x1, y1 = g(u + e, v)
    x2, y2 = g(u, v + e)
    m2_per_px2 = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / (e * e)
    return 1 / np.sqrt(m2_per_px2)


def metres_from_near_edge(H: np.ndarray, u, v, width: int, height: int) -> np.ndarray:
    """Ground distance (m) from the middle of the picture's bottom edge - the spot nearest the camera."""
    p = H @ np.vstack([np.asarray(u, float), np.asarray(v, float), np.ones(len(np.atleast_1d(u)))])
    q = H @ np.array([width / 2, height - 1, 1.0])
    return np.hypot(p[0] / p[2] - q[0] / q[2], p[1] / p[2] - q[1] / q[2])


def clean_boxes(run: Path) -> pd.DataFrame:
    """Confirmed single-hen boxes of one clip with their ground scale and distance."""
    from . import identity as I
    from .calibration import load_calibration
    from .recognizer import _crowding

    H = load_calibration(run)
    meta = json.loads((run / "meta.json").read_text())
    W, Hh = meta["video"]["width"], meta["video"]["height"]
    d = json.loads((run / "identity" / "identity.json").read_text())
    if H is None:
        return pd.DataFrame()
    det = I.load_detections(run, d)
    sure, bad = I.confirmed_by_you(d), I.partial(d) | I.excluded(d)  # your answers only, not Claude's
    lab = {t: c["name"] for c in d["chickens"] if c["name"] for t in c["tracks"] if t in sure and t not in bad}
    det = det.assign(crowd=_crowding(det))
    det = det[det.track_id.isin(lab) & (det.crowd == 0) & (det.frame % EVERY_N == 0)]
    for k in d.get("dismissed_checks", []):  # big boxes you said are still her: unusual poses, not her size
        kind, _, rest = k.partition(":")
        if kind == "merged_inside":
            try:
                tt, f0, f1 = map(int, rest.split(":"))
            except ValueError:
                continue
            det = det[~((det.track_id == tt) & (det.frame >= f0) & (det.frame <= f1))]
    from . import daylight as DL
    ir = DL.infrared_s(run)  # infrared boxes are out of scope (and the picture differs): left out
    if ir:
        det = det[~DL.in_infrared(det.frame.to_numpy(), d["fps"], ir)]
    det = det[(det.x1 > EDGE_PX) & (det.y1 > EDGE_PX) & (det.x2 < W - EDGE_PX) & (det.y2 < Hh - EDGE_PX)]
    u, v = ((det.x1 + det.x2) / 2).to_numpy(), det.y2.to_numpy()
    return pd.DataFrame({"hen": det.track_id.map(lab).to_numpy(), "area": ((det.x2 - det.x1) * (det.y2 - det.y1)).to_numpy(),
                         "ppm": pixels_per_metre(H, u, v), "metres": metres_from_near_edge(H, u, v, W, Hh), "u": u, "v": v})


def build(root: Path, runs: list[Path]) -> dict:
    from .metrics import HENS

    parts, used = [], 0
    for run in runs:
        if (run / "identity" / "identity.json").exists() and (run / "calibration.json").exists():
            try:
                b = clean_boxes(run)
            except (OSError, ValueError, KeyError):
                continue
            if len(b):
                parts.append(b)
                used += 1
    if not parts:
        raise ValueError("no confirmed pieces with a ground calibration yet")
    df = pd.concat(parts, ignore_index=True)
    hens = [h for h in HENS if h in set(df.hen)] + sorted(set(df.hen) - set(HENS))
    X = np.column_stack([(df.hen == h).to_numpy(float) for h in hens] + [np.log(df.ppm.to_numpy())])
    y = np.log(df.area.to_numpy())
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, hcoef = float(coef[-1]), dict(zip(hens, coef[:-1]))
    a = float(np.mean(list(hcoef.values())))  # the "average hen"
    res = y - X @ coef
    # where in the picture boxes still come out bigger/smaller than the ground calibration says
    # (raised lawn, slope, camera lens): a per-cell correction, then the residual spread without it
    gx, gy = (df.u.to_numpy() // GRID_PX[0]).astype(int), (df.v.to_numpy() // GRID_PX[1]).astype(int)
    cells = pd.DataFrame({"gx": gx, "gy": gy, "r": res}).groupby(["gx", "gy"]).r.agg(["median", "size"])
    grid = {f"{x},{yy}": round(float(m), 3) for (x, yy), (m, n) in cells.iterrows() if n >= GRID_MIN}
    res = res - np.array([grid.get(f"{a},{b}", 0.0) for a, b in zip(gx, gy)])
    out_hens = {}
    for h in hens:
        r = res[(df.hen == h).to_numpy()]
        out_hens[h] = {"factor": round(float(np.exp(hcoef[h] - a)), 3), "boxes": int(len(r)),
                       # how much her box varies at one spot (posture: standing tall, pecking, side-on)
                       "p10": round(float(np.exp(np.percentile(r, 10))), 2), "p25": round(float(np.exp(np.percentile(r, 25))), 2),
                       "p75": round(float(np.exp(np.percentile(r, 75))), 2), "p90": round(float(np.exp(np.percentile(r, 90))), 2)}
    # chart: median box side (sqrt area, px) per hen per half-metre of distance
    df["bin"] = (df.metres // BIN_M) * BIN_M + BIN_M / 2
    curve = {}
    for h in hens:
        g = df[df.hen == h].groupby("bin").area
        cnt, med = g.size(), g.median()
        curve[h] = [[round(float(m), 2), round(float(np.sqrt(med[m])), 1), int(cnt[m])] for m in med.index if cnt[m] >= MIN_BIN]
    cal = {"a": a, "b": b, "grid_px": list(GRID_PX), "grid": grid, "hens": out_hens, "curve": curve, "built": time.time(), "clips": used, "boxes": int(len(df)),
           "spread_all": [round(float(np.exp(np.percentile(res, q))), 2) for q in (10, 25, 75, 90)]}
    (Path(root) / FILE).write_text(json.dumps(cal, indent=1))
    return cal


def load(run_or_root: Path) -> dict | None:
    """The calibration in this folder or the nearest folder above it (clips sit in output/<day>/)."""
    p = Path(run_or_root).resolve()
    for d in (p, *p.parents):
        f = d / FILE
        if f.exists():
            try:
                return json.loads(f.read_text())
            except ValueError:
                return None
        if (d / "pyproject.toml").exists():  # don't wander out of the project
            break
    return None


def expected_area(cal: dict, H: np.ndarray, u, v, hen: str | None = None) -> np.ndarray:
    """Usual box area (px^2) at feet point (u, v) - for this hen, or the average hen."""
    h = cal["hens"].get(hen, {}).get("factor", 1.0) if hen else 1.0
    u, v = np.asarray(u, float), np.asarray(v, float)
    corr = np.zeros(len(np.atleast_1d(u)))
    if cal.get("grid"):
        gw, gh = cal["grid_px"]
        corr = np.array([cal["grid"].get(f"{int(a // gw)},{int(b // gh)}", 0.0) for a, b in zip(np.atleast_1d(u), np.atleast_1d(v))])
    return np.exp(cal["a"] + cal["b"] * np.log(pixels_per_metre(H, u, v)) + corr) * h
