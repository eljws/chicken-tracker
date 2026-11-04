"""Pecking: the head moves while the body stays still.

Uses the skeletons (``poses.csv``) for where each hen's head and back are in every frame,
and the video's pixel changes for what moved. For each hen in each frame:

* head motion  = mean absolute change since the previous frame, in a small circle at her head
* body motion  = the same at the middle of her back
* peck signal  = head motion minus body motion (walking moves both, so it cancels)
* body shift   = how far her back point moved since the previous frame, relative to her size
                 (a plain white body shows little pixel change even while walking)

A peck is a short peak in the peck signal while the body is still, at least ``min_gap_s``
after the previous one. Frame-to-frame change (not a running background) is used, so a
hen that just walked away leaves no "ghost". Pecks at a feeder count as eating; elsewhere
they're foraging.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .stations import at_station, load_stations

MIN_CONF = 0.5      # skeleton points less sure than this are ignored
BODY_STILL = 6.0    # body motion (grey levels) above this = she's walking, not pecking
BODY_SHIFT = 0.12   # ... or her back point moving more than 12% of her head-back length per frame
PECK_MIN = 6.0      # peck signal peak must reach this ...
MIN_GAP_S = 0.2     # ... and be at least this long after the previous peck


def _disk_motion(diff: np.ndarray, x: float, y: float, r: int) -> float:
    h, w = diff.shape
    x0, x1, y0, y1 = max(0, int(x) - r), min(w, int(x) + r + 1), max(0, int(y) - r), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    patch = diff[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    return float(patch[m].mean()) if m.any() else np.nan


def motion_signals(video: str, poses: pd.DataFrame, progress: bool = True, on_frame=None,
                   on_picture=None) -> pd.DataFrame:
    """Add head_motion / body_motion columns to poses (one row per hen per frame)."""
    poses = poses.copy()
    ok = (poses.head_c >= MIN_CONF) & (poses.back_c >= MIN_CONF)
    want = poses[ok]
    # radius: about a head's size, from the head-to-back distance
    hb = np.hypot(want.head_x - want.back_x, want.head_y - want.back_y)
    radius = np.clip(0.3 * hb, 5, 20).round().astype(int)
    by_frame = {f: idx for f, idx in want.groupby("frame").indices.items()}
    rows_idx = want.index.to_numpy()
    hx, hy, bx, by = (want[c].to_numpy() for c in ("head_x", "head_y", "back_x", "back_y"))
    head = np.full(len(poses), np.nan)
    body = np.full(len(poses), np.nan)
    pos = {i: k for k, i in enumerate(poses.index)}
    cap = cv2.VideoCapture(str(video))
    prev, f = None, 0
    last = max(by_frame) if by_frame else -1
    while f <= last:
        ok_, img = cap.read()
        if not ok_:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None and f in by_frame:
            diff = np.abs(g - prev).astype(np.float32)
            for k in by_frame[f]:
                i = pos[rows_idx[k]]
                head[i] = _disk_motion(diff, hx[k], hy[k], radius.iat[k])
                body[i] = _disk_motion(diff, bx[k], by[k], radius.iat[k])
            if on_picture is not None and on_picture():
                heat = cv2.applyColorMap(np.clip(diff * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                pic = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
                for k in by_frame[f]:  # head (yellow) and back (cyan) circles of each hen
                    r_ = int(radius.iat[k])
                    cv2.circle(pic, (int(hx[k]), int(hy[k])), r_, (0, 255, 255), 2)
                    cv2.circle(pic, (int(bx[k]), int(by[k])), r_, (255, 255, 0), 2)
                on_picture(pic)
        prev = g
        if on_frame is not None:
            on_frame(f + 1, last + 1)
        if progress and f % 3000 == 0:
            print(f"  motion: frame {f}/{last}", flush=True)
        f += 1
    cap.release()
    poses["head_motion"], poses["body_motion"] = head, body
    # back-point movement per frame, relative to head-back length (same hen, consecutive frames)
    poses = poses.sort_values(["chicken", "frame"])
    hb_all = np.hypot(poses.head_x - poses.back_x, poses.head_y - poses.back_y).clip(lower=5)
    step = np.hypot(poses.groupby("chicken").back_x.diff(), poses.groupby("chicken").back_y.diff())
    consecutive = poses.groupby("chicken").frame.diff() == 1
    poses["body_shift"] = (step / hb_all).where(consecutive)
    return poses.sort_index()


def find_pecks(sig: pd.DataFrame, fps: float, stations: list[dict] | None = None,
               peck_min: float = PECK_MIN, body_still: float = BODY_STILL, body_shift: float = BODY_SHIFT,
               min_gap_s: float = MIN_GAP_S) -> pd.DataFrame:
    """One row per peck: chicken, frame, time_s, strength, where (station id or 'ground')."""
    out = []
    gap = max(1, int(round(min_gap_s * fps)))
    for chicken, g in sig.dropna(subset=["head_motion", "body_motion"]).sort_values("frame").groupby("chicken"):
        fr = g.frame.to_numpy()
        s = (g.head_motion - g.body_motion).to_numpy()
        shift = g.body_shift.fillna(0).rolling(3, center=True, min_periods=1).max().to_numpy()
        still = (g.body_motion.to_numpy() <= body_still) & (shift <= body_shift)
        last = -10 ** 9
        for k in range(len(s)):
            # local peak among direct neighbours (in time, same hen), above threshold, body still
            if not still[k] or s[k] < peck_min:
                continue
            left = s[k - 1] if k and fr[k - 1] == fr[k] - 1 else -np.inf
            right = s[k + 1] if k + 1 < len(s) and fr[k + 1] == fr[k] + 1 else -np.inf
            if s[k] < left or s[k] < right or fr[k] - last < gap:
                continue
            last = fr[k]
            out.append({"chicken": chicken, "frame": int(fr[k]), "time_s": round(fr[k] / fps, 2),
                        "strength": round(float(s[k]), 1), "head_x": g.head_x.iat[k], "head_y": g.head_y.iat[k]})
    pecks = pd.DataFrame(out, columns=["chicken", "frame", "time_s", "strength", "head_x", "head_y"])
    pecks["where"] = "ground"
    for st in stations or []:
        # the head (not the whole box) must be at the station
        hb = pd.DataFrame({"x1": pecks.head_x, "x2": pecks.head_x, "y1": pecks.head_y, "y2": pecks.head_y})
        pecks.loc[at_station(hb, st["box"], 15.0) & (pecks["where"] == "ground"), "where"] = st["id"]
    return pecks


def write_pecking(run_dir: Path, video: str | None = None, **kw) -> dict:
    """pecks.csv + per-hen peck counts in chickens.csv (needs poses.csv from `pose`)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    fps = meta["video"]["fps"]
    poses = pd.read_csv(run_dir / "poses.csv")
    from .progress import Progress
    with Progress(run_dir, "pecks") as prog:
        def picture(img=None):
            if img is None:
                return prog.preview_due()
            prog.preview(img)
        sig = motion_signals(video or meta["video"]["path"], poses, on_frame=lambda f, t: prog.update(f, t),
                             on_picture=picture)
    sig[["frame", "chicken", "head_motion", "body_motion", "body_shift"]].to_csv(run_dir / "peck_signal.csv", index=False,
                                                                    float_format="%.1f")
    stations = load_stations(run_dir)
    pecks = find_pecks(sig, fps, stations, **kw)
    pecks.to_csv(run_dir / "pecks.csv", index=False)
    merge_peck_counts(run_dir)
    feeders = {s["id"] for s in stations if s["type"] == "feeder"}
    return {"pecks": len(pecks), "at_feeder": int(pecks["where"].isin(feeders).sum()),
            "ground": int((pecks["where"] == "ground").sum())}


def merge_peck_counts(run_dir: Path) -> None:
    """pecks_at_feeder / pecks_ground columns in chickens.csv from pecks.csv (re-run after naming edits)."""
    run_dir = Path(run_dir)
    pecks_p, summary_p = run_dir / "pecks.csv", run_dir / "chickens.csv"
    if not pecks_p.exists() or not summary_p.exists():
        return
    pecks = pd.read_csv(pecks_p)
    feeders = {s["id"] for s in load_stations(run_dir) if s["type"] == "feeder"}
    summary = pd.read_csv(summary_p)
    summary = summary.drop(columns=[c for c in summary.columns if c.startswith("pecks_")])
    summary["pecks_at_feeder"] = summary.chicken.map(pecks[pecks["where"].isin(feeders)].groupby("chicken").size()) \
        .fillna(0).astype(int)
    summary["pecks_ground"] = summary.chicken.map(pecks[pecks["where"] == "ground"].groupby("chicken").size()) \
        .fillna(0).astype(int)
    summary.to_csv(summary_p, index=False, float_format="%.2f")
