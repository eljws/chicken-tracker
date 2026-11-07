"""Friends and relationships between the hens, for the Metrics page.

Everything comes from where each named hen stands on the ground (feet point, in metres via the clip's
calibration), taken once a second (the median of her detections in that second; partly-hidden boxes and
infrared stretches left out). Per clip, for every pair of hens:

  together   seconds both are on camera and within CLOSE_M of each other (side by side), out of the
             seconds both are on camera - so a hen seen less isn't counted as less friendly
  nearest    seconds hen A's nearest visible hen is B (her "best friend" is the one most often nearest)
  alone      seconds hen A's nearest visible hen is over ALONE_M away (only when another hen is on camera)
  follows    seconds A walks where B walked 2-4 s before, both moving (and not the other way round too)
  displaces  A walks up to B (B standing), and within 3 s B moves away while A stays: pecking order
  chases     A runs at B while B runs away from her, close behind, for at least CHASE_MIN_S (positions
             every CHASE_STEP_S here, since a chase lasts only a second or two)

and the flock's spread (mean distance of the hens to their centre, while >= 3 are on camera) per
10 minutes. Cached per clip in social.json, redone when the clip's files change. A tracker swapping two
hens can fake any of these for a moment; whole-day numbers smooth that out, and every displacement links
to the player so it can be checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import load_calibration, to_ground
from .metrics import BIN_S, HENS, clip_start_s

VERSION = 3
CHASE_STEP_S = 0.25
CHASE_V = 0.5      # m/s, both hurrying (walking is ~0.1-0.4); checked by eye on 3 Nov: ~12 of 14 real
CHASE_GAP_M = 1.5  # chaser within this of the chased
CHASE_AIM = 0.7    # cos: chaser heads at the chased, chased heads away from the chaser
CHASE_MIN_S = 0.75
CHASE_MERGE_S = 0.5
CLOSE_M = 0.75     # side by side
ALONE_M = 2.5      # nobody within this: on her own
FOLLOW_M = 0.5     # A is now within this of where B was
FOLLOW_LAGS = (2, 3, 4)
MOVE_M = 0.6       # both walked at least this far over the lag
DISP_NEAR_M = 0.6  # displacement: A gets this close...
DISP_AWAY_M = 0.8  # ...then B moves at least this far within DISP_S...
DISP_S = 3
_SOURCES = ("detections_named.csv", "calibration.json", "daylight.json")
FILE = "social.json"


def positions(run: Path, step: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """(times s [S], xy [S, hens, 2] metres with NaN where unseen, hen names) for one clip, one sample
    per `step` seconds."""
    H = load_calibration(run)
    p = run / "detections_named.csv"
    if H is None or not p.exists():
        return None
    d = pd.read_csv(p, usecols=["time_s", "cx", "y2", "chicken", "partial"])
    d = d[d.chicken.isin(HENS) & ~d.partial.astype(bool)]
    if d.empty:
        return None
    from . import daylight
    ir = daylight.infrared_s(run, compute=False)
    if ir:
        t = d.time_s.to_numpy()
        d = d[~np.any([(t >= a) & (t <= b) for a, b in ir], axis=0)]
    g = to_ground(H, d.cx.to_numpy(), d.y2.to_numpy())
    d = d.assign(sec=(d.time_s / step).astype(int), gx=g[:, 0], gy=g[:, 1])
    m = d.groupby(["sec", "chicken"])[["gx", "gy"]].median()
    secs = np.arange(int(d.sec.max()) + 1)
    hens = [h for h in HENS if h in set(d.chicken)]
    xy = np.full((len(secs), len(hens), 2), np.nan)
    for k, h in enumerate(hens):
        mh = m.xs(h, level="chicken")
        xy[mh.index.to_numpy(), k] = mh.to_numpy()
    return secs * step, xy, hens


def chases(ts: np.ndarray, xy: np.ndarray, hens: list[str]) -> list[dict]:
    """Chase episodes {a chaser, b chased, t start s, s duration, v top speed m/s} from fine positions."""
    n, step = len(hens), CHASE_STEP_S
    if len(ts) < 4 or n < 2:
        return []
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) / (2 * step)                       # velocity over half a second
    sp = np.linalg.norm(v, axis=-1)
    out = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ab = xy[:, b] - xy[:, a]
            dist = np.linalg.norm(ab, axis=-1)
            u = ab / np.maximum(dist, 1e-6)[:, None]
            aim_a = (v[:, a] * u).sum(-1) / np.maximum(sp[:, a], 1e-6)
            aim_b = (v[:, b] * u).sum(-1) / np.maximum(sp[:, b], 1e-6)
            on = np.nan_to_num((sp[:, a] >= CHASE_V) & (sp[:, b] >= CHASE_V) & (dist < CHASE_GAP_M)
                               & (aim_a > CHASE_AIM) & (aim_b > CHASE_AIM), nan=0).astype(bool)
            idx = np.flatnonzero(on)
            if not len(idx):
                continue
            runs, start, prev = [], idx[0], idx[0]
            for i in idx[1:]:
                if (i - prev) * step > CHASE_MERGE_S + 1e-9:
                    runs.append((start, prev))
                    start = i
                prev = i
            runs.append((start, prev))
            for i, j in runs:
                dur = (j - i + 1) * step
                if dur >= CHASE_MIN_S:
                    out.append({"a": hens[a], "b": hens[b], "t": round(float(ts[i]), 2), "s": dur,
                                "v": round(float(np.nanmax(sp[i:j + 1, [a, b]])), 2)})
    # a pair seen both ways at once (side by side, running together) is not a chase
    out.sort(key=lambda e: e["t"])
    keep = [e for e in out if not any(o["a"] == e["b"] and o["b"] == e["a"] and abs(o["t"] - e["t"]) < 1 for o in out)]
    return keep


def analyse(secs: np.ndarray, xy: np.ndarray, hens: list[str], start_s: float) -> dict:
    """The clip's numbers (see the module notes); pair tables are {A: {B: seconds}}."""
    n = len(hens)
    seen = ~np.isnan(xy[..., 0])                                         # [S, n]
    dist = np.linalg.norm(xy[:, :, None] - xy[:, None, :], axis=-1)       # [S, n, n], NaN if either unseen
    both = seen[:, :, None] & seen[:, None, :]
    eye = np.eye(n, dtype=bool)
    together = ((dist < CLOSE_M) & both & ~eye).sum(0)
    covis = (both & ~eye).sum(0)
    dn = np.where(both & ~eye, dist, np.inf)
    near = dn.argmin(2)                                                  # [S, n]
    has = np.isfinite(dn.min(2)) & seen
    nearest = np.zeros((n, n), int)
    for i in range(n):
        nearest[i] = np.bincount(near[has[:, i], i], minlength=n)
    alone = ((dn.min(2) > ALONE_M) & has).sum(0)
    with_others = has.sum(0)

    # following: A now near where B was `lag` s ago, both walked; ambiguous both-ways moments dropped
    follow = np.zeros((n, n), int)
    S = len(secs)
    fl = np.zeros((S, n, n), bool)
    for lag in FOLLOW_LAGS:
        if S <= lag:
            continue
        now, before = xy[lag:], xy[:-lag]
        gap = np.linalg.norm(now[:, :, None] - before[:, None, :], axis=-1)   # A now vs B before
        moved = np.linalg.norm(now - before, axis=-1)                          # [S-lag, n]
        walk = moved >= MOVE_M
        f = (gap < FOLLOW_M) & walk[:, :, None] & walk[:, None, :] & (dist[lag:] < 2.0) & ~eye
        fl[lag:] |= np.nan_to_num(f, nan=0).astype(bool)
    fl &= ~fl.transpose(0, 2, 1)
    follow = fl.sum(0)

    # displacements: A closes in on a standing B, then B leaves while A stays
    disp = np.zeros((n, n), int)
    events, last = [], {}
    for t in range(DISP_S, S - DISP_S):
        for a in range(n):
            for b in range(n):
                if a == b or not (dist[t, a, b] < DISP_NEAR_M):
                    continue
                pa0, pb0, pa, pb = xy[t - DISP_S, a], xy[t - DISP_S, b], xy[t, a], xy[t, b]
                pa1, pb1 = xy[t + DISP_S, a], xy[t + DISP_S, b]
                if np.isnan([*pa0, *pb0, *pa1, *pb1]).any():
                    continue
                came = np.linalg.norm(pa - pa0) >= 0.3 and np.linalg.norm(pb - pb0) < 0.3
                left = np.linalg.norm(pb1 - pb) >= DISP_AWAY_M and np.linalg.norm(pa1 - pa) < 0.5 * np.linalg.norm(pb1 - pb)
                if came and left and dist[t + DISP_S, a, b] > dist[t, a, b] + 0.5 and t - last.get((a, b), -99) > 10:
                    last[(a, b)] = t
                    disp[a, b] += 1
                    events.append({"a": hens[a], "b": hens[b], "t": float(secs[t])})

    # flock spread per 10 min of clock time
    spread = {}
    many = seen.sum(1) >= 3
    if many.any():
        c = np.nanmean(xy[many], axis=1, keepdims=True)
        sp = np.nanmean(np.linalg.norm(xy[many] - c, axis=-1), axis=1)
        for s, v in zip(secs[many], sp):
            b = int((start_s + s) // BIN_S * BIN_S)
            r = spread.setdefault(b, [0.0, 0])
            r[0] += float(v)
            r[1] += 1

    tab = lambda a: {hens[i]: {hens[j]: int(a[i, j]) for j in range(n) if j != i} for i in range(n)}
    return {"hens": hens, "together": tab(together), "covisible": tab(covis), "nearest": tab(nearest),
            "follows": tab(follow), "displaces": tab(disp), "events": events,
            "alone": {h: int(alone[i]) for i, h in enumerate(hens)},
            "with_others": {h: int(with_others[i]) for i, h in enumerate(hens)},
            "spread": {str(b): [round(v[0], 2), v[1]] for b, v in spread.items()}}


def clip_social(run: Path) -> dict | None:
    run = Path(run)
    start = clip_start_s(run)
    if start is None or not (run / "detections_named.csv").exists() or not (run / "calibration.json").exists():
        return None
    stamp = [VERSION] + [round((run / f).stat().st_mtime, 3) if (run / f).exists() else 0 for f in _SOURCES]
    cache = run / FILE
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("stamp") == stamp:
                return c
        except ValueError:
            pass
    p = positions(run)
    out = {"stamp": stamp, "start_s": start, **(analyse(*p, start) if p else {"hens": []})}
    if p:
        f = positions(run, CHASE_STEP_S)
        out["chases"] = chases(*f) if f else []
    cache.write_text(json.dumps(out))
    return out


def day_social(root: Path, runs: list[Path], day: str) -> dict:
    """The day's relationships, summed over its named clips (seconds), plus shares the page shows."""
    root = Path(root)
    keys = ("together", "covisible", "nearest", "follows", "displaces")
    tot = {k: {} for k in keys}
    alone, others, spread, events, chase = {}, {}, {}, [], []
    for run in runs:
        rel = str(run.relative_to(root))
        if not rel.startswith(day):
            continue
        c = clip_social(run)
        if not c or not c.get("hens"):
            continue
        for k in keys:
            for a, row in c[k].items():
                for b, v in row.items():
                    tot[k].setdefault(a, {}).setdefault(b, 0)
                    tot[k][a][b] += v
        for h, v in c["alone"].items():
            alone[h] = alone.get(h, 0) + v
            others[h] = others.get(h, 0) + c["with_others"][h]
        for b, (s, n) in c["spread"].items():
            r = spread.setdefault(int(b), [0.0, 0])
            r[0] += s
            r[1] += n
        events += [{**e, "run": rel, "clock_s": c["start_s"] + e["t"]} for e in c["events"]]
        chase += [{**e, "run": rel, "clock_s": c["start_s"] + e["t"]} for e in c.get("chases", [])]
    hens = [h for h in HENS if h in others]
    share = lambda a, b: (tot["together"].get(a, {}).get(b, 0) / tot["covisible"][a][b]
                          if tot["covisible"].get(a, {}).get(b) else None)
    best = {}
    for h in hens:
        row = tot["nearest"].get(h, {})
        n = sum(row.values())
        if n:
            f = max(row, key=row.get)
            best[h] = {"hen": f, "share": round(row[f] / n, 3)}
    # pecking order: share of a hen's displacement encounters she won (ranks the flock)
    wins = {h: sum(tot["displaces"].get(h, {}).values()) for h in hens}
    losses = {h: sum(tot["displaces"].get(o, {}).get(h, 0) for o in hens) for h in hens}
    order = sorted(hens, key=lambda h: (-(wins[h] / (wins[h] + losses[h])) if wins[h] + losses[h] else 1, -wins[h]))
    chased = {a: {b: sum(e["a"] == a and e["b"] == b for e in chase) for b in hens if b != a} for a in hens}
    return {"hens": hens, "close_m": CLOSE_M,
            "chases": sorted(chase, key=lambda e: e["clock_s"]), "chase_matrix": chased,
            "chase_rule": {"v": CHASE_V, "gap_m": CHASE_GAP_M, "min_s": CHASE_MIN_S}, "alone_m": ALONE_M,
            "together_share": {a: {b: (round(share(a, b), 3) if share(a, b) is not None else None) for b in hens if b != a} for a in hens},
            **tot, "best_friend": best,
            "alone_share": {h: round(alone[h] / others[h], 3) if others[h] else None for h in hens},
            "wins": wins, "losses": losses, "pecking_order": order,
            "events": sorted(events, key=lambda e: e["clock_s"]),
            "spread": {str(b): round(s / n, 2) for b, (s, n) in sorted(spread.items()) if n >= 60}}
