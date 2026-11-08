"""Per-hen metrics for the Metrics page: how far each hen walked, when she was active, her runs.

A *run* is a burst of real speed: the hen's smoothed feet point moves at >= RUN_M_S for at least
RUN_MIN_S (short dips under the threshold, <= RUN_MERGE_S, don't split a run). Speeds come from
`calibration.kinematics`, which already drops impossible jumps; a tracker swapping between two
nearby hens can still fake a short dash, so every run links to the player at that moment.

Only named clips count (identity.json + chickens.csv), and only in metres (needs calibration.json).
Results are cached per clip in metrics.json and redone when the clip's files change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import kinematics, load_calibration, to_ground

HENS = ["Monster", "Ginger", "Rocky", "Salt", "Pepper"]  # fixed order = fixed colour on every chart
RUN_M_S = 0.8      # walking is ~0.1-0.4 m/s, a run clearly faster
RUN_MIN_S = 0.4
RUN_MERGE_S = 0.3
BIN_S = 600        # activity bins: 10 minutes of clock time
VERSION = 1        # bump when the computation changes, to redo cached clips

_CLIP = re.compile(r"(\d{2})(\d{2})(\d{2})-(\d{6})")
_SOURCES = ("detections_named.csv", "chickens.csv", "calibration.json", "pecks.csv", "station_visits.csv")


def clip_start_s(run: Path) -> float | None:
    """Clock time (seconds since midnight) the clip starts, from its name (HHMMSS-HHMMSS_...)."""
    m = _CLIP.search(run.name)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None


def run_events(samples, thr: float = RUN_M_S, min_s: float = RUN_MIN_S, merge_s: float = RUN_MERGE_S) -> list[dict]:
    """Runs from per-frame speed samples [(time_s, m/s, track), ...] of one hen."""
    out = []
    by_track: dict = {}
    for t, v, tid in samples:
        by_track.setdefault(tid, []).append((t, v))
    for tid, s in by_track.items():
        s.sort()
        t = np.array([x[0] for x in s])
        v = np.array([x[1] for x in s])
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.064
        fast = np.flatnonzero(v >= thr)
        if not len(fast):
            continue
        groups, cur = [], [fast[0]]
        for i in fast[1:]:
            if t[i] - t[cur[-1]] <= merge_s + dt:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for g in groups:
            a, b = g[0], g[-1]
            dur = float(t[b] - t[a] + dt)
            if dur < min_s:
                continue
            seg = slice(a, b + 1)
            out.append({"start_s": float(t[a]), "duration_s": round(dur, 2),
                        "peak_m_s": round(float(v[seg].max()), 2),
                        "metres": round(float(np.sum(v[seg]) * dt), 2), "piece": int(tid)})
    return sorted(out, key=lambda r: r["start_s"])


def _activity(d: pd.DataFrame, H, start: float, smooth: int = 7) -> dict[int, list[float]]:
    """{clock bin start: [metres walked, seconds visible]} for one hen, steps summed within tracks."""
    acc: dict[int, list[float]] = {}
    fps_dt = float(np.median(np.diff(np.sort(d.time_s.unique())))) if d.time_s.nunique() > 1 else 0.064
    for _, g in d.sort_values("frame").groupby("track_id"):
        p = to_ground(H, g.cx.to_numpy(), g.y2.to_numpy())
        if len(p) >= smooth:
            k = smooth // 2
            pad = np.pad(p, ((k, k), (0, 0)), mode="edge")
            p = np.stack([np.median(pad[i:i + smooth], axis=0) for i in range(len(p))])
        tb = ((start + g.time_s.to_numpy()) // BIN_S * BIN_S).astype(int)
        step = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
        for b in np.unique(tb):
            m = tb == b
            r = acc.setdefault(int(b), [0.0, 0.0])
            r[0] += float(step[m].sum())
            r[1] += float(m.sum() * fps_dt)
    return acc


def clip_metrics(run: Path) -> dict | None:
    run = Path(run)
    need = [run / "detections_named.csv", run / "chickens.csv", run / "calibration.json"]
    start = clip_start_s(run)
    if start is None or not all(p.exists() for p in need):
        return None
    stamp = [VERSION] + [int((run / f).stat().st_mtime) if (run / f).exists() else 0 for f in _SOURCES]
    cache = run / "metrics.json"
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("stamp") == stamp:
                return c
        except ValueError:
            pass
    H = load_calibration(run)
    det = pd.read_csv(run / "detections_named.csv")
    summ = pd.read_csv(run / "chickens.csv").set_index("chicken")
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    length = (meta.get("video", {}).get("frame_count", 0) / meta.get("video", {}).get("fps", 1)) or float(det.time_s.max())
    hens, runs, activity = {}, [], []
    for name, d in det.groupby("chicken"):
        name = str(name)
        part = d.partial.astype(bool) if "partial" in d else pd.Series(False, index=d.index)
        k = kinematics(d[~part], H)
        ev = run_events(k["samples"])
        s = summ.loc[name] if name in summ.index else {}
        get = lambda col, default=0.0: float(s[col]) if col in getattr(s, "index", []) and pd.notna(s[col]) else default
        hens[name] = {"distance_m": get("distance_m"), "visible_s": get("visible_s"),
                      "top_speed_m_s": get("top_speed_m_s"), "runs": len(ev),
                      "run_metres": round(sum(e["metres"] for e in ev), 1),
                      "water_visits": int(get("water_visits")), "water_time_s": get("water_time_s"),
                      "pecks": int(get("pecks_ground") + get("pecks_at_feeder"))}
        runs += [{"hen": name, **e, "clock_s": start + e["start_s"]} for e in ev]
        activity += [{"hen": name, "bin": b, "metres": round(m, 2), "visible_s": round(v, 1)}
                     for b, (m, v) in sorted(_activity(d[~part], H, start).items())]
    out = {"stamp": stamp, "start_s": start, "length_s": length, "hens": hens, "runs": runs, "activity": activity}
    cache.write_text(json.dumps(out))
    return out


def day_metrics(root: Path, runs: list[Path], day: str) -> dict:
    """Everything the Metrics page shows for one day, from that day's named clips."""
    root = Path(root)
    clips, tot, runs_all, act = [], {}, [], {}
    for run in runs:
        rel = str(run.relative_to(root))
        if not rel.startswith(day):
            continue
        m = clip_metrics(run)
        if m is None:
            continue
        clips.append({"run": rel, "start_s": m["start_s"], "end_s": m["start_s"] + m["length_s"],
                      "pecks": (run / "pecks.csv").exists()})
        for h, v in m["hens"].items():
            t = tot.setdefault(h, {"distance_m": 0.0, "visible_s": 0.0, "top_speed_m_s": 0.0, "runs": 0,
                                   "run_metres": 0.0, "water_visits": 0, "water_time_s": 0.0, "pecks": 0})
            for key in ("distance_m", "visible_s", "runs", "run_metres", "water_visits", "water_time_s", "pecks"):
                t[key] += v[key]
            t["top_speed_m_s"] = max(t["top_speed_m_s"], v["top_speed_m_s"])
        runs_all += [{**r, "run": rel} for r in m["runs"]]
        for a in m["activity"]:
            r = act.setdefault((a["hen"], a["bin"]), [0.0, 0.0])
            r[0] += a["metres"]
            r[1] += a["visible_s"]
    for t in tot.values():
        t["m_per_min"] = t["distance_m"] / (t["visible_s"] / 60) if t["visible_s"] else 0.0
        for k, v in t.items():
            t[k] = round(v, 2) if isinstance(v, float) else v
    # bins covered by some clip (so a gap between clips is "no video", not "didn't move")
    cover: dict[int, float] = {}  # seconds of video in each 10-min block
    for c in clips:
        for b in range(int(c["start_s"] // BIN_S * BIN_S), int(c["end_s"]), BIN_S):
            cover[b] = cover.get(b, 0.0) + max(0.0, min(c["end_s"], b + BIN_S) - max(c["start_s"], b))
    # a block only partly filmed is scaled up to 10 min; one with under 2 min of video is left out
    covered = sorted(b for b, s in cover.items() if s >= 120)
    hens = [h for h in HENS if h in tot] + sorted(h for h in tot if h not in HENS)
    return {"day": day, "hens": hens, "totals": tot, "clips": sorted(clips, key=lambda c: c["start_s"]),
            "bin_s": BIN_S, "bins": covered,
            "activity": {h: [round(act.get((h, b), [0, 0])[0] * BIN_S / min(BIN_S, cover[b]), 1) for b in covered]
                         for h in hens},
            "runs": sorted(runs_all, key=lambda r: r["clock_s"]),
            "run_threshold_m_s": RUN_M_S}
