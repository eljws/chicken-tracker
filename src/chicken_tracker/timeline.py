"""All recordings as one timeline, with dawn and dusk markers found from the camera's own switch.

The camera records ~32 min files (downloads/<device>/<date>/HHMMSS-HHMMSS_ch04.mp4). Here they are
lined up into one continuous timeline (no copying: the web page plays file after file), and the
camera's switches between colour and infrared are found automatically:

    dawn  = infrared -> colour      dusk = colour -> infrared

A "day" runs from one dawn to the next, named by the date of its dawn. Everything is saved in
<output>/timeline.json so processing can ask what a moment is:

    tl = timeline.load(root)
    timeline.phase_at(tl, when)    -> "day" | "night" | None (no recording)
    timeline.day_of(tl, when)      -> "2026-11-02" (the dawn-to-dawn day it belongs to)
    timeline.clip_phases(tl, name) -> [(start s, end s, "day"/"night"), ...] inside one clip

How the switch is found: infrared pictures have no colour, so the yard's mean colour saturation drops
to ~0 (daylight, even dull or glary, stays above ~15). Each file is sampled every 30 s; where the
phase flips, the moment is narrowed down to about a second. Flips shorter than MIN_PHASE_S
(a car's headlights, a glitch) are ignored. Measurements are cached per file.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

FILE = "timeline.json"
CACHE_DIR = "timeline_light"
SAMPLE_S = 30.0       # one colour sample every 30 s
IR_SAT = 6.0          # below = infrared (see daylight.py)
MIN_PHASE_S = 600.0   # a day or night stretch shorter than 10 min is a glitch, not a switch
REFINE_S = 1.0        # switch moments narrowed down to about a second
METHOD = 1
NAME = re.compile(r"(\d{6})-(\d{6})_ch(\d+)\.mp4$")

_lock = threading.Lock()
_scan = {"running": False, "done": 0, "total": 0, "error": None}


# ------------------------------------------------------------------ finding the recordings
def download_roots(root: Path) -> list[Path]:
    """Device folders the processed clips came from (…/downloads/<device>/), from their meta.json."""
    roots = set()
    for m in Path(root).glob("*/*/meta.json"):
        try:
            p = Path(json.loads(m.read_text())["video"]["path"])
        except (OSError, ValueError, KeyError):
            continue
        if p.parent.parent.exists():
            roots.add(p.parent.parent)
    return sorted(roots)


def _clip_times(path: Path, duration_s: float | None) -> tuple[dt.datetime, dt.datetime] | None:
    m = NAME.search(path.name)
    try:
        day = dt.date.fromisoformat(path.parent.name)
    except ValueError:
        return None
    if not m:
        return None
    s = dt.datetime.combine(day, dt.datetime.strptime(m[1], "%H%M%S").time())
    if duration_s:
        e = s + dt.timedelta(seconds=duration_s)
    else:
        e = dt.datetime.combine(day, dt.datetime.strptime(m[2], "%H%M%S").time())
        if e <= s:
            e += dt.timedelta(days=1)  # runs past midnight
    return s, e


def find_clips(roots: list[Path], root: Path) -> list[dict]:
    """Every recording on disk, in time order, with its processed run (if any)."""
    runs = {}
    for m in Path(root).glob("*/*/meta.json"):
        try:
            runs[Path(json.loads(m.read_text())["video"]["path"]).name] = str(m.parent.relative_to(root))
        except (OSError, ValueError, KeyError):
            pass
    out = []
    for r in roots:
        durs = {}
        man = r / "manifest.jsonl"
        if man.exists():
            for line in man.read_text().splitlines():
                try:
                    j = json.loads(line)
                    if j.get("video_file") and j.get("duration_s"):
                        durs[j["video_file"]] = float(j["duration_s"])
                except ValueError:
                    pass
        for p in sorted(r.glob("*/*_ch*.mp4")):
            rel = f"{p.parent.name}/{p.name}"
            dur = durs.get(rel)
            if dur is None:
                dur = _probe_duration(p)
            t = _clip_times(p, dur)
            if t is None:
                continue
            out.append({"name": p.name, "path": str(p), "rel": rel, "start": t[0].isoformat(), "end": t[1].isoformat(),
                        "duration_s": round((t[1] - t[0]).total_seconds(), 2), "run": runs.get(p.name),
                        "size": p.stat().st_size})
    out.sort(key=lambda c: c["start"])
    return out


def _probe_duration(p: Path) -> float | None:
    import cv2
    c = cv2.VideoCapture(str(p))
    fps, n = c.get(cv2.CAP_PROP_FPS), c.get(cv2.CAP_PROP_FRAME_COUNT)
    c.release()
    return n / fps if fps and n else None


# ------------------------------------------------------------------ colour samples per file
def _sat(cap, t_s: float) -> float | None:
    import cv2
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, img = cap.read()
    if not ok:
        return None
    small = cv2.resize(img[int(img.shape[0] * 0.15):], (96, 64), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean())


def light_samples(clip: dict, cache_dir: Path) -> dict:
    """{"t": [s...], "sat": [...], "switches": [{"t": s, "to": "day"/"night"}]} for one file, cached."""
    import cv2
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / (clip["name"] + ".json")
    key = [clip["size"], METHOD, SAMPLE_S, IR_SAT]
    if cf.exists():
        try:
            d = json.loads(cf.read_text())
            if d.get("key") == key:
                return d
        except ValueError:
            pass
    cap = cv2.VideoCapture(clip["path"])
    dur = clip["duration_s"]
    ts = list(np.arange(0, max(dur - 1, 0.5), SAMPLE_S)) + [max(0.0, dur - 2)]
    t_ok, sat = [], []
    for t in ts:
        s = _sat(cap, t)
        if s is not None:
            t_ok.append(round(float(t), 2))
            sat.append(round(s, 2))
    # narrow every flip inside the file down to ~1 s (between two samples, by halving)
    switches = []
    for i in range(1, len(sat)):
        a, b = sat[i - 1] < IR_SAT, sat[i] < IR_SAT
        if a == b:
            continue
        lo, hi = t_ok[i - 1], t_ok[i]
        while hi - lo > REFINE_S:
            mid = (lo + hi) / 2
            s = _sat(cap, mid)
            if s is None:
                break
            (lo, hi) = (mid, hi) if (s < IR_SAT) == a else (lo, mid)
        switches.append({"t": round(hi, 1), "to": "night" if b else "day"})
    cap.release()
    d = {"key": key, "t": t_ok, "sat": sat, "switches": switches, "first": "night" if sat and sat[0] < IR_SAT else "day"}
    cf.write_text(json.dumps(d))
    return d


# ------------------------------------------------------------------ markers and days
def _phases(clips: list[dict], lights: list[dict]) -> list[list]:
    """Continuous [start, end, phase] stretches over all recordings (gaps between files bridged
    when under 2 min), with stretches under MIN_PHASE_S folded into their neighbours."""
    segs = []
    for c, L in zip(clips, lights):
        s0 = dt.datetime.fromisoformat(c["start"])
        cur, at = L["first"], 0.0
        for sw in L["switches"]:
            segs.append([s0 + dt.timedelta(seconds=at), s0 + dt.timedelta(seconds=sw["t"]), cur])
            cur, at = sw["to"], sw["t"]
        segs.append([s0 + dt.timedelta(seconds=at), dt.datetime.fromisoformat(c["end"]), cur])
    # join neighbours with the same phase (across file boundaries and short gaps)
    out = []
    for s in segs:
        if out and out[-1][2] == s[2] and (s[0] - out[-1][1]).total_seconds() < 120:
            out[-1][1] = max(out[-1][1], s[1])
        else:
            out.append(list(s))
    # fold glitches: a short stretch between two of the other phase joins them
    changed = True
    while changed:
        changed = False
        for i in range(1, len(out) - 1):
            if (out[i][1] - out[i][0]).total_seconds() < MIN_PHASE_S and out[i - 1][2] == out[i + 1][2] != out[i][2]:
                out[i - 1][1] = out[i + 1][1]
                del out[i:i + 2]
                changed = True
                break
    return out


def build(root: Path, roots: list[Path] | None = None, progress=None) -> dict:
    root = Path(root)
    roots = roots or download_roots(root)
    clips = find_clips(roots, root)
    lights = []
    for i, c in enumerate(clips):
        lights.append(light_samples(c, root / CACHE_DIR))
        if progress:
            progress(i + 1, len(clips), c["name"])
    phases = _phases(clips, lights)
    markers = []
    for a, b in zip(phases, phases[1:]):
        if (b[0] - a[1]).total_seconds() < 120 and a[2] != b[2]:
            markers.append({"kind": "dawn" if b[2] == "day" else "dusk", "at": b[0].isoformat(timespec="seconds")})
    # days: dawn to next dawn; before the first dawn a partial night belongs to the day before
    dawns = [dt.datetime.fromisoformat(m["at"]) for m in markers if m["kind"] == "dawn"]
    days = []
    if clips:
        first, last = dt.datetime.fromisoformat(clips[0]["start"]), dt.datetime.fromisoformat(clips[-1]["end"])
        edges = [first] + [d for d in dawns if first < d < last] + [last]
        for a, b in zip(edges, edges[1:]):
            starts_at_dawn = a in dawns
            label = (a.date() if starts_at_dawn else (a.date() - dt.timedelta(days=1) if a.hour < 12 else a.date())).isoformat()
            dusk = next((m["at"] for m in markers if m["kind"] == "dusk" and a.isoformat() <= m["at"] < b.isoformat()), None)
            days.append({"day": label, "from": a.isoformat(timespec="seconds"), "to": b.isoformat(timespec="seconds"),
                         "dawn": a.isoformat(timespec="seconds") if starts_at_dawn else None, "dusk": dusk,
                         "complete": starts_at_dawn and b in dawns})
    for c, L in zip(clips, lights):
        c["first"] = L["first"]
        c["switches"] = L["switches"]
    tl = {"built": time.time(), "method": METHOD, "roots": [str(r) for r in roots], "clips": clips,
          "phases": [[a.isoformat(timespec="seconds"), b.isoformat(timespec="seconds"), p] for a, b, p in phases],
          "markers": markers, "days": days, "threshold": IR_SAT, "min_phase_s": MIN_PHASE_S}
    tmp = root / (FILE + ".tmp")
    tmp.write_text(json.dumps(tl, indent=1))
    tmp.replace(root / FILE)
    return tl


def scan_in_background(root: Path) -> dict:
    """Start (re)building the timeline in a thread; returns the scan status."""
    with _lock:
        if _scan["running"]:
            return dict(_scan)
        _scan.update(running=True, done=0, total=0, error=None, current=None)

    def run():
        try:
            build(root, progress=lambda i, n, name: _scan.update(done=i, total=n, current=name))
        except Exception as e:  # shown on the page
            _scan["error"] = str(e)
        finally:
            _scan["running"] = False
    threading.Thread(target=run, daemon=True).start()
    return dict(_scan)


def scan_status() -> dict:
    return dict(_scan)


# ------------------------------------------------------------------ asking the timeline
def load(root: Path) -> dict | None:
    p = Path(root) / FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def _when(when) -> dt.datetime:
    return when if isinstance(when, dt.datetime) else dt.datetime.fromisoformat(str(when))


def phase_at(tl: dict, when) -> str | None:
    w = _when(when).isoformat(timespec="seconds")
    for a, b, p in tl["phases"]:
        if a <= w < b:
            return p
    return None


def day_of(tl: dict, when) -> str | None:
    w = _when(when).isoformat(timespec="seconds")
    for d in tl["days"]:
        if d["from"] <= w < d["to"]:
            return d["day"]
    return None


def clip_phases(tl: dict, name: str) -> list[tuple[float, float, str]]:
    """[(start s, end s, phase)] inside one recording, by its file name."""
    c = next((c for c in tl["clips"] if c["name"] == name), None)
    if c is None:
        return []
    out, cur, at = [], c["first"], 0.0
    for sw in c["switches"]:
        out.append((at, sw["t"], cur))
        cur, at = sw["to"], sw["t"]
    out.append((at, c["duration_s"], cur))
    return out
