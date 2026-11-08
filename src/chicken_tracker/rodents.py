"""Rodents (and anything else) moving in the yard at night, from the infrared recordings.

**Finding movers** (`scan`): only keyframes are decoded (one per second in these recordings, by the Mac's
hardware), ~15-70 s per 32-min file. Each frame is compared with the yard over the last ~20 s (a median);
small changed spots that move on across >= 2 frames become an event. The printed clock and camera name
are masked. Only infrared stretches are scanned (timeline.json); files already scanned are skipped.

**Sorting** (the Rodents page): every event gets a label - rat, other animal, insect, person, light,
nothing, unsure - by you, or first by Claude from the pictures (tagged "by": "claude", shown with 🤖,
to check). Sorted events are the training material for a night rodent detector.

Everything lives in <output>/rodents/: events.json, labels.json, scanned.json, tiles/ (picture strips).
Event ids are "<recording>@<seconds>:<x>,<y>", stable across rescans.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import timeline as TL
from .player import ffmpeg_exe

DIR = "rodents"
KINDS = ["rat", "other_animal", "insect", "person", "light", "nothing", "unsure"]
W, H = 640, 512      # frames are scanned at this size (the recordings are 1920x1536: x3)
HIST = 20            # frames (~20 s) in the background median
DIFF = 18            # grey levels of change that count
MIN_A, MAX_A = 12, 4000   # spot area at 640 px wide: a rat ~50-150, a person/hen bigger
LINK_PX = 120        # a spot in the next frame this close = the same mover
MASK = [(0, 0, W, int(H * 0.11)), (int(W * 0.68), int(H * 0.86), W, H)]  # clock, camera name
_lock = threading.Lock()
_scan = {"running": False, "done": 0, "total": 0, "error": None}


def keyframes(path: Path):
    """(time s, grey frame) for every keyframe."""
    p = subprocess.Popen([ffmpeg_exe(), "-v", "info", "-hwaccel", "videotoolbox", "-skip_frame", "nokey", "-i", str(path),
                          "-fps_mode", "passthrough", "-vf", f"scale={W}:{H},format=gray,showinfo", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import threading
    times = []

    def read_err():
        for line in p.stderr:
            m = re.search(rb"pts_time:([\d.]+)", line)
            if m:
                times.append(float(m[1]))
    th = threading.Thread(target=read_err, daemon=True)
    th.start()
    i = 0
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        while len(times) <= i and th.is_alive():
            time.sleep(0.001)
        yield (times[i] if i < len(times) else i * 1.0), np.frombuffer(b, np.uint8).reshape(H, W)
        i += 1
    p.wait()


def spots(frame, bg):
    d = cv2.absdiff(cv2.GaussianBlur(frame, (5, 5), 0), bg)
    m = (d > DIFF).astype(np.uint8)
    for x1, y1, x2, y2 in MASK:
        m[y1:y2, x1:x2] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if MIN_A <= a <= MAX_A:
            out.append({"box": [int(x), int(y), int(x + w), int(y + h)], "area": int(a), "c": [float(cen[k][0]), float(cen[k][1])],
                        "diff": float(d[lab == k].mean())})
    if len(out) > 25:  # the whole picture changed (light switching, rain, camera adjusting): not a mover
        return []
    return out


def scan_file(clip: dict, night: list[tuple[float, float]]) -> list[dict]:
    hist, tracks, done, bg, k = [], [], [], None, 0
    for t, f in keyframes(Path(clip["path"])):
        g = cv2.GaussianBlur(f, (5, 5), 0)
        inside = any(a <= t <= b for a, b in night)
        k += 1
        if len(hist) >= 5 and inside:
            if bg is None or k % 4 == 0:  # the yard's look, refreshed every 4 s (a median is slow)
                bg = np.median(np.stack(hist[-HIST:]), axis=0).astype(np.uint8)
            sp = spots(f, bg)
            nxt = []
            for s in sp:
                best = min(tracks, key=lambda tr: np.hypot(*(np.array(tr["pts"][-1]["c"]) - s["c"])), default=None)
                if best is not None and best not in nxt and np.hypot(*(np.array(best["pts"][-1]["c"]) - s["c"])) < LINK_PX:
                    best["pts"].append({"t": t, **s})
                    nxt.append(best)
                else:
                    nxt.append({"pts": [{"t": t, **s}]})
            done += [tr for tr in tracks if tr not in nxt]
            tracks = nxt
        hist.append(g)
        if len(hist) > HIST:
            hist.pop(0)
    done += tracks
    ev = []
    for tr in done:
        p = tr["pts"]
        if len(p) < 2:
            continue
        moved = np.hypot(*(np.array(p[-1]["c"]) - p[0]["c"]))
        if moved < 6:  # stayed put: flicker on a leaf, a reflection
            continue
        ev.append({"id": event_id(clip["rel"], p[0]), "file": clip["rel"], "start": clip["start"], "t0": p[0]["t"], "t1": p[-1]["t"], "n": len(p), "moved_px": round(float(moved), 1),
                   "area": int(np.median([q["area"] for q in p])), "pts": [{"t": q["t"], "box": q["box"]} for q in p]})
    return ev




def event_id(rel: str, first: dict) -> str:
    """Stable id: recording, first moment, and where the mover was then (two can start together)."""
    b = first["box"]
    return f"{Path(rel).stem}@{first['t']:.1f}:{(b[0] + b[2]) // 2},{(b[1] + b[3]) // 2}"


# ------------------------------------------------------------------ files
def folder(root: Path) -> Path:
    d = Path(root) / DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _write(p: Path, data) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(p)


def events(root: Path) -> list[dict]:
    return _read(folder(root) / "events.json", [])


def labels(root: Path) -> dict:
    return _read(folder(root) / "labels.json", {})


def when(e: dict) -> dt.datetime:
    return dt.datetime.fromisoformat(e["start"]) + dt.timedelta(seconds=e["t0"])


def set_label(root: Path, eid: str, label: str | None, by: str = "you") -> dict:
    if label is not None and label not in KINDS:
        raise ValueError(f"unknown label {label}")
    with _lock:
        lab = labels(root)
        if label is None:
            lab.pop(eid, None)
        else:
            lab[eid] = {"label": label, "by": by, "at": time.time()}
        _write(folder(root) / "labels.json", lab)
    return lab


# ------------------------------------------------------------------ scanning
def scan(root: Path, progress=None) -> list[dict]:
    """Scan every infrared stretch not scanned yet; adds to events.json."""
    root = Path(root)
    tl = TL.load(root)
    if not tl:
        raise ValueError("no timeline yet: run chicken-tracker timeline")
    d = folder(root)
    done = set(_read(d / "scanned.json", []))
    todo = [c for c in tl["clips"] if c["rel"] not in done and any(p == "night" for *_, p in TL.clip_phases(tl, c["name"]))]
    ev = events(root)
    for i, c in enumerate(todo):
        night = [(a, b) for a, b, p in TL.clip_phases(tl, c["name"]) if p == "night"]
        ev = [e for e in ev if e["file"] != c["rel"]] + scan_file(c, night)
        done.add(c["rel"])
        _write(d / "events.json", ev)
        _write(d / "scanned.json", sorted(done))
        if progress:
            progress(i + 1, len(todo), c["name"])
    return ev


def scan_in_background(root: Path) -> dict:
    with _lock:
        if _scan["running"]:
            return dict(_scan)
        _scan.update(running=True, done=0, total=0, error=None)

    def run():
        try:
            scan(root, progress=lambda i, n, name: _scan.update(done=i, total=n, current=name))
        except Exception as e:  # shown on the page
            _scan["error"] = str(e)
        finally:
            _scan["running"] = False
    threading.Thread(target=run, daemon=True).start()
    return dict(_scan)


def scan_status() -> dict:
    return dict(_scan)


# ------------------------------------------------------------------ pictures and video of an event
def _path(root: Path, e: dict) -> str | None:
    tl = TL.load(root) or {"clips": []}
    p = TL.locate(next((c["path"] for c in tl["clips"] if c["rel"] == e["file"]), None), root)
    return str(p) if p else None


def _safe(eid: str) -> str:
    return re.sub(r"[^\w@.\-]", "_", eid)


def tile(root: Path, e: dict) -> bytes | None:
    """4 moments of the event: a square around the mover at full resolution, enlarged (cached jpg)."""
    f = folder(root) / "tiles" / (_safe(e["id"]) + ".jpg")
    if f.exists():
        return f.read_bytes()
    p = _path(root, e)
    if not p:
        return None
    cap = cv2.VideoCapture(p)
    pts = [e["pts"][i] for i in np.linspace(0, len(e["pts"]) - 1, 4).round().astype(int)]
    tiles = []
    for q in pts:
        cap.set(cv2.CAP_PROP_POS_MSEC, q["t"] * 1000)
        ok, img = cap.read()
        if not ok:
            tiles.append(np.zeros((180, 180, 3), np.uint8))
            continue
        x1, y1, x2, y2 = [v * 3 for v in q["box"]]
        r = max(60, int(max(x2 - x1, y2 - y1) * 0.8))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)
        tiles.append(cv2.resize(pad[cy:cy + 2 * r, cx:cx + 2 * r], (180, 180), interpolation=cv2.INTER_CUBIC))
    cap.release()
    ok, b = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(b.tobytes())
    return b.tobytes()


def clip(root: Path, e: dict) -> bytes | None:
    """A short video of the event, cut out around where the mover goes (2 s before to 2 s after)."""
    p = _path(root, e)
    if not p:
        return None
    bx = np.array([q["box"] for q in e["pts"]]) * 3
    x1, y1, x2, y2 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    m = 160
    x1, y1 = max(0, x1 - m), max(0, y1 - m)
    w, h = min(1920 - x1, x2 - x1 + 2 * m), min(1536 - y1, y2 - y1 + 2 * m)
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    t0, dur = max(0.0, e["t0"] - 2), e["t1"] - e["t0"] + 4
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        r = subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{t0:.2f}", "-i", p, "-t", f"{dur:.2f}", "-an",
                            "-vf", f"crop={w}:{h}:{int(x1)}:{int(y1)},scale='min(720,iw*2)':-2", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp.name],
                           capture_output=True)
        return Path(tmp.name).read_bytes() if r.returncode == 0 else None


_warm = {"running": False}


def warm_tiles(root: Path) -> None:
    """Make the missing picture strips in the background (sorted events first), so the page is quick."""
    if _warm["running"]:
        return
    _warm["running"] = True

    def run():
        try:
            lab = labels(root)
            for e in sorted(events(root), key=lambda e: e["id"] not in lab):
                if not (folder(root) / "tiles" / (_safe(e["id"]) + ".jpg")).exists():
                    try:
                        tile(root, e)
                    except (OSError, cv2.error):
                        pass
        finally:
            _warm["running"] = False
    threading.Thread(target=run, daemon=True).start()


def page_data(root: Path) -> dict:
    lab = labels(root)
    tl = TL.load(root)
    out = []
    for e in events(root):
        w = when(e)
        out.append({"id": e["id"], "at": w.isoformat(timespec="seconds"), "secs": e["n"], "area": e["area"],
                    "day": TL.day_of(tl, w) if tl else None,
                    "file": e["file"], "t0": e["t0"], **({"label": lab[e["id"]]["label"], "by": lab[e["id"]].get("by")} if e["id"] in lab else {})})
    out.sort(key=lambda x: x["at"])
    return {"events": out, "kinds": KINDS, "scan": scan_status()}
