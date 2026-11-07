"""Which parts of a clip are infrared (black-and-white): out of scope, daytime colour video only.

The camera switches to infrared in the late afternoon. Infrared pictures have no colour at all, so the
mean colour saturation of the yard drops from ~20-60 to ~0. One frame every few seconds is sampled
(the timestamp strip at the top is left out) and runs below the threshold become infrared stretches.

Saved as <run>/daylight.json and redone when the video changes. Used to keep infrared stretches out of
checks, questions, the size calibration and recogniser training; the pieces themselves stay as they are.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILE = "daylight.json"
STEP_S = 5.0     # one sample every 5 s
IR_SAT = 6.0     # mean saturation below this = no colour = infrared (daytime, even dull or glary, is 15+)
METHOD = 2       # bumped when measuring changes: older daylight.json files are redone
MIN_IR_S = 20.0  # shorter dips (a dark frame, a glitch) don't count


def _video(run: Path) -> tuple[Path | None, float]:
    """(video to measure, factor from its seconds to clip seconds). The browser copy is quickest (a keyframe
    every 10 s), but only when its frame rate matches the clip's; otherwise the original recording."""
    import cv2
    try:
        meta = json.loads((run / "meta.json").read_text())["video"]
    except (OSError, ValueError, KeyError):
        meta = {}
    fps = meta.get("fps")
    orig = Path(meta["path"]) if meta.get("path") else None
    vids = sorted(run.glob("browser/video_*.mp4"))
    if vids:
        c = cv2.VideoCapture(str(vids[0]))
        vf = c.get(cv2.CAP_PROP_FPS)
        c.release()
        if not fps or abs(vf - fps) < 0.05 * fps:
            return vids[0], 1.0
    if orig is not None and orig.exists():
        return orig, 1.0
    if vids:  # copy with the wrong frame rate written in: its frames are still the clip's frames
        return vids[0], (vf / fps) if fps and vf else 1.0
    return None, 1.0


def measure(video: Path, step_s: float = STEP_S) -> tuple[list[float], list[float]]:
    """(times s, mean saturation) at the video's keyframes (every ~10 s in the browser copies): only
    keyframes are decoded, so a 30 min clip takes a few seconds. Falls back to seeking every step_s."""
    import re
    import subprocess

    import cv2
    from .player import ffmpeg_exe
    w, h = 64, 48
    try:
        r = subprocess.run([ffmpeg_exe(), "-v", "info", "-skip_frame", "nokey", "-i", str(video), "-an",
                            "-vf", f"scale={w}:{h},showinfo", "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
                            "-f", "rawvideo", "-"], capture_output=True, timeout=600)
        times = [float(m) for m in re.findall(rb"pts_time:([\d.]+)", r.stderr)]
        imgs = np.frombuffer(r.stdout, np.uint8)
        k = len(imgs) // (w * h * 3)
        if r.returncode == 0 and k and k == len(times) and (k > 3 or times[-1] < 60):
            imgs = imgs[: k * w * h * 3].reshape(k, h, w, 3)
            sat = [round(float(cv2.cvtColor(im[int(h * 0.15):], cv2.COLOR_BGR2HSV)[..., 1].mean()), 2) for im in imgs]
            return [round(t, 2) for t in times], sat
    except (OSError, subprocess.SubprocessError):
        pass
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts, sat = [], []
    for f in range(0, n, max(1, int(round(fps * step_s)))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(img[int(img.shape[0] * 0.15):], cv2.COLOR_BGR2HSV)
        ts.append(round(f / fps, 2))
        sat.append(round(float(hsv[..., 1].mean()), 2))
    cap.release()
    return ts, sat


def stretches(ts: list[float], sat: list[float], step_s: float | None = None) -> list[list[float]]:
    """Infrared stretches [start s, end s] from the samples (edges halfway to the neighbouring sample)."""
    if step_s is None:
        step_s = float(np.median(np.diff(ts))) if len(ts) > 1 else STEP_S
    out, start = [], None
    for t, s in zip(ts + [None], sat + [None]):
        ir = s is not None and s < IR_SAT
        if ir and start is None:
            start = t
        elif not ir and start is not None:
            end = (t if t is not None else ts[-1] + step_s)
            if end - start >= MIN_IR_S:
                out.append([round(max(0.0, start - step_s / 2), 1), round(end - step_s / 2, 1)])
            start = None
    return out


def load(run: Path, compute: bool = True) -> dict | None:
    """{"infrared": [[start s, end s], ...], ...} for the clip; measured once, then read from daylight.json."""
    run = Path(run)
    v, k = _video(run)
    p = run / FILE
    key = round(v.stat().st_mtime, 3) if v else None
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if d.get("method") == METHOD and (d.get("video_mtime") == key or v is None):
                return d
        except ValueError:
            pass
    if not compute or v is None:
        return None
    ts, sat = measure(v)
    ts = [round(t * k, 2) for t in ts]
    d = {"method": METHOD, "infrared": stretches(ts, sat), "video": v.name, "video_mtime": key,
         "threshold": IR_SAT, "samples": len(ts)}
    p.write_text(json.dumps(d))
    return d


def infrared_s(run: Path, compute: bool = True) -> list[list[float]]:
    """Infrared stretches [start s, end s] of a processed clip: from the timeline's markers when the
    timeline knows its recording (output/timeline.json, see timeline.py), else measured for this clip."""
    from . import timeline as TL
    run = Path(run)
    try:
        name = Path(json.loads((run / "meta.json").read_text())["video"]["path"]).name
    except (OSError, ValueError, KeyError):
        name = None
    for d in (run, *run.parents):
        if (d / TL.FILE).exists():
            tl = TL.load(d)
            ph = TL.clip_phases(tl, name) if tl and name else []
            if ph:
                return [[round(a, 1), round(b, 1)] for a, b, p in ph if p == "night"]
            break
        if (d / "pyproject.toml").exists():
            break
    d = load(run, compute)
    return d["infrared"] if d else []


def in_infrared(frames, fps: float, ir: list[list[float]]) -> np.ndarray:
    """Bool per frame: inside an infrared stretch."""
    t = np.asarray(frames, float) / fps
    out = np.zeros(len(t), bool)
    for a, b in ir:
        out |= (t >= a) & (t <= b)
    return out


def infrared_tracks(data: dict, ir: list[list[float]], share: float = 0.5) -> set[int]:
    """Pieces that are mostly (> share of their time) in infrared."""
    if not ir:
        return set()
    out = set()
    for t, f in data["fragments"].items():
        a, b = f["start"], f["end"]
        if b <= a:
            inside = any(x <= a <= y for x, y in ir)
        else:
            inside = sum(max(0.0, min(b, y) - max(a, x)) for x, y in ir) / (b - a) > share
        if inside:
            out.add(int(t))
    return out
