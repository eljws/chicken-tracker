"""Render a watchable H.264 preview of a run: boxes, names, trails, hidden markers."""

from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HIDDEN_TAU_S = 20.0  # same decay as identity.export


def _color(key: str) -> tuple[int, int, int]:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _text(img, s, org, color, scale=0.6, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for i in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            e = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, s, e, color, 2, cv2.LINE_AA)


def parse_time(s: str | float) -> float:
    """'17:30' or '1050' -> seconds into the video."""
    if isinstance(s, (int, float)):
        return float(s)
    parts = [float(p) for p in str(s).split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def render(run_dir: Path, start_s: float, duration_s: float, out: Path | None = None,
           video: str | None = None, width: int = 1280, trail_s: float = 4.0) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    video = video or meta["video"]["path"]
    fps = meta["video"]["fps"]
    named_path = run_dir / "detections_named.csv"
    named = named_path.exists()
    det = pd.read_csv(named_path if named else run_dir / "detections.csv")
    det["label"] = det["chicken"] if named else "#" + det.track_id.astype(str)
    out = Path(out or run_dir / ("preview_named.mp4" if named else "preview.mp4"))

    # hidden segments: where/when each named chicken was last seen before vanishing
    hidden = []
    tl_path = run_dir / "chicken_timeline.csv"
    if named and tl_path.exists():
        tl = pd.read_csv(tl_path)
        for _, h in tl[tl.state == "hidden"].iterrows():
            last = det[det.track_id == h.track_before].sort_values("frame").tail(1)
            if len(last):
                hidden.append((h.chicken, h.start_s, h.end_s, last.iloc[0]))
    names = sorted(det.label.unique()) if named else []

    start = int(start_s * fps)
    end = start + int(duration_s * fps)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {video} - pass --video if it moved.")
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = width / W0
    W, H = width, int(round(H0 * s / 2) * 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    window = det[(det.frame >= start - int(trail_s * fps)) & (det.frame < end)]
    by_frame = {f: g for f, g in window.groupby("frame")}
    trails: dict[str, list] = {}
    for f in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (W, H))
        t = f / fps
        visible = set()
        for _, r in (by_frame[f].iterrows() if f in by_frame else []):
            lab = str(r.label)
            c = _color(lab)
            visible.add(lab)
            trails.setdefault(lab, []).append((f, int(r.cx * s), int(r.y2 * s)))
            cv2.rectangle(img, (int(r.x1 * s), int(r.y1 * s)), (int(r.x2 * s), int(r.y2 * s)), c, 2)
            _text(img, lab, (int(r.x1 * s), int(r.y1 * s) - 7), c)
        for lab, pts in trails.items():
            pts[:] = [p for p in pts if f - p[0] < trail_s * fps]
            if len(pts) > 1:
                cv2.polylines(img, [np.array([(x, y) for _, x, y in pts], np.int32)], False,
                              _color(lab), 2, cv2.LINE_AA)
        status = {}
        for name, h0, h1, last in hidden:
            if h0 <= t <= h1 and name not in visible:
                conf = float(np.exp(-(t - h0) / HIDDEN_TAU_S))
                c = tuple(int(v * (0.4 + 0.6 * conf)) for v in _color(name))
                _dashed_rect(img, (int(last.x1 * s), int(last.y1 * s)), (int(last.x2 * s), int(last.y2 * s)), c)
                _text(img, f"{name}? {conf:.0%}", (int(last.x1 * s), int(last.y1 * s) - 7), c, 0.55)
                status[name] = f"hidden {conf:.0%}"
        # legend
        if names:
            # below the camera's own timestamp overlay, on a dark panel
            y0 = int(H * 0.11)
            panel = img[y0:y0 + 26 * len(names) + 12, 6:250]
            panel[:] = (panel * 0.45).astype(np.uint8)
            y = y0 + 26
            for n in names:
                st = "visible" if n in visible else status.get(n, "away")
                _text(img, f"{n}: {st}", (14, y), _color(n) if st != "away" else (150, 150, 150), 0.6)
                y += 26
        clock = int(t)
        _text(img, f"{clock // 60:02d}:{clock % 60:02d} into video", (W - 230, H - 20), (255, 255, 255), 0.6)
        writer.write(img)
    writer.release()
    cap.release()
    return out
