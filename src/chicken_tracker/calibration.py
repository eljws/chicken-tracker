"""Ground-plane calibration: image pixels -> metres on the ground.

The camera looks down at an angle, so one "pixels per metre" figure is wrong almost
everywhere. Instead we store a homography (3x3) that maps image points on the ground
(a chicken's feet) to a flat ground plan in metres.

For camera 04 it was fitted to the patio's 600 mm slabs (staggered rows, half-slab offset):
red slabs gave a first guess, then ``refine_to_joints`` moved the grid until the predicted
joints sat on the dark joint lines across the whole patio. Cross-checks: a 2 L water
bottle measures 0.33 m, chicken footprints 0.26-0.39 m.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def save_calibration(path: Path, image_to_ground: np.ndarray, **info) -> Path:
    path = Path(path)
    path.write_text(json.dumps({"image_to_ground": np.asarray(image_to_ground, float).tolist(),
                                "units": "m", **info}, indent=1))
    return path


def load_calibration(run_dir: Path) -> np.ndarray | None:
    p = Path(run_dir) / "calibration.json"
    return np.array(json.loads(p.read_text())["image_to_ground"], float) if p.exists() else None


def to_ground(H: np.ndarray, xs, ys) -> np.ndarray:
    pts = np.stack([np.asarray(xs, np.float32), np.asarray(ys, np.float32)], axis=1).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)


def walked_metres(det, H: np.ndarray, smooth: int = 7) -> float:
    """Distance walked in metres from feet points, smoothed per track to remove box jitter.

    `det` rows for one chicken (columns track_id, frame, cx, y2). Steps are only summed
    within a track: gaps between tracks (hidden time) don't add made-up straight lines.
    """
    if len(det) < 2:
        return 0.0
    det = det.sort_values(["track_id", "frame"])
    g = to_ground(H, det.cx.to_numpy(), det.y2.to_numpy())
    total = 0.0
    tid = det.track_id.to_numpy()
    for t in np.unique(tid):
        p = g[tid == t]
        if len(p) < 2:
            continue
        if len(p) >= smooth:  # rolling median, edges padded
            k = smooth // 2
            pad = np.pad(p, ((k, k), (0, 0)), mode="edge")
            p = np.stack([np.median(pad[i:i + smooth], axis=0) for i in range(len(p))])
        total += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return total


def refine_to_joints(image: np.ndarray, ground_to_image: np.ndarray, valid: np.ndarray,
                     slab_m: float = 0.6, stagger: float = 0.5,
                     anchors=((-1.2, -2.4), (1.8, -2.4), (1.8, 0.6), (-1.2, 0.6))) -> tuple[np.ndarray, float]:
    """Nudge a paving-grid homography until predicted joints lie on dark joint lines.

    image: BGR frame. ground_to_image: initial guess (ground metres -> pixels).
    valid: bool mask of paved, uncluttered pixels to score on. Rows of slabs run along
    ground x; every other row is shifted by `stagger` of a slab. Returns (H, score);
    score = mean darkness on the predicted joints (compare with the mask average).
    """
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), (0, 0), 1.5)
    T = slab_m
    pts = []
    for r in range(-12, 6):
        y = r * T
        pts += [(x, y) for x in np.arange(-5, 5, 0.01)]
        off = 0.0 if r % 2 == 0 else stagger * T
        for k in range(-10, 10):
            pts += [(k * T + off, yy) for yy in np.arange(y + 0.03, y + T - 0.03, 0.01)]
    pts = np.array(pts, np.float32).reshape(-1, 1, 2)
    ctrl = np.array(anchors, np.float32)

    def H_of(c):
        return cv2.getPerspectiveTransform(ctrl, c.reshape(4, 2).astype(np.float32))

    def score(c):
        p = cv2.perspectiveTransform(pts, H_of(c)).reshape(-1, 2)
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < g.shape[1]) & (yi >= 0) & (yi < g.shape[0])
        xi, yi = xi[ok], yi[ok]
        v = valid[yi, xi]
        return float(dark[yi[v], xi[v]].mean()) if v.sum() > 800 else 0.0

    c = cv2.perspectiveTransform(ctrl.reshape(-1, 1, 2), ground_to_image).reshape(-1)
    best = score(c)
    for step in (16, 8, 4, 2, 1, 0.5):  # simple pattern search on the 4 anchor pixels
        improved = True
        while improved:
            improved = False
            for i in range(8):
                for d in (step, -step):
                    t = c.copy()
                    t[i] += d
                    s = score(t)
                    if s > best + 1e-4:
                        best, c, improved = s, t, True
    return H_of(c), best


def kinematics(det, H: np.ndarray, window_s: float = 0.5, vmax: float = 5.0, smooth: int = 7) -> dict:
    """Speed and acceleration of one chicken, in m/s and m/s^2.

    `det`: her detections (track_id, frame, time_s, cx, y2). Per track, feet points are
    mapped to the ground and median-smoothed; speed is net displacement over `window_s`
    (a real movement, not frame-to-frame box wobble) and acceleration is the change of
    that speed over `window_s`. Windows touching a physically impossible jump (> vmax
    between consecutive frames) or missing frames are dropped. Returns per-moment samples
    plus the peaks; peaks should still be eyeballed - a tracker swapping between two
    nearby birds can look like a short dash.
    """
    v_all, a_all = [], []
    for tid, g in det.sort_values("frame").groupby("track_id"):
        t = g.time_s.to_numpy()
        if len(t) < 3:
            continue
        dt_frame = np.median(np.diff(t)) if len(t) > 1 else 0.064
        h = max(1, int(round(window_s / dt_frame / 2)))
        if len(t) < 4 * h + 3:
            continue
        raw = to_ground(H, g.cx.to_numpy(), g.y2.to_numpy())
        step_v = np.linalg.norm(np.diff(raw, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
        bad = np.r_[False, step_v > vmax] | np.r_[step_v > vmax, False]
        gap = np.diff(t) > 3 * dt_frame
        bad |= np.r_[False, gap] | np.r_[gap, False]
        k = smooth // 2
        pad = np.pad(raw, ((k, k), (0, 0)), mode="edge")
        p = np.stack([np.median(pad[i:i + smooth], axis=0) for i in range(len(raw))])
        v = np.linalg.norm(p[2 * h:] - p[:-2 * h], axis=1) / (t[2 * h:] - t[:-2 * h])
        tv = t[h:-h]
        badw = np.array([bad[i:i + 2 * h + 1].any() for i in range(len(v))])
        v = np.where(badw, np.nan, v)
        a = np.abs(v[2 * h:] - v[:-2 * h]) / (tv[2 * h:] - tv[:-2 * h])
        v_all += [(tt, vv, tid) for tt, vv in zip(tv, v) if not np.isnan(vv)]
        a_all += [(tt, aa, tid) for tt, aa in zip(tv[h:-h], a) if not np.isnan(aa)]
    out = {"top_speed_m_s": np.nan, "top_speed_at_s": np.nan, "top_speed_piece": None,
           "brisk_speed_m_s": np.nan, "top_accel_m_s2": np.nan, "top_accel_at_s": np.nan,
           "top_accel_piece": None, "moments": [],
           "samples": v_all}  # (time_s, speed m/s, track) every frame with a valid speed
    if v_all:
        tv, vv, tid = max(v_all, key=lambda x: x[1])
        out.update(top_speed_m_s=vv, top_speed_at_s=tv, top_speed_piece=int(tid),
                   brisk_speed_m_s=float(np.percentile([x[1] for x in v_all], 95)))
        # fastest distinct moments (>= 2 s apart) for checking by eye
        for t, v, tid in sorted(v_all, key=lambda x: -x[1]):
            if all(abs(t - m["at_s"]) >= 2 for m in out["moments"]):
                out["moments"].append({"at_s": t, "speed_m_s": v, "piece": int(tid)})
            if len(out["moments"]) == 3:
                break
    if a_all:
        ta, aa, tid = max(a_all, key=lambda x: x[1])
        out.update(top_accel_m_s2=aa, top_accel_at_s=ta, top_accel_piece=int(tid))
    return out
