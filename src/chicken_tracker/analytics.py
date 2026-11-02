"""Turn raw per-frame tracks into chicken analytics.

Everything here is pure pandas/numpy so it can be re-run on a saved
``detections.csv`` without re-processing the video.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .zones import Zones, points_in_polygon


@dataclass
class AnalyticsConfig:
    fps: float
    vid_stride: int = 1
    total_frames: int | None = None
    min_track_s: float = 1.0        # drop tracks shorter than this (false positives / fragments)
    smooth_window: int = 5          # rolling-median window (samples) to remove bbox jitter
    px_per_meter: float | None = None  # calibration; if set distances/speeds are in metres
    active_speed: float | None = None  # units/s above which a chicken counts as "active"
    zone_anchor: str = "bottom"     # "bottom" (feet, good for angled cams) or "center"
    zone_min_visit_s: float = 1.0   # ignore zone visits shorter than this
    zone_gap_s: float = 1.0         # merge visits separated by short dropouts
    timeline_bin_s: float = 10.0

    @property
    def frame_dt(self) -> float:
        return self.vid_stride / self.fps

    @property
    def unit(self) -> str:
        return "m" if self.px_per_meter else "px"

    @property
    def active_threshold(self) -> float:
        if self.active_speed is not None:
            return self.active_speed
        return 0.05 if self.px_per_meter else 15.0


def prepare(det: pd.DataFrame, cfg: AnalyticsConfig) -> pd.DataFrame:
    """Filter short tracks, smooth positions and add per-step kinematics."""
    if det.empty:
        return det.assign(x=[], y=[], ax=[], ay=[], step=[], speed=[])

    df = det.sort_values(["track_id", "frame"]).reset_index(drop=True)
    span = df.groupby("track_id")["time_s"].agg(lambda s: s.max() - s.min())
    keep = span[span + cfg.frame_dt >= cfg.min_track_s].index
    df = df[df["track_id"].isin(keep)].reset_index(drop=True)
    if df.empty:
        return df.assign(x=[], y=[], ax=[], ay=[], step=[], speed=[])

    g = df.groupby("track_id")
    w = max(1, cfg.smooth_window)
    df["x"] = g["cx"].transform(lambda s: s.rolling(w, center=True, min_periods=1).median())
    df["y"] = g["cy"].transform(lambda s: s.rolling(w, center=True, min_periods=1).median())

    if cfg.zone_anchor == "bottom":
        df["ax"], df["ay"] = df["cx"], df["y2"]
    else:
        df["ax"], df["ay"] = df["cx"], df["cy"]

    scale = 1.0 / cfg.px_per_meter if cfg.px_per_meter else 1.0
    g = df.groupby("track_id")
    dx = g["x"].diff()
    dy = g["y"].diff()
    dt = g["time_s"].diff()
    df["step"] = (np.hypot(dx, dy) * scale).fillna(0.0)
    df["speed"] = (df["step"] / dt).fillna(0.0)
    return df


def zone_visits(df: pd.DataFrame, zones: Zones, cfg: AnalyticsConfig) -> pd.DataFrame:
    """One row per (chicken, zone, visit)."""
    cols = ["track_id", "zone", "start_s", "end_s", "duration_s"]
    if df.empty or not zones:
        return pd.DataFrame(columns=cols)

    out = []
    for name, poly in zones.items():
        inside = df[points_in_polygon(df["ax"].to_numpy(), df["ay"].to_numpy(), poly)]
        for tid, rows in inside.groupby("track_id"):
            t = rows["time_s"].to_numpy()
            # new visit whenever the gap between in-zone samples exceeds zone_gap_s
            breaks = np.flatnonzero(np.diff(t) > cfg.zone_gap_s + cfg.frame_dt) + 1
            for seg in np.split(t, breaks):
                dur = seg[-1] - seg[0] + cfg.frame_dt
                if dur >= cfg.zone_min_visit_s:
                    out.append((tid, name, seg[0], seg[-1] + cfg.frame_dt, dur))
    return pd.DataFrame(out, columns=cols)


def track_summary(df: pd.DataFrame, visits: pd.DataFrame, zones: Zones,
                  cfg: AnalyticsConfig) -> pd.DataFrame:
    """One row per tracked chicken."""
    u = cfg.unit
    if df.empty:
        return pd.DataFrame(columns=["track_id"])

    thr = cfg.active_threshold
    df = df.assign(active=df["speed"] > thr, area=(df.x2 - df.x1) * (df.y2 - df.y1))
    g = df.groupby("track_id")
    s = pd.DataFrame({
        "first_seen_s": g["time_s"].min(),
        "last_seen_s": g["time_s"].max(),
        "frames_seen": g.size(),
    })
    s["duration_s"] = s["last_seen_s"] - s["first_seen_s"] + cfg.frame_dt
    s["visible_fraction"] = (s["frames_seen"] * cfg.frame_dt / s["duration_s"]).clip(upper=1.0)
    s[f"distance_{u}"] = g["step"].sum()
    s[f"mean_speed_{u}_s"] = s[f"distance_{u}"] / s["duration_s"]
    s[f"p95_speed_{u}_s"] = g["speed"].quantile(0.95)
    s["active_fraction"] = g["active"].mean()
    s["mean_conf"] = g["conf"].mean()
    s["mean_bbox_area_px"] = g["area"].mean()

    for name in zones:
        v = visits[visits["zone"] == name].groupby("track_id")
        s[f"{name}_visits"] = v.size().reindex(s.index, fill_value=0).astype(int)
        s[f"{name}_time_s"] = v["duration_s"].sum().reindex(s.index, fill_value=0.0)

    return s.reset_index()


def flock_timeline(df: pd.DataFrame, zones: Zones, cfg: AnalyticsConfig) -> pd.DataFrame:
    """Flock-level metrics aggregated into fixed time bins."""
    if cfg.total_frames:
        frames = np.arange(0, cfg.total_frames, cfg.vid_stride)
    elif not df.empty:
        frames = np.arange(0, int(df["frame"].max()) + 1, cfg.vid_stride)
    else:
        return pd.DataFrame(columns=["bin_start_s", "chickens_visible"])

    per_frame = pd.DataFrame(index=pd.Index(frames, name="frame"))
    if df.empty:
        per_frame["chickens_visible"] = 0
        per_frame["speed"] = np.nan
    else:
        g = df.groupby("frame")
        per_frame["chickens_visible"] = g.size().reindex(frames, fill_value=0)
        per_frame["speed"] = g["speed"].mean().reindex(frames)
        for name, poly in zones.items():
            inside = points_in_polygon(df["ax"].to_numpy(), df["ay"].to_numpy(), poly)
            per_frame[f"in_{name}"] = (
                df[inside].groupby("frame").size().reindex(frames, fill_value=0)
            )

    per_frame["bin"] = (frames / cfg.fps // cfg.timeline_bin_s).astype(int)
    tl = per_frame.groupby("bin").agg(
        chickens_visible=("chickens_visible", "mean"),
        chickens_visible_max=("chickens_visible", "max"),
        **{f"mean_speed_{cfg.unit}_s": ("speed", "mean")},
        **{f"in_{z}": (f"in_{z}", "mean") for z in zones if f"in_{z}" in per_frame},
    )
    tl.insert(0, "bin_start_s", tl.index * cfg.timeline_bin_s)
    return tl.reset_index(drop=True)


def flock_summary(df: pd.DataFrame, tracks: pd.DataFrame, timeline: pd.DataFrame,
                  zones: Zones, cfg: AnalyticsConfig) -> dict:
    u = cfg.unit
    out = {
        "tracks": int(len(tracks)),
        # Track count over-estimates flock size when IDs switch; peak simultaneous
        # visible chickens is the more reliable estimate of how many are present.
        "estimated_flock_size": int(timeline["chickens_visible_max"].max()) if len(timeline) else 0,
        "mean_chickens_visible": float(timeline["chickens_visible"].mean()) if len(timeline) else 0.0,
        "distance_unit": u,
        "active_speed_threshold": cfg.active_threshold,
    }
    if len(tracks):
        out[f"total_distance_{u}"] = float(tracks[f"distance_{u}"].sum())
        out["mean_active_fraction"] = float(
            np.average(tracks["active_fraction"], weights=tracks["frames_seen"])
        )
        for z in zones:
            out[f"{z}_total_time_s"] = float(tracks[f"{z}_time_s"].sum())
            out[f"{z}_total_visits"] = int(tracks[f"{z}_visits"].sum())
    return out


def render_heatmap(df: pd.DataFrame, background: np.ndarray, zones: Zones,
                   out_path: str, alpha: float = 0.55) -> None:
    """Occupancy heatmap of chicken positions blended over a video frame."""
    h, w = background.shape[:2]
    canvas = background.copy()
    if not df.empty:
        hist, _, _ = np.histogram2d(
            df["ay"].clip(0, h - 1), df["ax"].clip(0, w - 1),
            bins=(h // 4, w // 4), range=[[0, h], [0, w]],
        )
        hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), sigmaX=4)
        if hist.max() > 0:
            norm = (255 * np.sqrt(hist / hist.max())).astype(np.uint8)
            norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
            color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            mask = (norm > 8)[..., None]
            blended = cv2.addWeighted(background, 1 - alpha, color, alpha, 0)
            canvas = np.where(mask, blended, background)
    for name, poly in zones.items():
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)
        x, y = poly.min(axis=0).astype(int)
        cv2.putText(canvas, name, (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
