"""Glue: video -> detections -> analytics files on disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import analytics as A
from .tracking import TrackConfig, VideoInfo, read_frame, track_video
from .zones import Zones


def write_analytics(det: pd.DataFrame, info: VideoInfo, zones: Zones,
                    cfg: A.AnalyticsConfig, out_dir: Path) -> dict:
    df = A.prepare(det, cfg)
    visits = A.zone_visits(df, zones, cfg)
    tracks = A.track_summary(df, visits, zones, cfg)
    timeline = A.flock_timeline(df, zones, cfg)
    summary = A.flock_summary(df, tracks, timeline, zones, cfg)
    summary = {"video": asdict(info), "duration_s": info.duration_s, **summary}

    tracks.to_csv(out_dir / "tracks.csv", index=False, float_format="%.3f")
    timeline.to_csv(out_dir / "timeline.csv", index=False, float_format="%.3f")
    if zones:
        visits.to_csv(out_dir / "zone_visits.csv", index=False, float_format="%.3f")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        bg = read_frame(info.path, 0)
        A.render_heatmap(df, bg, zones, str(out_dir / "heatmap.png"))
    except (ValueError, FileNotFoundError):
        pass  # source video moved; skip the heatmap
    return summary


def process_video(video: Path, out_root: Path, track_cfg: TrackConfig, zones: Zones,
                  analytics_kwargs: dict, save_video: bool = False) -> dict:
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    det, info = track_video(
        video, track_cfg, annotated_out=out_dir / "annotated.mp4" if save_video else None
    )
    det.to_csv(out_dir / "detections.csv", index=False, float_format="%.2f")
    (out_dir / "meta.json").write_text(json.dumps(
        {"video": asdict(info), "track_config": asdict(track_cfg)}, indent=2
    ))

    cfg = A.AnalyticsConfig(fps=info.fps, vid_stride=track_cfg.vid_stride,
                            total_frames=info.frame_count, **analytics_kwargs)
    return write_analytics(det, info, zones, cfg, out_dir)


def reanalyze(out_dir: Path, zones: Zones, analytics_kwargs: dict) -> dict:
    """Recompute analytics from a previous run's detections.csv (e.g. after editing zones)."""
    meta = json.loads((out_dir / "meta.json").read_text())
    info = VideoInfo(**meta["video"])
    det = pd.read_csv(out_dir / "detections.csv")
    cfg = A.AnalyticsConfig(fps=info.fps, vid_stride=meta["track_config"]["vid_stride"],
                            total_frames=info.frame_count, **analytics_kwargs)
    return write_analytics(det, info, zones, cfg, out_dir)
