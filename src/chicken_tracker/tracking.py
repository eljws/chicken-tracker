"""Detection + multi-object tracking over a video file.

Produces one row per (frame, chicken) with a persistent ``track_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import cv2
import pandas as pd

DETECTION_COLUMNS = [
    "frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "cls",
]

# Class names we treat as "chicken" when auto-selecting classes from a model.
# A custom-trained model should use "chicken"/"hen"/"rooster"; stock COCO
# weights only know "bird", which is the best available proxy.
_CHICKEN_NAMES = ("chicken", "hen", "rooster", "chick", "poultry")
_FALLBACK_NAMES = ("bird",)


def default_tracker_config() -> str:
    return str(resources.files("chicken_tracker") / "trackers" / "chicken_botsort.yaml")


@dataclass
class TrackConfig:
    model: str = "yolo11s.pt"
    classes: list[int] | None = None  # None = auto-detect chicken-like classes
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 960  # CCTV chickens are often small; larger than YOLO's 640 default helps
    vid_stride: int = 1
    tracker: str = field(default_factory=default_tracker_config)
    device: str | None = None  # e.g. "cpu", "mps", "0"


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return VideoInfo(
            path=str(path),
            fps=float(fps),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frame(path: str | Path, frame_index: int = 0):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index} from {path}")
        return frame
    finally:
        cap.release()


def resolve_classes(names: dict[int, str], requested: list[int] | None) -> list[int] | None:
    """Pick which model classes count as chickens."""
    if requested is not None:
        return requested
    lowered = {i: n.lower() for i, n in names.items()}
    for candidates in (_CHICKEN_NAMES, _FALLBACK_NAMES):
        hits = [i for i, n in lowered.items() if n in candidates]
        if hits:
            return hits
    return None  # custom single-purpose model: keep every class


def track_video(
    path: str | Path,
    cfg: TrackConfig,
    annotated_out: str | Path | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, VideoInfo]:
    """Run detection + tracking and return per-frame detections."""
    from ultralytics import YOLO  # heavy import, keep lazy

    info = probe_video(path)
    model = YOLO(cfg.model)
    classes = resolve_classes(model.names, cfg.classes)
    if progress:
        label = [model.names[c] for c in classes] if classes is not None else "all"
        print(f"[{Path(path).name}] {info.width}x{info.height} @ {info.fps:.1f}fps, "
              f"{info.frame_count} frames; tracking classes: {label}")

    writer = None
    if annotated_out is not None:
        out_fps = info.fps / cfg.vid_stride
        writer = cv2.VideoWriter(
            str(annotated_out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (info.width, info.height)
        )

    rows: list[tuple] = []
    results = model.track(
        source=str(path),
        stream=True,
        persist=True,
        tracker=cfg.tracker,
        classes=classes,
        conf=cfg.conf,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        vid_stride=cfg.vid_stride,
        device=cfg.device,
        verbose=False,
    )
    try:
        for i, r in enumerate(results):
            frame = i * cfg.vid_stride
            t = frame / info.fps
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.int().cpu().numpy()
                for (x1, y1, x2, y2), tid, c, k in zip(xyxy, ids, confs, clss):
                    rows.append((
                        frame, t, int(tid), float(x1), float(y1), float(x2), float(y2),
                        float((x1 + x2) / 2), float((y1 + y2) / 2), float(c), int(k),
                    ))
            if writer is not None:
                writer.write(r.plot(line_width=2))
            if progress and i % 200 == 0 and info.frame_count:
                pct = 100 * frame / info.frame_count
                print(f"  frame {frame}/{info.frame_count} ({pct:.0f}%)", flush=True)
    finally:
        if writer is not None:
            writer.release()

    return pd.DataFrame(rows, columns=DETECTION_COLUMNS), info
