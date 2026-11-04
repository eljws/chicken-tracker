"""Detection + multi-object tracking over a video file.

Produces one row per (frame, chicken) with a persistent ``track_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
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
    skip_empty: bool = False   # quick scan first; track only the stretches where hens are in view
    scan_every_s: float = 1.0  # quick scan: one look per second ...
    scan_imgsz: int = 640      # ... at low detail ...
    scan_conf: float = 0.15    # ... with a low bar (a hen missed in one look is caught in the next)
    scan_pad_s: float = 5.0    # track from this long before a hen is first seen to this long after


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


def scan_for_hens(path: str | Path, model, cfg: "TrackConfig", info: "VideoInfo", classes=None,
                  on_frame=None) -> list[tuple[int, int]]:
    """Quick look once a second at low detail; returns (first, last) frame stretches with hens, padded."""
    step = max(1, int(round(cfg.scan_every_s * info.fps)))
    pad = int(round(cfg.scan_pad_s * info.fps))
    cap = cv2.VideoCapture(str(path))
    seen, f = [], 0
    try:
        while True:
            if not cap.grab():
                break
            if f % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    r = model.predict(img, conf=cfg.scan_conf, imgsz=cfg.scan_imgsz, classes=classes,
                                      device=cfg.device, verbose=False)[0]
                    if len(r.boxes):
                        seen.append(f)
                if on_frame is not None:
                    on_frame(f + 1, info.frame_count, None)
            f += 1
    finally:
        cap.release()
    last = max(f, info.frame_count) - 1
    spans: list[list[int]] = []
    for s in seen:
        a, b = max(0, s - pad), min(last, s + pad)
        if spans and a <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans]


def track_video(
    path: str | Path,
    cfg: TrackConfig,
    annotated_out: str | Path | None = None,
    progress: bool = True,
    on_frame=None,
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
    if cfg.skip_empty and cfg.vid_stride == 1:
        try:
            _track_segments(path, model, cfg, info, classes, rows, writer, on_frame, progress)
        finally:
            if writer is not None:
                writer.release()
        return drop_container_boxes(pd.DataFrame(rows, columns=DETECTION_COLUMNS)), info
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
            if on_frame is not None:
                on_frame(frame + 1, info.frame_count, r)
            if progress and i % 200 == 0 and info.frame_count:
                pct = 100 * frame / info.frame_count
                print(f"  frame {frame}/{info.frame_count} ({pct:.0f}%)", flush=True)
    finally:
        if writer is not None:
            writer.release()

    return drop_container_boxes(pd.DataFrame(rows, columns=DETECTION_COLUMNS)), info


def _track_segments(path, model, cfg, info, classes, rows, writer, on_frame, progress) -> None:
    """Scan, then track only the stretches with hens. The tracker restarts in each stretch;
    its ids are offset so pieces from different stretches never share an id."""
    phase = {"name": "scan"}

    def report(f, total, r):
        if on_frame is not None:
            on_frame(f, total, r, phase["name"])

    spans = scan_for_hens(path, model, cfg, info, classes, on_frame=report)
    kept = sum(b - a + 1 for a, b in spans)
    if progress:
        print(f"  hens in view in {len(spans)} stretches, {kept}/{info.frame_count} frames "
              f"({100 * kept / max(1, info.frame_count):.0f}%) - tracking only those", flush=True)
    phase["name"] = "track"
    phase["kept"] = kept
    cap = cv2.VideoCapture(str(path))
    f, k, offset, max_id, done = 0, 0, 0, 0, 0
    try:
        while k < len(spans):
            a, b = spans[k]
            if f < a:  # skip to the stretch (decoding only)
                if not cap.grab():
                    break
                f += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            r = model.track(img, persist=f != a, tracker=cfg.tracker, classes=classes, conf=cfg.conf,
                            iou=cfg.iou, imgsz=cfg.imgsz, device=cfg.device, verbose=False)[0]
            t = f / info.fps
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                for (x1, y1, x2, y2), tid, c, cl in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy(),
                                                          boxes.conf.cpu().numpy(), boxes.cls.int().cpu().numpy()):
                    tid = int(tid) + offset
                    max_id = max(max_id, tid)
                    rows.append((f, t, tid, float(x1), float(y1), float(x2), float(y2),
                                 float((x1 + x2) / 2), float((y1 + y2) / 2), float(c), int(cl)))
            if writer is not None:
                writer.write(r.plot(line_width=2))
            done += 1
            if on_frame is not None:
                on_frame(done, kept, r, "track")
            if progress and done % 500 == 0:
                print(f"  tracked {done}/{kept} frames with hens (video frame {f})", flush=True)
            f += 1
            if f > b:          # stretch finished: next one starts a fresh tracker with new ids
                k += 1
                offset = max_id
    finally:
        cap.release()

def drop_container_boxes(det: pd.DataFrame, inside: float = 0.7, bigger: float = 1.3) -> pd.DataFrame:
    """Drop boxes drawn around two or more hens that also have their own boxes in that frame.

    A box counts as one if at least two other boxes of the frame lie mostly (``inside``) within
    it and it is clearly (``bigger``) larger than them. Both hens are already tracked, so the
    big box is a ghost: an extra "hen", wrong speeds, a blob for skeletons. A box standing in
    for two hens pressed together (no smaller boxes inside) is kept - it's the only trace of them.
    """
    drop = container_box_index(det, inside, bigger)
    return det.drop(index=drop).reset_index(drop=True) if drop else det


def container_box_index(det: pd.DataFrame, inside: float = 0.7, bigger: float = 1.3) -> list:
    """Index labels of the rows ``drop_container_boxes`` removes."""
    drop = []
    for _, g in det.groupby("frame", sort=False):
        if len(g) < 3:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        w = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
        h = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
        share = w * h / np.maximum(a[None, :], 1e-9)        # [i, j]: share of box j inside box i
        np.fill_diagonal(share, 0)
        n = ((share >= inside) & (a[:, None] > bigger * a[None, :])).sum(1)
        drop.extend(g.index[n >= 2])
    return drop
