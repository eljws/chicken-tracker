"""Chicken skeletons over a whole video, matched to the named hens, for the player overlay.

The skeleton model (``train --pose``) finds each hen and her 5 midline points (beak, head,
neck, back, tail). Each skeleton is matched to a named detection in the same frame by box
overlap, so it carries the hen's name and colour. Output: ``poses.csv`` in the run folder,
one row per hen per frame, with x, y and confidence for every point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labeling import KEYPOINTS, SKELETON


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def run_pose(run_dir: Path, weights: Path, video: str | None = None, device: str | None = None,
             conf: float = 0.25, imgsz: int = 1280, min_iou: float = 0.4, progress: bool = True) -> Path:
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    video = video or meta["video"]["path"]
    named = pd.read_csv(run_dir / "detections_named.csv")
    by_frame = {int(f): g for f, g in named.groupby("frame")}
    model = YOLO(str(weights))
    rows = []
    total = meta["video"]["frame_count"]
    from .progress import Progress
    with Progress(run_dir, "skeletons", total=total) as prog:
        for f, r in enumerate(model.predict(source=str(video), stream=True, conf=conf, imgsz=imgsz,
                                            device=device, verbose=False)):
            prog.update(f + 1)
            if prog.preview_due() and r.keypoints is not None and len(r.boxes):
                prog.preview(r.plot(line_width=2, kpt_radius=6))
            if progress and f % 1500 == 0:
                print(f"  frame {f}/{total}", flush=True)
            g = by_frame.get(f)
            if g is None or r.keypoints is None or not len(r.boxes):
                continue
            pb = r.boxes.xyxy.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy()  # (n, K, 3): x, y, conf
            nb = g[["x1", "y1", "x2", "y2"]].to_numpy()
            iou = _iou_matrix(nb, pb)
            used = set()
            for i in np.argsort(-iou.max(1)):  # best matches first, one skeleton per hen
                j = int(iou[i].argmax())
                if iou[i, j] < min_iou or j in used:
                    continue
                used.add(j)
                row = {"frame": f, "track_id": int(g.track_id.iloc[i]), "chicken": g.chicken.iloc[i]}
                for k, name in enumerate(KEYPOINTS):
                    row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_c"] = (round(float(v), 1 if q < 2 else 2)
                                                                         for q, v in enumerate(kp[j, k]))
                rows.append(row)
    out = run_dir / "poses.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (run_dir / "poses.json").write_text(json.dumps({"keypoints": KEYPOINTS, "skeleton": SKELETON,
                                                    "weights": str(weights)}))
    print(f"{len(rows)} skeletons over {len({r['frame'] for r in rows})} frames -> {out}")
    return out


def overlay_poses(run_dir: Path, chickens: list[str], min_conf: float = 0.5) -> dict | None:
    """{"keypoints", "skeleton", "frames": {frame: [chicken_idx, x, y, seen, ...]}} for the player."""
    p = Path(run_dir) / "poses.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    idx = {n: i for i, n in enumerate(chickens)}
    df = df[df.chicken.isin(idx)]
    cols = []
    for k in KEYPOINTS:
        cols += [df[f"{k}_x"].round().astype(int), df[f"{k}_y"].round().astype(int), (df[f"{k}_c"] >= min_conf).astype(int)]
    arr = np.column_stack([df.chicken.map(idx).to_numpy(), *[c.to_numpy() for c in cols]]).astype(int)
    frames: dict[str, list[int]] = {}
    for f, row in zip(df.frame.to_numpy(), arr):
        frames.setdefault(str(int(f)), []).extend(row.tolist())
    return {"keypoints": KEYPOINTS, "skeleton": SKELETON, "frames": frames}
