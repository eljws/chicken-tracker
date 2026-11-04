"""Build a chicken detection dataset with a human answering yes/no.

1. ``mine``   - scan videos with a prompt-able (or previously trained) detector, keep a
                time-diverse set of frames plus their candidate boxes.
2. ``review`` - human accepts / rejects / adds boxes (see review_server.py).
3. ``export`` - write the reviewed frames in YOLO format for training.

Everything lives in one dataset folder::

    dataset/
      images/<video>__f<frame>.jpg
      labels.json   {image: {video, frame, time_s, proposals, boxes, status,
                             obj_proposals, objects, obj_status}}

Chickens (class 0) are reviewed in the first pass. Feeders and water (classes 1, 2) are
reviewed in a separate "objects" pass over the same frames (``review --objects``); once any
frame has objects reviewed, training uses only frames where both passes are done, so no
frame teaches the model that an unlabelled feeder is background.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import threading
from pathlib import Path

import cv2

OPEN_VOCAB_HINTS = ("yoloe", "world")

CLASSES = {0: "chicken", 1: "feeder", 2: "water"}
# chicken "skeleton" for the pose model, in click order; seen from above, feet are mostly hidden
KEYPOINTS = ["beak", "head", "neck", "back", "tail"]
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4)]
# text prompts per object class for the open-vocabulary proposal model
OBJECT_PROMPTS = {1: ["feed tray", "bowl of chicken feed", "chicken feeder"],
                  2: ["plastic bottle", "chicken waterer", "water bowl"]}


class LabelStore:
    """labels.json with atomic, thread-safe writes."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "labels.json"
        self.images = self.root / "images"
        self._lock = threading.Lock()
        self.items: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _reload(self) -> None:
        """Pick up changes saved by another page (review / objects / pose) before changing anything,
        so two pages open at once can't overwrite each other's work."""
        if self.path.exists():
            self.items = json.loads(self.path.read_text())

    def save(self) -> None:
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(self.items, f, indent=1)
            os.replace(tmp, self.path)

    def update(self, name: str, status: str, boxes: list[list[float]],
               outlines: list | None = None) -> None:
        """outlines: optional, one per box - a [[x, y], ...] polygon (click-to-outline) or None."""
        with self._lock:
            self._reload()
            self.items[name]["status"] = status
            self.items[name]["boxes"] = [[round(float(v), 1) for v in b[:4]] for b in boxes]
            if outlines is not None and len(outlines) == len(boxes) and any(outlines):
                self.items[name]["outlines"] = outlines
            else:
                self.items[name].pop("outlines", None)
            self.items[name].pop("auto_outlines", None)  # boxes changed: automatic outlines are stale
        self.save()

    def update_objects(self, name: str, status: str, objects: list[list[float]],
                       rejected: list[list[float]] = ()) -> None:
        """objects: [[x1, y1, x2, y2, cls], ...] with cls in OBJECT_PROMPTS.

        rejected: suggestion boxes the human turned down - remembered so the same spot
        (static camera) starts out rejected in other frames.
        """
        with self._lock:
            self._reload()
            self.items[name]["obj_status"] = status
            self.items[name]["objects"] = [[*(round(float(v), 1) for v in b[:4]), int(b[4])]
                                           for b in objects if int(b[4]) in OBJECT_PROMPTS]
            self.items[name]["obj_rejected"] = [[round(float(v), 1) for v in b[:4]] for b in rejected]
        self.save()

    def init_pose(self) -> int:
        """Make reviewed frames with chickens available to the pose pass; returns how many."""
        n = 0
        with self._lock:
            self._reload()
            for it in self.items.values():
                if it["status"] == "done" and it["boxes"] and "pose_status" not in it:
                    it["pose_status"] = "pending"
                    n += 1
        if n:
            self.save()
        return n

    def update_pose(self, name: str, status: str, poses: list) -> None:
        """poses: one per box - [[x, y, v] x len(KEYPOINTS)] (v 2 = seen, 0 = hidden) or None (skipped hen)."""
        with self._lock:
            self._reload()
            it = self.items[name]
            it["pose_status"] = status
            it["poses"] = [None if p is None else [[round(float(x), 1), round(float(y), 1), int(v)] for x, y, v in p]
                           for p in poses][:len(it["boxes"])]
        self.save()

    def counts(self, key: str = "status") -> dict[str, int]:
        out = {"pending": 0, "done": 0, "skipped": 0}
        for it in self.items.values():
            if key in it:
                out[it[key]] = out.get(it[key], 0) + 1
        return out


def load_detector(model: str, prompts: list[str]):
    from ultralytics import YOLO, YOLOE, YOLOWorld

    low = Path(model).name.lower()
    if "yoloe" in low:
        m = YOLOE(model)
        m.set_classes(prompts)
    elif "world" in low:
        m = YOLOWorld(model)
        m.set_classes(prompts)
    else:
        m = YOLO(model)
    return m


def mine(
    videos: list[Path],
    root: Path,
    model: str = "yoloe-11l-seg.pt",
    prompts: list[str] | None = None,
    every_s: float = 2.0,
    conf: float = 0.03,
    imgsz: int = 1280,
    max_frames: int = 300,
    negative_frac: float = 0.1,
    max_boxes: int = 10,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """Sample frames from videos and store candidate chicken boxes for review."""
    prompts = prompts or ["chicken"]
    store = LabelStore(root)
    store.images.mkdir(parents=True, exist_ok=True)
    det = load_detector(model, prompts)

    sampled: list[dict] = []  # every sampled frame (image written to disk)
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"skip (cannot open): {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(every_s * fps))
        print(f"[{video.name}] sampling every {step} frames ({total // step} samples)")
        idx = 0
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                name = f"{video.stem}__f{idx:07d}.jpg"
                if name not in store.items:
                    r = det.predict(frame, conf=conf, imgsz=imgsz, device=device,
                                    agnostic_nms=True, max_det=max_boxes, verbose=False)[0]
                    props = [[*map(lambda v: round(float(v), 1), b), round(float(c), 3)]
                             for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                    cv2.imwrite(str(store.images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    sampled.append({"name": name, "video": str(video), "frame": idx,
                                    "time_s": round(idx / fps, 2), "proposals": props})
                if (idx // step) % 100 == 0:
                    print(f"  {idx}/{total}", flush=True)
            idx += 1
        cap.release()

    positives = [s for s in sampled if s["proposals"]]
    negatives = [s for s in sampled if not s["proposals"]]
    n_neg = min(len(negatives), round(max_frames * negative_frac))
    n_pos = min(len(positives), max_frames - n_neg)
    chosen = _spread(positives, n_pos) + random.Random(seed).sample(negatives, n_neg)
    keep = {s["name"] for s in chosen}

    for s in sampled:
        if s["name"] in keep:
            name = s.pop("name")
            store.items[name] = {**s, "boxes": [], "status": "pending"}
        else:
            (store.images / s["name"]).unlink(missing_ok=True)
    store.save()
    return {"sampled": len(sampled), "with_candidates": len(positives),
            "added": len(keep), "total_in_dataset": len(store.items)}


def _nms(boxes: list[list[float]], iou: float) -> list[list[float]]:
    """Greedy NMS over [x1, y1, x2, y2, conf] boxes (highest conf first)."""
    keep: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(b, k) < iou for k in keep):
            keep.append(b)
    return keep


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def _inside(small, big) -> float:
    """Fraction of `small` covered by `big`."""
    w = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    h = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    return w * h / ((small[2] - small[0]) * (small[3] - small[1]) + 1e-9)


def split_merged(props: list[list[float]], dup_iou: float = 0.6) -> tuple[list[list[float]], list[list[float]]]:
    """(single-bird suggestions, merged-box suggestions).

    Near-duplicates are dropped; a box mostly containing two or more other suggestions is
    a merged box (several birds in one), shown pre-rejected in review.
    """
    kept = _nms(props, dup_iou)
    merged = [b for b in kept if sum(_inside(o, b) >= 0.7 for o in kept if o is not b
                                         and (o[2] - o[0]) * (o[3] - o[1]) < (b[2] - b[0]) * (b[3] - b[1])) >= 2]
    return [b for b in kept if b not in merged], merged


def mine_crowded(run_dir: Path, root: Path, model: str, n: int = 80, min_gap_s: float = 1.5,
                 video: str | None = None, device: str | None = None, conf: float = 0.05,
                 imgsz: int = 1280) -> dict:
    """Add frames where hens bunch up - the ones the detector boxes as one - for review.

    Picks frames from pieces set aside as "several chickens" (the detector's merged boxes)
    and frames where named birds' boxes overlap, spread over time. Suggestions come from
    the trained detector at low confidence with lenient overlap suppression, plus the
    open-vocabulary model, so separate birds inside a merged box are offered too. These
    frames are reviewed first (priority 1).
    """
    import numpy as np
    import pandas as pd

    from .identity import IdentityStore, load_detections

    run_dir = Path(run_dir)
    data = IdentityStore(run_dir).data
    det = load_detections(run_dir, data)
    fps, video = data["fps"], video or data["video"]
    multi = set(data.get("multiple", []))
    # crowding per frame: merged boxes count most, then overlapping/touching boxes
    xy = det[["x1", "y1", "x2", "y2"]].to_numpy()
    score: dict[int, float] = {}
    for f, idx in det.groupby("frame").indices.items():
        s = 3.0 * sum(int(t) in multi for t in det.track_id.to_numpy()[idx])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if _iou(xy[idx[a]], xy[idx[b]]) > 0.02:
                    s += 1.0
        if s > 0:
            score[int(f)] = s
    store = LabelStore(root)
    store.images.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem
    taken = [it["frame"] for it in store.items.values() if Path(it["video"]).stem == stem]
    gap = min_gap_s * fps
    chosen: list[int] = []
    for f in sorted(score, key=lambda f: (-score[f], f)):  # most crowded first, kept apart in time
        if all(abs(f - g) >= gap for g in chosen + taken):
            chosen.append(f)
        if len(chosen) >= n:
            break
    chosen.sort()
    trained = load_detector(model, ["chicken"])
    openv = load_detector("yoloe-11l-seg.pt", ["chicken"])
    cap = cv2.VideoCapture(str(video))
    added, pos = 0, 0
    for f in chosen:
        if f - pos > 150 or f < pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            pos = f
        while pos < f:
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break
        props = []
        for m, c in ((trained, conf), (openv, 0.1)):
            r = m.predict(frame, conf=c, iou=0.8, imgsz=imgsz, device=device, max_det=20, verbose=False)[0]
            props += [[*(round(float(v), 1) for v in b), round(float(k), 3)]
                      for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        name = f"{stem}__f{f:07d}.jpg"
        cv2.imwrite(str(store.images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        single, merged = split_merged(props)
        store.items[name] = {"video": str(video), "frame": int(f), "time_s": round(f / fps, 2),
                             "proposals": single[:12], "merged": merged, "boxes": [], "status": "pending",
                             "priority": 1, "source": "crowded"}
        added += 1
    cap.release()
    store.save()
    return {"added": added, "crowded_frames_seen": len(score), "total_in_dataset": len(store.items)}


def mine_objects(root: Path, model: str = "yoloe-11l-seg.pt", conf: float = 0.05,
                 imgsz: int = 1280, max_boxes: int = 8, device: str | None = None) -> dict:
    """Add feeder/water suggestions to every frame already in the dataset."""
    store = LabelStore(root)
    prompts, prompt_cls = [], []
    for cls, ps in OBJECT_PROMPTS.items():
        prompts += ps
        prompt_cls += [cls] * len(ps)
    det = load_detector(model, prompts)
    trained = not any(h in Path(model).name.lower() for h in OPEN_VOCAB_HINTS)
    # only frames whose chickens are reviewed: those are the ones training can use
    todo = [k for k, v in store.items.items() if v["status"] == "done" and v.get("obj_status") != "done"]
    for i, name in enumerate(todo):
        img = cv2.imread(str(store.images / name))
        r = det.predict(img, conf=conf, imgsz=imgsz, device=device, agnostic_nms=True,
                        max_det=50 if trained else max_boxes, verbose=False)[0]
        props = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            cls = int(k) if trained else prompt_cls[int(k)]
            if cls in OBJECT_PROMPTS:
                props.append([*(round(float(v), 1) for v in b), round(float(c), 3), cls])
        store.items[name]["obj_proposals"] = props[:max_boxes]
        store.items[name].setdefault("objects", [])
        store.items[name]["obj_status"] = "pending"
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    store.save()
    return {"frames": len(todo), "with_suggestions": sum(1 for k in todo if store.items[k]["obj_proposals"])}


def _spread(items: list, n: int) -> list:
    """Pick n items evenly spaced through a time-ordered list (avoids near-duplicates)."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    stride = len(items) / n
    return [items[int(i * stride)] for i in range(n)]


def split_train_val(items: dict[str, dict], val_frac: float = 0.2, block: int = 5) -> dict[str, str]:
    """Time-blocked, stratified split.

    Frames are sorted by time and grouped into blocks of ``block`` consecutive samples;
    every k-th block goes to val. Neighbouring (near-identical) frames therefore stay
    on the same side, and doing it separately for frames with and without chickens (and
    for crowded frames) guarantees val contains each kind. Adding a new kind of frame
    leaves the split of the others unchanged.
    """
    every = max(2, round(1 / val_frac))
    splits: dict[str, str] = {}
    kind = lambda v: (bool(v["boxes"]), v.get("source", ""))
    for group in sorted({kind(v) for v in items.values()}):
        keys = sorted((k for k, v in items.items() if kind(v) == group),
                      key=lambda k: (items[k]["video"], items[k]["frame"]))
        for i, k in enumerate(keys):
            splits[k] = "val" if (i // block) % every == every // 2 else "train"
    return splits


def export_yolo(root: Path, class_name: str = "chicken", with_objects: bool = False) -> Path:
    """Write reviewed frames as a YOLO dataset; returns path to data.yaml."""
    store = LabelStore(root)
    out = Path(root) / "yolo"
    if out.exists():
        shutil.rmtree(out)
    done = {k: v for k, v in store.items.items() if v["status"] == "done"}
    if not done:
        raise SystemExit("No reviewed frames yet - run `chicken-tracker review` first.")
    # Feeders/water come from stations.json, so the detector only needs chickens. Opt in to
    # also learning feeder/water (needs the objects pass done on most frames).
    if with_objects:
        # only frames where feeders/water were reviewed too, else they'd be learnt as background
        done = {k: v for k, v in done.items() if v.get("obj_status") == "done"}
    names = CLASSES if with_objects else {0: class_name}

    splits = split_train_val(done)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for name, it in done.items():
        split = splits[name]
        src = store.images / name
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        dst = out / "images" / split / name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        lines = []
        boxes = [(*b[:4], 0) for b in it["boxes"]]
        if with_objects:
            boxes += [tuple(o) for o in it.get("objects", [])]
        for x1, y1, x2, y2, cls in boxes:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))

    data = out / "data.yaml"
    data.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                    "names:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items()))
    n_val = sum(1 for s in splits.values() if s == "val")
    print(f"Exported {len(done)} frames ({len(done) - n_val} train / {n_val} val) -> {data}")
    return data


def train(root: Path, model: str = "yolo11s.pt", epochs: int = 100, imgsz: int = 1280,
          batch: int = 8, device: str | None = None, with_objects: bool = False,
          name: str = "chicken") -> Path:
    from ultralytics import YOLO

    data = export_yolo(root, with_objects=with_objects)
    runs = (Path(root) / "runs").resolve()
    YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                      project=str(runs), name=name, exist_ok=True, patience=50,
                      # Ultralytics' "auto" optimizer picks AdamW for small datasets, which diverged
                      # here (val cls loss -> 1e7, mAP 0). Plain SGD fine-tuning is stable.
                      optimizer="SGD", lr0=0.01,
                      fliplr=0.5, flipud=0.0, mosaic=1.0)
    best = runs / name / "weights" / "best.pt"
    print(f"\nTrained model: {best}\nUse it with: chicken-tracker process VIDEO --model {best}")
    return best


# ---------------------------------------------------------------- hen outlines (seg)

def auto_outline(root: Path, device: str | None = None, outliner=None, redo: bool = False) -> dict:
    """SAM outlines for reviewed hens that have a box but no outline (e.g. boxes drawn by hand).

    Stored per frame as ``auto_outlines`` (one per box, None where you outlined it yourself or
    SAM failed). Your own outlines are never touched. Returns counts.
    """
    if outliner is None:
        from .outline import Outliner
        outliner = Outliner(device=device)
    store = LabelStore(root)
    todo = [k for k, v in store.items.items() if v["status"] == "done" and v["boxes"]
            and (redo or "auto_outlines" not in v) and not _all_mine(v)]
    made = failed = 0
    for i, k in enumerate(sorted(todo)):
        it = store.items[k]
        mine = it.get("outlines") or []
        need = [j for j, b in enumerate(it["boxes"]) if not (j < len(mine) and mine[j])]
        if not need:
            continue
        outs = outliner.outline_boxes(store.images / k, [it["boxes"][j] for j in need], clip=True)
        auto = [None] * len(it["boxes"])
        for j, o in zip(need, outs):
            auto[j] = o["poly"] if o else None
            made += o is not None
            failed += o is None
        with store._lock:
            store._reload()
            if store.items[k]["boxes"] == it["boxes"]:  # unchanged while we worked
                store.items[k]["auto_outlines"] = auto
        store.save()
        if (i + 1) % 25 == 0:
            print(f"  outlined {i + 1}/{len(todo)} frames", flush=True)
    return {"frames": len(todo), "outlined": made, "failed": failed}


def _all_mine(it: dict) -> bool:
    mine = it.get("outlines") or []
    return len(mine) == len(it["boxes"]) and all(mine)


def frame_polys(it: dict) -> list | None:
    """One outline per box (yours first, else automatic), or None if any hen has no outline -
    a hen without an outline would teach the seg model that she's background."""
    mine, auto = it.get("outlines") or [], it.get("auto_outlines") or []
    polys = []
    for j in range(len(it["boxes"])):
        p = (mine[j] if j < len(mine) else None) or (auto[j] if j < len(auto) else None)
        if not p or len(p) < 3:
            return None
        polys.append(p)
    return polys


def export_seg(root: Path, exclude_source: str | None = None) -> Path:
    """Reviewed frames with every hen outlined, as a YOLO segmentation dataset; returns data.yaml.

    Frames with no hens are kept (they teach "empty yard"). ``exclude_source`` leaves a group
    of frames out (e.g. one day, to test the model on a day it never saw).
    """
    store = LabelStore(root)
    out = Path(root) / "yolo_seg"
    if out.exists():
        shutil.rmtree(out)
    done, dropped = {}, 0
    for k, v in store.items.items():
        if v["status"] != "done" or (exclude_source and v.get("source", "").startswith(exclude_source)):
            continue
        polys = frame_polys(v) if v["boxes"] else []
        if polys is None:
            dropped += 1
            continue
        done[k] = {**v, "polys": polys}
    if not done:
        raise SystemExit("No outlined frames yet - run `chicken-tracker outline` first.")
    splits = split_train_val(done)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for name, it in done.items():
        split = splits[name]
        src = store.images / name
        h, w = cv2.imread(str(src)).shape[:2]
        try:
            os.link(src, out / "images" / split / name)
        except OSError:
            shutil.copy2(src, out / "images" / split / name)
        lines = ["0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in poly) for poly in it["polys"]]
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
    data = out / "data.yaml"
    data.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: chicken\n")
    n_val = sum(1 for s in splits.values() if s == "val")
    print(f"Exported {len(done)} frames ({len(done) - n_val} train / {n_val} val), "
          f"{dropped} left out (a hen without an outline) -> {data}")
    return data


def train_seg(root: Path, model: str = "yolo11s-seg.pt", epochs: int = 100, imgsz: int = 1280,
              batch: int = 8, device: str | None = None, name: str = "seg",
              exclude_source: str | None = None) -> Path:
    from ultralytics import YOLO

    data = export_seg(root, exclude_source=exclude_source)
    runs = (Path(root) / "runs").resolve()
    YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                      project=str(runs), name=name, exist_ok=True, patience=50,
                      optimizer="SGD", lr0=0.01, fliplr=0.5, flipud=0.0, mosaic=1.0)
    best = runs / name / "weights" / "best.pt"
    print(f"\nTrained outline model: {best}")
    return best


# ---------------------------------------------------------------- chicken skeletons (pose)

def _frame_poses(it: dict, use_auto: bool, min_auto_conf: int) -> list | None:
    """Skeletons to train on for one frame: yours if you did this frame, else confident auto ones."""
    if it.get("pose_status") == "done" and any(it.get("poses") or []):
        return it["poses"]
    if use_auto and it.get("auto_poses"):
        ps = [p["pts"] if p and p["conf"] >= min_auto_conf else None for p in it["auto_poses"]]
        return ps if any(ps) else None
    return None


def export_pose(root: Path, use_auto: bool = True, min_auto_conf: int = 2) -> Path:
    """Frames with skeletons, as a YOLO pose dataset; returns data.yaml.

    Your skeletons (``review --pose``) always win; frames you haven't done use automatic
    skeletons (``auto_poses``, from outlines + movement/feeder/comb cues) whose head direction
    was confirmed by at least one strong cue. Every boxed hen is written (a box without a
    skeleton gets hidden keypoints), so no hen in a labelled frame is taught as background.
    """
    store = LabelStore(root)
    out = Path(root) / "yolo_pose"
    if out.exists():
        shutil.rmtree(out)
    done = {}
    for k, v in store.items.items():
        ps = _frame_poses(v, use_auto, min_auto_conf) if v["status"] == "done" and v["boxes"] else None
        if ps:
            done[k] = {**v, "poses": ps}
    if not done:
        raise SystemExit("No skeletons yet - run `chicken-tracker review --pose` first.")
    splits = split_train_val(done)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    nk = len(KEYPOINTS)
    for name, it in done.items():
        split = splits[name]
        src = store.images / name
        h, w = cv2.imread(str(src)).shape[:2]
        try:
            os.link(src, out / "images" / split / name)
        except OSError:
            shutil.copy2(src, out / "images" / split / name)
        poses = (it.get("poses") or []) + [None] * len(it["boxes"])
        lines = []
        for (x1, y1, x2, y2), pose in zip(it["boxes"], poses):
            kp = []
            for x, y, v in (pose or [[0, 0, 0]] * nk):
                kp += [f"{x / w:.6f}", f"{y / h:.6f}", str(int(v))] if v else ["0", "0", "0"]
            lines.append(" ".join([f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                   f"{abs(x2 - x1) / w:.6f} {abs(y2 - y1) / h:.6f}", *kp]))
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
    data = out / "data.yaml"
    data.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                    f"kpt_shape: [{nk}, 3]\nflip_idx: {list(range(nk))}\n"  # midline points: a flip keeps them
                    f"kpt_names:\n  0: {KEYPOINTS}\nnames:\n  0: chicken\n")
    n_val = sum(1 for s in splits.values() if s == "val")
    hens = sum(1 for v in done.values() for p in v["poses"] if p)
    print(f"Exported {len(done)} frames, {hens} skeletons ({len(done) - n_val} train / {n_val} val frames) -> {data}")
    return data


def train_pose(root: Path, model: str = "yolo11s-pose.pt", epochs: int = 150, imgsz: int = 1280,
               batch: int = 8, device: str | None = None, use_auto: bool = True) -> Path:
    from ultralytics import YOLO

    data = export_pose(root, use_auto=use_auto)
    runs = (Path(root) / "runs").resolve()
    YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                      project=str(runs), name="pose", exist_ok=True, patience=50,
                      optimizer="SGD", lr0=0.01, fliplr=0.5, flipud=0.0, mosaic=1.0)
    best = runs / "pose" / "weights" / "best.pt"
    print(f"\nTrained skeleton model: {best}")
    return best
