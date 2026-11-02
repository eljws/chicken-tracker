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

    def save(self) -> None:
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(self.items, f, indent=1)
            os.replace(tmp, self.path)

    def update(self, name: str, status: str, boxes: list[list[float]]) -> None:
        with self._lock:
            self.items[name]["status"] = status
            self.items[name]["boxes"] = [[round(float(v), 1) for v in b[:4]] for b in boxes]
        self.save()

    def update_objects(self, name: str, status: str, objects: list[list[float]],
                       rejected: list[list[float]] = ()) -> None:
        """objects: [[x1, y1, x2, y2, cls], ...] with cls in OBJECT_PROMPTS.

        rejected: suggestion boxes the human turned down - remembered so the same spot
        (static camera) starts out rejected in other frames.
        """
        with self._lock:
            self.items[name]["obj_status"] = status
            self.items[name]["objects"] = [[*(round(float(v), 1) for v in b[:4]), int(b[4])]
                                           for b in objects if int(b[4]) in OBJECT_PROMPTS]
            self.items[name]["obj_rejected"] = [[round(float(v), 1) for v in b[:4]] for b in rejected]
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
    on the same side, and doing it separately for frames with and without chickens
    guarantees val contains chickens.
    """
    every = max(2, round(1 / val_frac))
    splits: dict[str, str] = {}
    for has_chickens in (True, False):
        keys = sorted((k for k, v in items.items() if bool(v["boxes"]) == has_chickens),
                      key=lambda k: (items[k]["video"], items[k]["frame"]))
        for i, k in enumerate(keys):
            splits[k] = "val" if (i // block) % every == every // 2 else "train"
    return splits


def export_yolo(root: Path, class_name: str = "chicken") -> Path:
    """Write reviewed frames as a YOLO dataset; returns path to data.yaml."""
    store = LabelStore(root)
    out = Path(root) / "yolo"
    if out.exists():
        shutil.rmtree(out)
    done = {k: v for k, v in store.items.items() if v["status"] == "done"}
    if not done:
        raise SystemExit("No reviewed frames yet - run `chicken-tracker review` first.")
    with_objects = any(v.get("obj_status") == "done" for v in store.items.values())
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
          batch: int = 8, device: str | None = None) -> Path:
    from ultralytics import YOLO

    data = export_yolo(root)
    runs = (Path(root) / "runs").resolve()
    YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                      project=str(runs), name="chicken", exist_ok=True, patience=50,
                      # Ultralytics' "auto" optimizer picks AdamW for small datasets, which diverged
                      # here (val cls loss -> 1e7, mAP 0). Plain SGD fine-tuning is stable.
                      optimizer="SGD", lr0=0.01,
                      fliplr=0.5, flipud=0.0, mosaic=1.0)
    best = runs / "chicken" / "weights" / "best.pt"
    print(f"\nTrained model: {best}\nUse it with: chicken-tracker process VIDEO --model {best}")
    return best
