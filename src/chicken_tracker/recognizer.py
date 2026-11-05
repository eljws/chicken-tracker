"""A recogniser trained on *your* chickens: crop -> which hen (or "several").

Built from the naming work: every named piece gives labelled crops of that hen; pieces set
aside as "several chickens" give a ``several`` class, so the model also learns to spot
two-bird boxes. Pieces are split into train/val as whole pieces (neighbouring frames are
near-identical, so splitting by frame would make the test meaningless).
"""

from __future__ import annotations

import json
import time
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SEVERAL = "several"


def _crowding(det: pd.DataFrame) -> np.ndarray:
    """Per row: largest fraction of this box covered by another box in the same frame."""
    out = np.zeros(len(det))
    xy = det[["x1", "y1", "x2", "y2"]].to_numpy()
    for _, idx in det.groupby("frame").indices.items():
        if len(idx) < 2:
            continue
        b = xy[idx]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        for k, i in enumerate(idx):
            o = np.delete(b, k, axis=0)
            iw = np.clip(np.minimum(b[k, 2], o[:, 2]) - np.maximum(b[k, 0], o[:, 0]), 0, None)
            ih = np.clip(np.minimum(b[k, 3], o[:, 3]) - np.maximum(b[k, 1], o[:, 1]), 0, None)
            out[i] = float((iw * ih / area[k]).max())
    return out


def build_id_dataset(run_dir: Path, out: Path, video: str | None = None, every_s: float = 1.0,
                     val_frac: float = 0.2, max_crowding: float = 0.3, seed: int = 0) -> dict:
    from .identity import IdentityStore, load_detections

    run_dir, out = Path(run_dir), Path(out)
    data = IdentityStore(run_dir).data
    det = load_detections(run_dir, data)
    fps = data["fps"]
    video = video or data["video"]
    flagged = set(data.get("partial", [])) | set(data.get("loose", [])) | set(data.get("rejected", []))
    label = {t: (c["name"] or f"Chicken {c['id'] + 1}") for c in data["chickens"] for t in c["tracks"]}
    label |= {t: SEVERAL for t in data.get("multiple", [])}
    det = det[det.track_id.isin(label.keys()) & ~det.track_id.isin(flagged)].copy()
    det["crowd"] = _crowding(det)
    det["label"] = det.track_id.map(label)

    # pick ~1 crop per second per piece, skipping crowded crops of single hens
    step = max(1, int(round(every_s * fps)))
    picks = det[(det.frame % step == 0) & ((det.label == SEVERAL) | (det.crowd <= max_crowding))]

    # whole pieces to val, per class, ~val_frac of each class's crops
    rng = random.Random(seed)
    split = {}
    for lab, g in picks.groupby("label"):
        pieces = list(g.groupby("track_id").size().items())
        rng.shuffle(pieces)
        target, acc = val_frac * len(g), 0
        for t, n in pieces:
            split[t] = "val" if acc < target and len(pieces) > 1 else "train"
            acc += n if split[t] == "val" else 0
    picks = picks.assign(split=picks.track_id.map(split))

    if out.exists():
        shutil.rmtree(out)
    want = {f: g for f, g in picks.groupby("frame")}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    f, frames, written = 0, sorted(want), 0
    rows = []
    for target in frames:
        if target - f > 150:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            f = target
        while f < target:
            cap.grab()
            f += 1
        ok, img = cap.read()
        f += 1
        if not ok:
            break
        H, W = img.shape[:2]
        for _, r in want[target].iterrows():
            w, h = r.x2 - r.x1, r.y2 - r.y1
            x1, y1 = int(max(0, r.x1 - .1 * w)), int(max(0, r.y1 - .1 * h))
            x2, y2 = int(min(W, r.x2 + .1 * w)), int(min(H, r.y2 + .1 * h))
            d = out / r.split / r.label
            d.mkdir(parents=True, exist_ok=True)
            name = f"{run_dir.name}_t{int(r.track_id)}_f{int(r.frame)}.jpg"
            cv2.imwrite(str(d / name), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": f"{r.split}/{r.label}/{name}", "label": r.label, "split": r.split,
                         "track_id": int(r.track_id), "frame": int(r.frame)})
            written += 1
    cap.release()
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    counts = index.groupby(["label", "split"]).size().unstack(fill_value=0)
    return {"crops": written, "counts": counts}


def train_recognizer(dataset: Path, model: str = "yolo11s-cls.pt", epochs: int = 40, imgsz: int = 224,
                     device: str | None = None, batch: int = 64, lr0: float = 0.002) -> Path:
    from ultralytics import YOLO

    from . import records

    dataset = Path(dataset).resolve()
    rec = dataset.parent / "records"
    try:
        info = records.backup(dataset.parent, reason="before training recogniser")
        print(f"Backup before training: {info['path']} ({info['copied']} files copied, {info['linked']} unchanged)")
    except Exception as e:
        print(f"WARNING: backup before training failed: {e}")
    runs = dataset.parent / "recognizer_runs"
    YOLO(model).train(data=str(dataset), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                      project=str(runs), name="hens", exist_ok=True, patience=15, plots=False,
                      # SGD (AdamW diverged on small data before); low lr: lr0=0.01 made val accuracy
                      # swing 59-94% between epochs on ~3k crops
                      optimizer="SGD", lr0=lr0, cos_lr=True,
                      fliplr=0.5, flipud=0.0, hsv_h=0.0)  # no hue jitter: colour IS the identity
    best = runs / "hens" / "weights" / "best.pt"
    if best.exists():
        version = records.register("recognizer", best, version=f"hens-{time.strftime('%Y%m%d-%H%M')}",
                                   parent=records.version_of("recognizer", model, root=rec),
                                   trained_on={"dataset": str(dataset), "epochs": epochs, "imgsz": imgsz,
                                               "start": str(model)}, root=rec)
        print(f"Registered as recognizer {version}. Use it with: chicken-tracker models --use recognizer {version}")
    return best


def evaluate(dataset: Path, weights: Path, device: str | None = None) -> dict:
    """Crop- and piece-level accuracy on the held-out pieces."""
    from ultralytics import YOLO

    dataset = Path(dataset)
    idx = pd.read_csv(dataset / "index.csv")
    val = idx[idx.split == "val"].reset_index(drop=True)
    m = YOLO(str(weights))
    names = m.names
    probs = []
    for i in range(0, len(val), 64):
        rs = m.predict([str(dataset / p) for p in val.file[i:i + 64]], device=device, verbose=False, imgsz=224)
        probs += [r.probs.data.cpu().numpy() for r in rs]
    P = np.stack(probs)
    val["pred"] = [names[int(k)] for k in P.argmax(1)]
    crop_acc = float((val.pred == val.label).mean())
    piece = []
    for t, g in val.groupby("track_id"):
        mean = P[g.index].mean(0)
        piece.append((t, g.label.iloc[0], names[int(mean.argmax())], len(g)))
    piece = pd.DataFrame(piece, columns=["track_id", "label", "pred", "n"])
    confusion = pd.crosstab(val.label, val.pred)
    return {"crop_acc": crop_acc, "piece_acc": float((piece.label == piece.pred).mean()),
            "pieces": piece, "confusion": confusion, "val_crops": len(val)}


# ------------------------------------------------------------------ cross-fitted predictions
#
# To audit labels, each piece must be judged by a model that never trained on it (otherwise
# a mislabelled piece is "confirmed" by having been learnt). So: split pieces into K folds,
# train K models each without one fold, and predict every 1/s crop of every piece with the
# model that left its piece out. Predictions are keyed by (frame, box), which cuts don't
# change, so they stay valid as pieces are split and renamed.

PRED_FILE, PRED_META = "recognizer.csv", "recognizer.json"


def fold_of(track_id: int, folds: int) -> int:
    return (int(track_id) * 2654435761 % 2**32) % folds  # stable, well-mixed


def _crop(img: np.ndarray, r) -> np.ndarray:
    H, W = img.shape[:2]
    w, h = r.x2 - r.x1, r.y2 - r.y1
    x1, y1 = int(max(0, r.x1 - .1 * w)), int(max(0, r.y1 - .1 * h))
    x2, y2 = int(min(W, r.x2 + .1 * w)), int(min(H, r.y2 + .1 * h))
    return img[y1:y2, x1:x2]


def _all_crops(run_dir: Path, det: pd.DataFrame, video: str, every_s: float, fps: float, out: Path,
               on_frame=None) -> pd.DataFrame:
    """1/s crop of every detection of every piece (named or not), written to `out`."""
    step = max(1, int(round(every_s * fps)))
    rows = det[det.frame % step == 0].copy()
    rows["crowd"] = _crowding(det)[(det.frame % step == 0).to_numpy()]
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    f, files = 0, {}
    for target, g in rows.groupby("frame"):
        if target - f > 150:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            f = target
        while f < target:
            cap.grab()
            f += 1
        ok, img = cap.read()
        f += 1
        if not ok:
            break
        if on_frame is not None:
            on_frame(int(target))
        for i, r in g.iterrows():
            p = out / f"t{int(r.track_id)}_f{int(r.frame)}_{i}.jpg"
            cv2.imwrite(str(p), _crop(img, r), [cv2.IMWRITE_JPEG_QUALITY, 92])
            files[i] = p
    cap.release()
    rows = rows.loc[list(files)]
    rows["path"] = [str(files[i]) for i in rows.index]
    return rows


def _predict(weights: Path, paths: list[str], device: str | None) -> tuple[np.ndarray, list[str]]:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    probs = []
    for i in range(0, len(paths), 64):
        rs = m.predict(paths[i:i + 64], device=device, verbose=False, imgsz=224)
        probs += [r.probs.data.cpu().numpy() for r in rs]
    return (np.stack(probs) if probs else np.zeros((0, len(m.names)))), [m.names[k] for k in sorted(m.names)]


def cross_fit(run_dir: Path, dataset: Path = Path("dataset_id"), folds: int = 4, epochs: int = 12,
              device: str | None = None, video: str | None = None, every_s: float = 1.0) -> dict:
    """Train `folds` recognisers, each without one fold of pieces; predict every piece out-of-fold."""
    from ultralytics import YOLO

    from .identity import IdentityStore, load_detections

    run_dir, dataset = Path(run_dir), Path(dataset).resolve()
    data = IdentityStore(run_dir).data
    det = load_detections(run_dir, data)
    build_id_dataset(run_dir, dataset, video=video, every_s=every_s)  # fresh labels
    idx = pd.read_csv(dataset / "index.csv")
    idx["fold"] = idx.track_id.map(lambda t: fold_of(t, folds))
    work = dataset.parent / "recognizer_runs" / "folds"
    if work.exists():
        shutil.rmtree(work)
    print("Cropping every piece once a second ...")
    crops = _all_crops(run_dir, det, video or data["video"], every_s, data["fps"], work / "all_crops")
    crops["fold"] = crops.track_id.map(lambda t: fold_of(t, folds))

    parts, classes, acc = [], None, {}
    for k in range(folds):
        root = work / f"fold{k}"
        for _, r in idx.iterrows():  # train = other folds, val = this fold (accuracy for free)
            d = root / ("val" if r.fold == k else "train") / r.label
            d.mkdir(parents=True, exist_ok=True)
            (d / Path(r.file).name).symlink_to(dataset / r.file)
        for lab in idx.label.unique():  # same class list in every fold
            for split in ("train", "val"):
                (root / split / lab).mkdir(parents=True, exist_ok=True)
        print(f"Fold {k + 1}/{folds}: training without {int((idx.fold == k).sum())} crops ...")
        YOLO("yolo11s-cls.pt").train(data=str(root), epochs=epochs, imgsz=224, batch=64, device=device,
                                     project=str(work), name=f"run{k}", exist_ok=True, plots=False,
                                     optimizer="SGD", lr0=0.002, cos_lr=True, patience=epochs,
                                     fliplr=0.5, flipud=0.0, hsv_h=0.0, verbose=False)
        w = work / f"run{k}" / "weights" / "last.pt"
        sel = crops[crops.fold == k]
        P, classes = _predict(w, sel.path.tolist(), device)
        parts.append(pd.concat([sel.reset_index(drop=True),
                                pd.DataFrame(P, columns=[f"p_{c}" for c in classes])], axis=1))
        v = idx[idx.fold == k]
        if len(v):
            pv, _ = _predict(w, [str(dataset / f) for f in v.file], device)
            acc[k] = float((np.array(classes)[pv.argmax(1)] == v.label.to_numpy()).mean())

    pred = pd.concat(parts, ignore_index=True)
    keep = ["frame", "x1", "y1", "x2", "y2", "track_id", "fold", "crowd"] + [f"p_{c}" for c in classes]
    out = run_dir / "identity" / PRED_FILE
    pred[keep].to_csv(out, index=False, float_format="%.4f")
    (run_dir / "identity" / PRED_META).write_text(json.dumps(
        {"classes": classes, "folds": folds, "fold_accuracy": acc, "every_s": every_s}, indent=1))
    shutil.rmtree(work / "all_crops", ignore_errors=True)
    return {"crops": len(pred), "fold_accuracy": acc, "file": str(out)}


def load_predictions(run_dir: Path, det: pd.DataFrame) -> tuple[pd.DataFrame, list[str]] | None:
    """Out-of-fold predictions joined to the *current* pieces (after cuts) by frame + box."""
    p = Path(run_dir) / "identity" / PRED_FILE
    if not p.exists():
        return None
    classes = json.loads((p.parent / PRED_META).read_text())["classes"]
    pred = pd.read_csv(p).drop(columns=["track_id"])
    key = lambda d: d.assign(_x=d.x1.round().astype(int), _y=d.y1.round().astype(int))
    cur = key(det[["frame", "x1", "y1", "track_id"]])[["frame", "_x", "_y", "track_id"]]
    pred = key(pred).merge(cur.drop_duplicates(["frame", "_x", "_y"]), on=["frame", "_x", "_y"])
    return pred.drop(columns=["_x", "_y"]), classes


# ------------------------------------------------------------------ naming a new day

def predict_run(run_dir: Path, weights: Path, device: str | None = None, video: str | None = None,
                every_s: float = 1.0) -> tuple[pd.DataFrame, list[str]]:
    """Recogniser verdicts for 1/s crops of every piece of a run the model never saw.

    Written in the same format as ``cross_fit`` (fold -1), so the Checks tab's recogniser
    checks work on this run too.
    """
    from .identity import IdentityStore, load_detections

    run_dir = Path(run_dir)
    data = IdentityStore(run_dir).data
    det = load_detections(run_dir, data)
    from .progress import Progress

    tmp = run_dir / "identity" / "rec_crops"
    shutil.rmtree(tmp, ignore_errors=True)
    last = int(det.frame.max()) if len(det) else 0
    with Progress(run_dir, "naming", total=last) as prog:
        prog.update(0, phase="cutting a crop per second of every piece", force=True)
        crops = _all_crops(run_dir, det, video or data["video"], every_s, data["fps"], tmp,
                           on_frame=lambda f: prog.update(f))
        from .preview_img import crop_grid
        prog.preview(crop_grid(crops.path.sample(min(40, len(crops)), random_state=0).tolist()), width=800)
        prog.update(last, phase=f"recogniser on {len(crops)} crops", force=True)
        P, classes = _predict(weights, crops.path.tolist(), device)
        # the picture that matters: what the recogniser thinks each hen looks like
        from .preview_img import hen_rows
        lab = np.array(classes)[P.argmax(1)]
        rows = {}
        for c in classes:
            sel = np.flatnonzero((lab == c) & (P.max(1) >= 0.6) & (crops.crowd.to_numpy() <= 0.3))
            if len(sel):
                pick = np.random.default_rng(0).choice(sel, min(10, len(sel)), replace=False)
                rows[f"{c} ({(lab == c).sum()} crops)"] = [crops.path.iloc[i] for i in pick]
        prog.preview(hen_rows(rows), width=1000)
    shutil.rmtree(tmp, ignore_errors=True)
    pred = pd.concat([crops.reset_index(drop=True).drop(columns=["path"]),
                      pd.DataFrame(P, columns=[f"p_{c}" for c in classes])], axis=1)
    pred["fold"] = -1
    keep = ["frame", "x1", "y1", "x2", "y2", "track_id", "fold", "crowd"] + [f"p_{c}" for c in classes]
    pred[keep].to_csv(run_dir / "identity" / PRED_FILE, index=False, float_format="%.4f")
    (run_dir / "identity" / PRED_META).write_text(json.dumps(
        {"classes": classes, "folds": 0, "weights": str(weights), "every_s": every_s,
         "note": "full model on a run it never saw"}, indent=1))
    return pred[keep], classes


def assign_pieces(verdict: dict, frags: dict, classes: list[str], min_p: float = 0.15,
                  several_p: float = 0.5) -> tuple[list[list[int]], list[int], dict]:
    """verdict: {track: (mean probs over classes, n crops)} -> (tracks per hen, several pieces, report).

    Most certain pieces first; each to the likeliest hen not already elsewhere at that time.
    """
    from .identity import _conflict

    hens = [c for c in classes if c != SEVERAL]
    tracks: list[list[int]] = [[] for _ in hens]
    placed: list[list] = [[] for _ in hens]
    several: list[int] = []
    report = {"named": 0, "second_choice": 0, "several": 0, "conflict": 0}
    for t in sorted(verdict, key=lambda t: -(verdict[t][0].max() * np.log1p(verdict[t][1]))):
        if t not in frags:
            continue
        p = verdict[t][0]
        if classes[int(p.argmax())] == SEVERAL and p.max() >= several_p:
            several.append(t)
            report["several"] += 1
            continue
        ranked = sorted(((float(p[classes.index(h)]), i) for i, h in enumerate(hens)), reverse=True)
        choice = None
        for k, (ph, i) in enumerate(ranked):
            if k and ph < min_p:
                break
            if not _conflict([frags[t]], placed[i]):
                choice = i
                report["second_choice" if k else "named"] += 1
                break
        if choice is None:  # nobody free: keep the best guess, it shows up as a conflict
            choice = ranked[0][1]
            report["conflict"] += 1
        placed[choice].append(frags[t])
        tracks[choice].append(t)
    return [sorted(t) for t in tracks], sorted(several), report


def auto_name(run_dir: Path, weights: Path, device: str | None = None, video: str | None = None,
              min_p: float = 0.15, several_p: float = 0.5) -> dict:
    """Name every piece of a freshly built run with the recogniser.

    Pieces are taken most-certain first; each goes to the most likely hen that isn't already
    somewhere else at the same time (else her next most likely, down to ``min_p``). Pieces the
    recogniser calls "several" are set aside. Pieces too short for a crop go to the nearest
    hen by appearance. Nothing is marked as confirmed: the naming page still asks about doubtful
    pieces, and conflicts / checks show what to look at.
    """
    from .identity import IdentityStore, export, load_fragments, place_piece, refresh_confidence

    run_dir = Path(run_dir)
    pred, classes = predict_run(run_dir, weights, device, video)
    store = IdentityStore(run_dir)
    data = store.data
    frags = load_fragments(run_dir)
    hens = [c for c in classes if c != SEVERAL]
    cols = [f"p_{c}" for c in classes]
    verdict = {}
    for t, g in pred.groupby("track_id"):
        clear = g[g.crowd <= 0.3]
        p = (clear if len(clear) else g)[cols].mean().to_numpy()
        verdict[int(t)] = (p, len(g))

    tracks, several, report = assign_pieces(verdict, frags, classes, min_p, several_p)
    data["chickens"] = [{"id": i, "name": h, "tracks": tracks[i]} for i, h in enumerate(hens)]
    data["multiple"] = several
    for key in ("rejected", "partial", "loose"):
        data[key] = []
    report["by_appearance"] = 0
    for t in frags:  # pieces too short for a 1/s crop
        if t not in verdict:
            report["by_appearance"] += place_piece(data, frags, t) is not None
    for c in data["chickens"]:
        c["tracks"].sort()
    data["answers"] = []
    data["auto_named"] = {"weights": str(weights), **report}
    refresh_confidence(data, frags)
    store.save()
    export(run_dir)
    return report
