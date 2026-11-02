"""Turn anonymous track fragments into named chickens.

Pipeline (all state lives in ``<run_dir>/identity/``)::

    build      crops -> appearance embeddings -> constrained clustering
    (review)   human names clusters and answers "is this Rocky?" (identity_server.py)
    export     per-chicken timeline incl. inferred "hidden" gaps + per-chicken summary

Two fragments can never be the same chicken if they are visible at the same time
(cannot-link). Fragments that end/start close in time and space get a continuity
bonus, so a chicken that walks behind a bush and reappears is linked back up.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CROPS_PER_TRACK = 8
OVERLAP_TOLERANCE_S = 0.5   # tracker may briefly double-track one bird
CONFLICT_MIN_OVERLAP_S = 1.5  # shorter same-name overlaps are tracker hand-overs, not real conflicts
MAX_SPEED_PX_S = 250.0      # faster than this between fragments = implausible link
CONTINUITY_TAU_S = 15.0
CONTINUITY_WEIGHT = 0.15


# --------------------------------------------------------------------------- crops

def _iou_any(boxes: np.ndarray, i: int) -> float:
    """Max IoU of box i with any other box in the same frame."""
    if len(boxes) < 2:
        return 0.0
    b = boxes[i]
    others = np.delete(boxes, i, axis=0)
    ix1 = np.maximum(b[0], others[:, 0]); iy1 = np.maximum(b[1], others[:, 1])
    ix2 = np.minimum(b[2], others[:, 2]); iy2 = np.minimum(b[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return float((inter / (area(b) + area(others) - inter + 1e-9)).max())


def select_crops(det: pd.DataFrame, track_ids: list[int], per_track: int = CROPS_PER_TRACK) -> pd.DataFrame:
    """Pick the clearest, time-spread detections of each track to crop."""
    det = det.copy()
    xy = det[["x1", "y1", "x2", "y2"]].to_numpy()
    overlap = np.zeros(len(det))
    for _, idx in det.groupby("frame").indices.items():
        if len(idx) > 1:
            for k, i in enumerate(idx):
                overlap[i] = _iou_any(xy[idx], k)
    det["overlap"] = overlap
    det["area"] = (det.x2 - det.x1) * (det.y2 - det.y1)

    picks = []
    for tid in track_ids:
        t = det[det.track_id == tid]
        clean = t[t.overlap < 0.05]
        pool = clean if len(clean) >= min(per_track, len(t)) else t
        pool = pool.sort_values("frame")
        # split track into equal time slices, take the best-scoring detection in each
        n = min(per_track, len(pool))
        for part in np.array_split(np.arange(len(pool)), n):
            chunk = pool.iloc[part]
            score = chunk.conf * np.sqrt(chunk.area)
            picks.append(chunk.loc[score.idxmax()])
    return pd.DataFrame(picks).reset_index(drop=True)


def extract_crops(video: str, picks: pd.DataFrame, out_dir: Path, pad: float = 0.1) -> list[str]:
    """Read the video once, sequentially, saving each picked crop as JPEG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    want = picks.sort_values("frame")
    by_frame = {f: g for f, g in want.groupby("frame")}
    frames = sorted(by_frame)
    paths: dict[int, str] = {}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video} (moved?)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    f = frames[0]
    fi = 0
    while fi < len(frames):
        target = frames[fi]
        if target - f > 150:  # long jump: seeking is cheaper than decoding
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            f = target
        while f < target:
            cap.grab(); f += 1
        ok, img = cap.read(); f += 1
        if not ok:
            break
        H, W = img.shape[:2]
        for i, r in by_frame[target].iterrows():
            w, h = r.x2 - r.x1, r.y2 - r.y1
            x1, y1 = int(max(0, r.x1 - pad * w)), int(max(0, r.y1 - pad * h))
            x2, y2 = int(min(W, r.x2 + pad * w)), int(min(H, r.y2 + pad * h))
            name = f"t{int(r.track_id)}_f{int(r.frame)}.jpg"
            cv2.imwrite(str(out_dir / name), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths[i] = name
        fi += 1
    cap.release()
    return [paths.get(i, "") for i in picks.index]


# ----------------------------------------------------------------------- features

def _color_hist(img: np.ndarray) -> np.ndarray:
    """HSV histogram of the crop centre (plumage colour, little background)."""
    h, w = img.shape[:2]
    c = img[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist = np.sqrt(hist / (hist.sum() + 1e-9))  # Hellinger
    return hist / (np.linalg.norm(hist) + 1e-9)


def embed_crops(paths: list[Path], model: str = "yolo11s-cls.pt", device: str | None = None,
                color_weight: float = 0.5) -> np.ndarray:
    """Appearance vector per crop: CNN embedding + colour histogram, L2-normalised."""
    from ultralytics import YOLO

    m = YOLO(model)
    imgs = [cv2.imread(str(p)) for p in paths]
    deep = []
    for i in range(0, len(imgs), 32):
        out = m.embed(imgs[i:i + 32], imgsz=224, device=device, verbose=False)
        deep += [o.cpu().numpy().ravel() for o in out]
    deep = np.stack(deep)
    deep /= np.linalg.norm(deep, axis=1, keepdims=True) + 1e-9
    col = np.stack([_color_hist(im) for im in imgs])
    feat = np.hstack([deep * (1 - color_weight) ** .5, col * color_weight ** .5])
    return feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------- clustering

@dataclass
class Frag:
    tid: int
    start: float
    end: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    n: int
    emb: np.ndarray  # mean appearance, normalised


def fragments(det: pd.DataFrame, picks: pd.DataFrame, feats: np.ndarray) -> dict[int, Frag]:
    out = {}
    for tid, t in det.groupby("track_id"):
        rows = np.flatnonzero(picks.track_id.to_numpy() == tid)
        if not len(rows):
            continue
        e = feats[rows].mean(0)
        t = t.sort_values("time_s")
        out[int(tid)] = Frag(int(tid), float(t.time_s.iloc[0]), float(t.time_s.iloc[-1]),
                             (float(t.cx.iloc[0]), float(t.y2.iloc[0])),
                             (float(t.cx.iloc[-1]), float(t.y2.iloc[-1])),
                             len(t), e / (np.linalg.norm(e) + 1e-9))
    return out


def _conflict(a: list[Frag], b: list[Frag]) -> bool:
    for x in a:
        for y in b:
            if min(x.end, y.end) - max(x.start, y.start) > OVERLAP_TOLERANCE_S:
                return True
    return False


def _continuity(a: list[Frag], b: list[Frag]) -> float:
    """Bonus if one cluster's fragment plausibly continues into the other's."""
    best = 0.0
    for x in a:
        for y in b:
            first, second = (x, y) if x.end <= y.start else (y, x)
            gap = second.start - first.end
            if gap < -OVERLAP_TOLERANCE_S:
                continue
            gap = max(gap, 1e-3)
            dist = np.hypot(second.start_xy[0] - first.end_xy[0], second.start_xy[1] - first.end_xy[1])
            if dist / gap <= MAX_SPEED_PX_S or dist < 80:
                best = max(best, float(np.exp(-gap / CONTINUITY_TAU_S)))
    return best


def _centroid(frags: list[Frag]) -> np.ndarray:
    w = np.array([np.sqrt(f.n) for f in frags])
    e = (np.stack([f.emb for f in frags]) * w[:, None]).sum(0)
    return e / (np.linalg.norm(e) + 1e-9)


def cluster(frags: dict[int, Frag], threshold: float = 0.80, max_clusters: int | None = None,
            must: list[tuple[int, int]] = (), cannot: list[tuple[int, int]] = ()) -> list[list[int]]:
    """Greedy agglomerative clustering with cannot-link (time overlap) constraints."""
    clusters: list[list[Frag]] = [[f] for f in sorted(frags.values(), key=lambda f: f.start)]
    cannot = {frozenset(p) for p in cannot}
    idx = {f.tid: i for i, c in enumerate(clusters) for f in c}

    def merge(i: int, j: int):
        clusters[i] += clusters[j]
        clusters[j] = []

    for a, b in must:  # human-confirmed links first
        if a in idx and b in idx and idx[a] != idx[b]:
            i, j = idx[a], idx[b]
            merge(i, j)
            for f in clusters[i]:
                idx[f.tid] = i

    def blocked(ci, cj):
        if _conflict(ci, cj):
            return True
        ta, tb = {f.tid for f in ci}, {f.tid for f in cj}
        return any(frozenset((x, y)) in cannot for x in ta for y in tb)

    while True:
        live = [i for i, c in enumerate(clusters) if c]
        if max_clusters and len(live) <= max_clusters:
            break
        cents = {i: _centroid(clusters[i]) for i in live}
        best, pair = -1.0, None
        for a_i, i in enumerate(live):
            for j in live[a_i + 1:]:
                s = float(cents[i] @ cents[j]) + CONTINUITY_WEIGHT * _continuity(clusters[i], clusters[j])
                if s > best and not blocked(clusters[i], clusters[j]):
                    best, pair = s, (i, j)
        if pair is None or (max_clusters is None and best < threshold):
            break
        merge(*pair)

    out = [sorted(f.tid for f in c) for c in clusters if c]
    # biggest (most observed) chickens first
    return sorted(out, key=lambda c: -sum(frags[t].n for t in c))


def assignment_confidence(frags: dict[int, Frag], clusters: list[list[int]]) -> dict[int, dict]:
    """How sure is each fragment's cluster vs the best alternative it could belong to."""
    cents = [_centroid([frags[t] for t in c]) for c in clusters]
    res = {}
    for ci, c in enumerate(clusters):
        for t in c:
            f = frags[t]
            own = [frags[x] for x in c if x != t]
            own_sim = float(_centroid(own) @ f.emb) if own else 1.0
            alts = []
            for cj, d in enumerate(clusters):
                if cj == ci or _conflict([f], [frags[x] for x in d]):
                    continue
                alts.append((float(cents[cj] @ f.emb), cj))
            alt_sim, alt = max(alts) if alts else (0.0, None)
            res[t] = {"own_sim": round(own_sim, 3), "alt_sim": round(alt_sim, 3),
                      "alt_cluster": alt, "margin": round(own_sim - alt_sim, 3)}
    return res


# ------------------------------------------------------------------------- state

class IdentityStore:
    """identity.json: clusters (chickens), per-fragment status, human answers.

    Every change can be preceded by ``snapshot(label)``; ``undo()`` restores the latest one.
    Snapshots live in identity/history/ (the newest HISTORY_LIMIT are kept).
    """

    HISTORY_LIMIT = 200

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "identity"
        self.path = self.root / "identity.json"
        self._lock = threading.Lock()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    # ---------------------------------------------------------------- history
    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    def _snapshots(self) -> list[Path]:
        return sorted(self.history_dir.glob("*.json")) if self.history_dir.exists() else []

    def snapshot(self, label: str, with_features: bool = False) -> None:
        """Save the current state before a change. Cuts also change crops.csv/features.npy."""
        import datetime
        import shutil as _sh
        self.history_dir.mkdir(parents=True, exist_ok=True)
        snaps = self._snapshots()
        seq = int(snaps[-1].stem) + 1 if snaps else 1
        base = self.history_dir / f"{seq:06d}"
        if with_features:
            _sh.copy2(self.root / "crops.csv", f"{base}_crops.csv")
            _sh.copy2(self.root / "features.npy", f"{base}_features.npy")
        Path(f"{base}.json").write_text(json.dumps({
            "label": label, "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "with_features": with_features, "data": self.data}))
        for old in snaps[:max(0, len(snaps) + 1 - self.HISTORY_LIMIT)]:
            for p in self.history_dir.glob(f"{old.stem}*"):
                p.unlink()

    def history(self) -> dict:
        snaps = self._snapshots()
        last = json.loads(snaps[-1].read_text())["label"] if snaps else None
        return {"count": len(snaps), "label": last}

    def undo(self) -> str | None:
        """Restore the state saved before the most recent change; returns its label."""
        import shutil as _sh
        snaps = self._snapshots()
        if not snaps:
            return None
        snap = snaps[-1]
        rec = json.loads(snap.read_text())
        if rec.get("with_features"):
            _sh.move(f"{self.history_dir / snap.stem}_crops.csv", self.root / "crops.csv")
            _sh.move(f"{self.history_dir / snap.stem}_features.npy", self.root / "features.npy")
        self.data = rec["data"]
        self.save()
        snap.unlink()
        return rec["label"]

    def save(self):
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=1)
            os.replace(tmp, self.path)


def build(run_dir: Path, threshold: float = 0.80, flock_size: int | None = None,
          device: str | None = None, video: str | None = None) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    video = video or meta["video"]["path"]
    det = pd.read_csv(run_dir / "detections.csv")
    tracks = pd.read_csv(run_dir / "tracks.csv")
    tids = tracks.track_id.astype(int).tolist()
    det = det[det.track_id.isin(tids)].reset_index(drop=True)

    store = IdentityStore(run_dir)
    store.root.mkdir(parents=True, exist_ok=True)
    crops_dir = store.root / "crops"

    print(f"Selecting crops for {len(tids)} fragments ...")
    picks = select_crops(det, tids)
    print(f"Extracting {len(picks)} crops from {Path(video).name} ...")
    picks["crop"] = extract_crops(video, picks, crops_dir)
    picks = picks[picks.crop != ""].reset_index(drop=True)
    print("Embedding appearance ...")
    feats = embed_crops([crops_dir / c for c in picks.crop], device=device)
    np.save(store.root / "features.npy", feats)
    picks.to_csv(store.root / "crops.csv", index=False)

    frags = fragments(det, picks, feats)
    clusters = cluster(frags, threshold=threshold, max_clusters=flock_size)
    store.data = _fresh_state(frags, clusters, picks, video, meta)
    store.save()
    n_q = len(questions(store.data))
    print(f"{len(frags)} fragments -> {len(clusters)} candidate chickens; {n_q} questions for you.")
    return store.data


def _fresh_state(frags, clusters, picks, video, meta, names=None, answers=None) -> dict:
    conf = assignment_confidence(frags, clusters)
    names = names or {}
    return {
        "video": video,
        "fps": meta["video"]["fps"],
        "chickens": [{"id": i, "name": names.get(str(i), ""), "tracks": c} for i, c in enumerate(clusters)],
        "fragments": {
            str(t): {"start": f.start, "end": f.end, "n": f.n,
                     "crops": picks[picks.track_id == t].crop.tolist(),
                     **conf[t]}
            for t, f in frags.items()
        },
        "answers": answers or [],   # [{"track": t, "chicken": id | None, "verdict": "yes"|"no"|"not_chicken"}]
    }


def chicken_of(data: dict) -> dict[int, int]:
    return {t: c["id"] for c in data["chickens"] for t in c["tracks"]}


# Pieces excluded from every chicken, by reason. Both lists are kept so the detector
# can be retrained on them: "multiple" = one box around 2+ birds (should be split),
# "not_chicken" = false detection (should be background).
EXCLUDED = ("not_chicken", "multiple")
EXCLUDED_KEYS = {"not_chicken": "rejected", "multiple": "multiple"}


def excluded(data: dict) -> set[int]:
    return set(data.get("rejected", [])) | set(data.get("multiple", []))


# Box-quality flags: the piece IS that chicken, but its box is off. It still counts for the
# chicken, is never a conflict with the chicken's proper box, and is listed for retraining.
BOX_FLAGS = ("partial", "loose")  # only part of the bird / bird plus a bit of a neighbour


def confirmed(data: dict) -> set[int]:
    return {a["track"] for a in data["answers"] if a["verdict"] in ("yes", "moved", *BOX_FLAGS, *EXCLUDED)}


def box_flag(data: dict, track: int) -> str | None:
    return next((f for f in BOX_FLAGS if track in data.get(f, [])), None)


def partial(data: dict) -> set[int]:
    """Pieces with an imperfect box (any BOX_FLAGS): only part of the bird, or a bit too much."""
    return {t for f in BOX_FLAGS for t in data.get(f, [])}


def questions(data: dict, max_margin: float = 0.03, min_sim: float = 0.75,
              min_duration_s: float = 2.0) -> list[dict]:
    """Doubtful, unanswered fragments, longest (most valuable) first."""
    done = confirmed(data) | excluded(data)
    asked_no = {(a["track"], a["chicken"]) for a in data["answers"] if a["verdict"] == "no"}
    owner = chicken_of(data)
    qs = []
    for t, f in data["fragments"].items():
        t = int(t)
        if t in done or t not in owner or f["end"] - f["start"] < min_duration_s:
            continue
        doubtful = f["margin"] < max_margin or f["own_sim"] < min_sim
        if not doubtful:
            continue
        # ask about the current owner first; once rejected, about the best alternative
        for cand in (owner[t], f["alt_cluster"]):
            if cand is not None and (t, cand) not in asked_no and any(c["id"] == cand for c in data["chickens"]):
                qs.append({"track": t, "chicken": cand, "duration_s": round(f["end"] - f["start"], 1)})
                break
    return sorted(qs, key=lambda q: -q["duration_s"])


def conflicts(data: dict, tolerance_s: float = CONFLICT_MIN_OVERLAP_S) -> list[dict]:
    """Pairs of pieces given the same chicken but visible at the same time (impossible)."""
    fr = data["fragments"]
    part = partial(data)  # a partial box next to the full box is one bird boxed twice, not a conflict
    out = []
    for c in data["chickens"]:
        ts = sorted((t for t in c["tracks"] if t not in part), key=lambda t: fr[str(t)]["start"])
        for i, a in enumerate(ts):
            fa = fr[str(a)]
            for b in ts[i + 1:]:
                fb = fr[str(b)]
                if fb["start"] >= fa["end"]:
                    break
                ov = min(fa["end"], fb["end"]) - fb["start"]
                if ov > tolerance_s:
                    out.append({"chicken": c["id"], "a": a, "b": b, "overlap_s": round(ov, 1)})
    return sorted(out, key=lambda x: -x["overlap_s"])


COLOR_A, COLOR_B = (255, 220, 0), (0, 220, 255)  # BGR: cyan, yellow


def overlap_snapshot(video: str, det: pd.DataFrame, a: int, b: int, k: int = 1, n: int = 3,
                     fps: float = 25.0, margin: int = 140, width: int = 520) -> bytes | None:
    """JPEG of the real frame where pieces a and b are both visible, both boxes drawn.

    k picks one of n moments spread across their overlap (0 = start ... n-1 = end).
    """
    da = det[det.track_id == a].drop_duplicates("frame").set_index("frame")
    db = det[det.track_id == b].drop_duplicates("frame").set_index("frame")
    common = sorted(set(da.index) & set(db.index))
    if not common:
        return None
    f = common[min(len(common) - 1, int((k + 0.5) / n * len(common)))]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    H, W = img.shape[:2]
    ra, rb = da.loc[f], db.loc[f]
    x1 = int(max(0, min(ra.x1, rb.x1) - margin)); y1 = int(max(0, min(ra.y1, rb.y1) - margin))
    x2 = int(min(W, max(ra.x2, rb.x2) + margin)); y2 = int(min(H, max(ra.y2, rb.y2) + margin))
    for r, col, lab in ((ra, COLOR_A, "A"), (rb, COLOR_B, "B")):
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(img, p1, p2, (0, 0, 0), 6)
        cv2.rectangle(img, p1, p2, col, 3)
        cv2.putText(img, lab, (p1[0] + 4, p1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, lab, (p1[0] + 4, p1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    crop = img[y1:y2, x1:x2]
    out = cv2.resize(crop, (width, int(crop.shape[0] * width / crop.shape[1])))
    t = int(f / fps)
    label = f"{t // 60:02d}:{t % 60:02d}"
    cv2.putText(out, label, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def rename(data: dict, chicken: int, name: str) -> None:
    for c in data["chickens"]:
        if c["id"] == chicken:
            c["name"] = name.strip()[:40]


def merge_chickens(data: dict, src: int, dst: int) -> None:
    s = next(c for c in data["chickens"] if c["id"] == src)
    d = next(c for c in data["chickens"] if c["id"] == dst)
    d["tracks"] = sorted(d["tracks"] + s["tracks"])
    if not d["name"]:
        d["name"] = s["name"]
    data["chickens"] = [c for c in data["chickens"] if c["id"] != src]
    data.setdefault("merges", []).append([src, dst])


# --------------------------------------------------------------------- cutting

SPLIT_ID_BASE = 100_000  # ids of pieces created by cutting start here


def apply_splits(det: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """Re-label detections according to cuts: piece `track` from `at_frame` on becomes `new`."""
    if not splits:
        return det
    det = det.copy()
    for s in splits:  # in order, so a piece created by a cut can itself be cut
        m = (det.track_id == s["track"]) & (det.frame >= s["at_frame"])
        det.loc[m, "track_id"] = s["new"]
    return det


def load_detections(run_dir: Path, data: dict | None = None) -> pd.DataFrame:
    det = pd.read_csv(Path(run_dir) / "detections.csv")
    if data is None:
        data = IdentityStore(run_dir).data
    return apply_splits(det, data.get("splits", []))


def register_split(data: dict, det: pd.DataFrame, track: int, at_frame: int) -> tuple[int, pd.DataFrame]:
    """Record a cut in `data` (membership, exclusions, answers) and return (new id, new det)."""
    rows = det[det.track_id == track]
    if not ((rows.frame < at_frame).any() and (rows.frame >= at_frame).any()):
        raise ValueError("the cut must fall inside the piece")
    used = {int(t) for t in data["fragments"]} | {s["new"] for s in data.get("splits", [])}
    nid = max(SPLIT_ID_BASE, max(used) + 1)
    split = {"track": int(track), "at_frame": int(at_frame), "new": nid}
    data.setdefault("splits", []).append(split)
    det = apply_splits(det, [split])
    for c in data["chickens"]:          # both halves start in the same chicken ...
        if track in c["tracks"]:
            c["tracks"] = sorted(c["tracks"] + [nid])
    for key in (*EXCLUDED_KEYS.values(), *BOX_FLAGS):  # ... or the same pile / box flag
        if track in data.get(key, []):
            data[key].append(nid)
    # earlier answers about the whole piece may now be true for only one half
    data["answers"] = [a for a in data["answers"] if a["track"] != track]
    for t in (track, nid):
        r = det[det.track_id == t]
        data["fragments"][str(t)] = {"start": float(r.time_s.min()), "end": float(r.time_s.max()),
                                     "n": int(len(r)), "crops": [], "margin": 0.0, "own_sim": 1.0,
                                     "alt_cluster": None}
    return nid, det


def split_piece(run_dir: Path, data: dict, det: pd.DataFrame, track: int, at_frame: int,
                device: str | None = None) -> pd.DataFrame:
    """Cut a piece in two and give both halves fresh crops + appearance features."""
    root = Path(run_dir) / "identity"
    nid, det = register_split(data, det, track, at_frame)
    picks = pd.read_csv(root / "crops.csv")
    feats = np.load(root / "features.npy")
    keep = (picks.track_id != track).to_numpy()
    picks, feats = picks[keep].reset_index(drop=True), feats[keep]
    new = select_crops(det, [track, nid])
    new["crop"] = extract_crops(data["video"], new, root / "crops")
    new = new[new.crop != ""].reset_index(drop=True)
    new_feats = embed_crops([root / "crops" / c for c in new.crop], device=device)
    pd.concat([picks, new[picks.columns]], ignore_index=True).to_csv(root / "crops.csv", index=False)
    np.save(root / "features.npy", np.vstack([feats, new_feats]))
    for t in (track, nid):
        data["fragments"][str(t)]["crops"] = new[new.track_id == t].crop.tolist()
    return det


def overlap_cuts(det: pd.DataFrame, track: int, other: int) -> list[int]:
    """Cut frames that isolate the part of `track` overlapping `other` in time.

    Returns 0-2 frames: where the overlap starts (if `track` began earlier) and the first
    frame after it (if `track` continues later). Cutting there leaves the overlap as its own piece.
    """
    a = np.sort(det.loc[det.track_id == track, "frame"].to_numpy())
    b = det.loc[det.track_id == other, "frame"]
    if not len(a) or not len(b):
        return []
    lo, hi = int(b.min()), int(b.max())
    inside = a[(a >= lo) & (a <= hi)]
    if not len(inside):
        return []
    cuts = []
    if a[0] < inside[0]:
        cuts.append(int(inside[0]))
    after = a[a > inside[-1]]
    if len(after):
        cuts.append(int(after[0]))
    return cuts


def strip_frames(det: pd.DataFrame, track: int, n: int = 12) -> list[int]:
    """n frames spread evenly over the piece's whole life (for filmstrips)."""
    frames = np.sort(det.loc[det.track_id == track, "frame"].to_numpy())
    if not len(frames):
        return []
    idx = np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int))
    return [int(frames[i]) for i in idx]


def piece_frame(video: str, det: pd.DataFrame, track: int, frame: int, fps: float,
                margin: int = 90, height: int = 170) -> bytes | None:
    """JPEG of the piece's box at one frame, with surroundings, for filmstrips."""
    r = det[(det.track_id == track) & (det.frame == frame)]
    if r.empty:
        return None
    r = r.iloc[0]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    H, W = img.shape[:2]
    p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
    cv2.rectangle(img, p1, p2, (0, 0, 0), 5)
    cv2.rectangle(img, p1, p2, (255, 255, 255), 2)
    crop = img[max(0, p1[1] - margin):min(H, p2[1] + margin), max(0, p1[0] - margin):min(W, p2[0] + margin)]
    out = cv2.resize(crop, (int(crop.shape[1] * height / crop.shape[0]), height))
    t = int(frame / fps)
    lab = f"{t // 60:02d}:{t % 60:02d}"
    cv2.putText(out, lab, (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, lab, (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def load_fragments(run_dir: Path) -> dict[int, Frag]:
    root = Path(run_dir) / "identity"
    det = load_detections(run_dir)
    picks = pd.read_csv(root / "crops.csv")
    feats = np.load(root / "features.npy")
    det = det[det.track_id.isin(picks.track_id.unique())]
    return fragments(det, picks, feats)


def resort(data: dict, frags: dict[int, Frag]) -> int:
    """Re-assign every unconfirmed fragment to the most similar compatible chicken.

    Chickens are represented by their confirmed fragments (or all, if none confirmed yet).
    Returns how many fragments changed chicken.
    """
    sure = confirmed(data)
    rejected = excluded(data)
    anchors = {}
    for c in data["chickens"]:
        conf_t = [t for t in c["tracks"] if t in sure]
        anchors[c["id"]] = conf_t or list(c["tracks"])
    owner = chicken_of(data)
    moved = 0
    # longest fragments first so they claim time slots before short noisy ones
    for t in sorted((t for t in owner if t not in sure and t not in rejected and t in frags),
                    key=lambda t: -frags[t].n):
        f = frags[t]
        best, best_c = -1.0, owner[t]
        for cid, members in anchors.items():
            others = [frags[m] for m in members if m != t and m in frags]
            if not others:
                continue
            busy = [frags[m] for m in next(c for c in data["chickens"] if c["id"] == cid)["tracks"]
                    if m != t and m in frags]
            if _conflict([f], busy):
                continue
            sim = float(_centroid(others) @ f.emb) + CONTINUITY_WEIGHT * _continuity([f], others)
            if sim > best:
                best, best_c = sim, cid
        if best_c != owner[t]:
            for c in data["chickens"]:
                if t in c["tracks"]:
                    c["tracks"].remove(t)
                if c["id"] == best_c:
                    c["tracks"] = sorted(c["tracks"] + [t])
            owner[t] = best_c
            moved += 1
    refresh_confidence(data, frags)
    return moved


def refresh_confidence(data: dict, frags: dict[int, Frag]) -> None:
    clusters = [[t for t in c["tracks"] if t in frags] for c in data["chickens"]]
    ids = [c["id"] for c in data["chickens"]]
    keep = [i for i, c in enumerate(clusters) if c]
    conf = assignment_confidence(frags, [clusters[i] for i in keep])
    for t, v in conf.items():
        alt = v["alt_cluster"]
        v["alt_cluster"] = ids[keep[alt]] if alt is not None else None
        data["fragments"][str(t)].update(v)


def apply_answer(data: dict, track: int, chicken: int | None, verdict: str) -> None:
    """Record a human answer and move the fragment accordingly."""
    data["answers"].append({"track": track, "chicken": chicken, "verdict": verdict})
    if verdict == "no":
        return
    for c in data["chickens"]:
        if track in c["tracks"]:
            c["tracks"].remove(track)
    if verdict in ("yes", "moved", *BOX_FLAGS) and chicken is not None:
        target = next(c for c in data["chickens"] if c["id"] == chicken)
        target["tracks"].append(track)
        target["tracks"].sort()
    elif verdict == "new":
        new_id = max(c["id"] for c in data["chickens"]) + 1
        data["chickens"].append({"id": new_id, "name": "", "tracks": [track]})
        data["answers"][-1]["verdict"] = "moved"
        data["answers"][-1]["chicken"] = new_id
    for key in EXCLUDED_KEYS.values():  # a later answer overrides an earlier exclusion
        if track in data.get(key, []):
            data[key].remove(track)
    if verdict in EXCLUDED:
        data.setdefault(EXCLUDED_KEYS[verdict], []).append(track)
    for flag in BOX_FLAGS:
        if track in data.get(flag, []):
            data[flag].remove(track)
    if verdict in BOX_FLAGS:
        data.setdefault(verdict, []).append(track)
    data["chickens"] = [c for c in data["chickens"] if c["tracks"] or c["name"]]


# ------------------------------------------------------------------------- export

def export(run_dir: Path, max_hidden_s: float = 120.0, hidden_tau_s: float = 20.0) -> dict:
    """Write per-chicken timeline (visible + inferred hidden segments) and summary."""
    run_dir = Path(run_dir)
    store = IdentityStore(run_dir)
    data = store.data
    det = load_detections(run_dir, data)
    fps = data["fps"]
    rejected = excluded(data)

    part = partial(data)
    owner = {t: (c["name"] or f"Chicken {c['id'] + 1}") for c in data["chickens"] for t in c["tracks"]
             if t not in rejected}
    # one row per chicken per frame: prefer the full box over a partial one, then the most confident
    per_chicken = (det.assign(chicken=det.track_id.map(owner), partial=det.track_id.isin(part))
                   .dropna(subset=["chicken"])
                   .sort_values(["chicken", "frame", "partial", "conf"], ascending=[True, True, True, False])
                   .drop_duplicates(["chicken", "frame"]))

    segs, rows = [], []
    for c in data["chickens"]:
        name = c["name"] or f"Chicken {c['id'] + 1}"
        tracks = [t for t in c["tracks"] if t not in rejected]
        frags = sorted(((data["fragments"][str(t)]["start"], data["fragments"][str(t)]["end"], t)
                        for t in tracks))
        prev = None
        for start, end, t in frags:
            if prev is not None:
                gap = start - prev[1]
                if 0 < gap <= max_hidden_s:
                    segs.append({"chicken": name, "state": "hidden", "start_s": prev[1], "end_s": start,
                                 "duration_s": gap, "track_before": prev[2], "track_after": t,
                                 # confidence decays with how long the bird was out of view
                                 "confidence": round(float(np.exp(-gap / hidden_tau_s)), 3)})
            segs.append({"chicken": name, "state": "visible", "start_s": start, "end_s": end,
                         "duration_s": end - start + 1 / fps, "track_before": t, "track_after": t,
                         "confidence": 1.0})
            prev = (start, end, t)
        d = per_chicken[per_chicken.chicken == name]
        if len(d):
            step = np.hypot(d.cx.diff(), d.cy.diff()).where(d.track_id.eq(d.track_id.shift()), 0)
            rows.append({"chicken": name, "fragments": len(tracks),
                         "first_seen_s": d.time_s.min(), "last_seen_s": d.time_s.max(),
                         "visible_s": len(d) / fps,
                         "distance_px": float(step.fillna(0).sum())})
    timeline = pd.DataFrame(segs).sort_values(["chicken", "start_s"]) if segs else pd.DataFrame()
    summary = pd.DataFrame(rows)
    if len(timeline):
        hid = timeline[timeline.state == "hidden"].groupby("chicken").duration_s.sum()
        summary["hidden_s"] = summary.chicken.map(hid).fillna(0.0)
    per_chicken.sort_values("frame").to_csv(run_dir / "detections_named.csv", index=False, float_format="%.2f")
    timeline.to_csv(run_dir / "chicken_timeline.csv", index=False, float_format="%.2f")
    summary.to_csv(run_dir / "chickens.csv", index=False, float_format="%.2f")
    open_conflicts = conflicts(data)
    flagged = [{"track_id": t, "reason": reason, "chicken": ""} for reason, key in EXCLUDED_KEYS.items()
               for t in data.get(key, [])]
    flagged += [{"track_id": t, "reason": f, "chicken": owner.get(t, "")}
                for f in BOX_FLAGS for t in sorted(data.get(f, []))]
    pd.DataFrame(flagged, columns=["track_id", "reason", "chicken"]).to_csv(run_dir / "flagged_pieces.csv", index=False)
    return {"chickens": len(summary), "timeline_segments": len(timeline),
            "not_chicken": len(data.get("rejected", [])), "multiple": len(data.get("multiple", [])),
            "conflicts": len(open_conflicts)}
