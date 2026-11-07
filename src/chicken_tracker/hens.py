"""The Hens page: every named piece of every clip in one place, per hen, to look her over as a whole.

One picture per piece (its middle crop), grouped however you like on the page (by day and clip, time of
day, how sure, box, or how unlike her it looks). Nothing here decides anything: "looks unlike her" only
sorts, and every change is a button you press, sent to that clip's own naming page.

"Unlike her": each piece's appearance vector (the mean of its crops' features.npy rows) against the
mean of every *confirmed* piece of that hen across all clips; 0 = typical of her, higher = less like
her. If another hen's mean is closer, the piece says so ("closer to Salt").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import identity as I
from .metrics import HENS, clip_start_s

_cache: dict[str, tuple] = {}
ASIDE = {"multiple": "several", "rejected": "not a chicken", "unsure": "can't tell"}


def _stamp(run: Path) -> tuple:
    files = ("identity/identity.json", "identity/features.npy", "identity/crops.csv", "identity/recognizer.csv")
    return tuple(round((run / f).stat().st_mtime, 3) if (run / f).exists() else 0 for f in files)


def _run_pieces(root: Path, run: Path) -> tuple[list[dict], dict[int, np.ndarray], dict]:
    """(pieces, {index in pieces: mean feature}, {hen name: chicken id}) for one clip, cached."""
    key, stamp = str(run), _stamp(run)
    if key in _cache and _cache[key][0] == stamp:
        return _cache[key][1]
    d = json.loads((run / "identity" / "identity.json").read_text())
    fr = d["fragments"]
    rel = str(run.relative_to(root))
    start = clip_start_s(run) or 0
    sure = I.confirmed(d)
    feats, row_of = None, {}
    if (run / "identity" / "features.npy").exists() and (run / "identity" / "crops.csv").exists():
        crops = pd.read_csv(run / "identity" / "crops.csv")
        f = np.load(run / "identity" / "features.npy")
        if len(f) == len(crops):
            feats = f
            for i, (t, c) in enumerate(zip(crops.track_id, crops.crop)):
                row_of.setdefault(int(t), []).append(i)
    rec = {}
    if (run / "identity" / "recognizer.csv").exists():
        r = pd.read_csv(run / "identity" / "recognizer.csv")
        cols = [c for c in r.columns if c.startswith("p_")]
        if cols:
            m = r.groupby("track_id")[cols].mean()
            for t, row in m.iterrows():
                c = row.idxmax()
                rec[int(t)] = (c[2:], round(float(row[c]), 2))
    names = {c["id"]: (c["name"] or f"Chicken {c['id'] + 1}") for c in d["chickens"]}
    owner = {t: c["id"] for c in d["chickens"] for t in c["tracks"]}
    aside = {t: label for k, label in ASIDE.items() for t in d.get(k, [])}
    pieces, vec = [], {}
    for t_s, f in fr.items():
        t = int(t_s)
        if t not in owner and t not in aside:
            continue
        crops = f.get("crops") or []
        p = {"run": rel, "track": t, "crop": crops[len(crops) // 2] if crops else "",
             "crops": crops[:: max(1, len(crops) // 4)][:4],
             "hen": names[owner[t]] if t in owner else None, "aside": aside.get(t),
             "chicken": owner.get(t), "clock_s": round(start + f["start"], 1), "t": round(f["start"], 1),
             "duration_s": round(f["end"] - f["start"], 1), "sure": t in sure, "flag": I.box_flag(d, t)}
        if t in rec:
            p["rec"], p["rec_p"] = rec[t]
        if feats is not None and row_of.get(t):
            v = feats[row_of[t]].mean(axis=0)
            vec[len(pieces)] = v / (np.linalg.norm(v) + 1e-9)
        pieces.append(p)
    ids = {n: i for i, n in names.items()}
    out = (pieces, vec, ids)
    _cache[key] = (stamp, out)
    return out


def hens_data(root: Path, runs: list[Path]) -> dict:
    root = Path(root)
    pieces, vecs, ids = [], [], {}
    for run in runs:
        if not (run / "identity" / "identity.json").exists():
            continue
        try:
            ps, vec, names = _run_pieces(root, run)
        except (ValueError, KeyError, OSError):  # half-written file: skip this clip this time
            continue
        rel = str(run.relative_to(root))
        ids[rel] = names
        for i, p in enumerate(ps):
            pieces.append(dict(p))
            vecs.append(vec.get(i))
    # each hen's typical look: mean of her confirmed pieces, over all clips
    cent = {}
    for h in {p["hen"] for p in pieces if p["hen"]}:
        vs = [v for p, v in zip(pieces, vecs) if p["hen"] == h and p["sure"] and v is not None]
        if len(vs) >= 3:
            m = np.mean(vs, axis=0)
            cent[h] = m / (np.linalg.norm(m) + 1e-9)
    for p, v in zip(pieces, vecs):
        if v is None or not cent:
            continue
        sims = {h: float(v @ c) for h, c in cent.items()}
        if p["hen"] in sims:
            p["odd"] = round(1 - sims[p["hen"]], 3)
            best = max(sims, key=sims.get)
            if best != p["hen"] and sims[best] - sims[p["hen"]] > 0.03:  # clearly closer, not a tie
                p["closer_to"] = best
    hens = [h for h in HENS if any(p["hen"] == h for p in pieces)]
    hens += sorted({p["hen"] for p in pieces if p["hen"] and p["hen"] not in hens})
    return {"hens": hens, "pieces": pieces, "ids": ids, "typical": sorted(cent)}


def crop_path(root: Path, run_rel: str, name: str) -> Path | None:
    if ".." in run_rel or not re.fullmatch(r"[\w.\-]+\.jpg", name):
        return None
    p = Path(root) / run_rel / "identity" / "crops" / name
    return p if p.is_file() else None
