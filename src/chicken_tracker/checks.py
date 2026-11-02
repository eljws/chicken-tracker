"""Automatic audit of the naming work: likely mistakes the conflict/question logic can't see.

Conflicts only catch one name in two places; questions only catch doubtful *unconfirmed*
pieces. These checks look for the other ways things go wrong, each with a suggested fix:

* ``aside_single``  a piece set aside ("several chickens" / "not a chicken") whose box is
                    the size of ONE bird for a long stretch -> cut that stretch out and
                    give it to the chicken it looks like.
* ``merged_inside`` a named piece whose box grows to two birds' size for a while ->
                    cut that stretch out and set it aside as "several chickens".
* ``swap``          a named piece whose look changes part-way (tracker jumped to another
                    bird) -> cut where it changes; both halves are then re-checked.

Box size is judged against the typical single-chicken box *at that height in the image*
(birds near the camera look bigger), learned from the named detections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SINGLE_MAX = 1.35   # box up to 1.35x a typical single bird = one bird
MERGED_MIN = 1.7    # box over 1.7x = two birds in one box
SMOOTH = 15         # frames (~1 s) of rolling median on the size ratio


def size_model(named: pd.DataFrame) -> np.ndarray:
    """log(box area) ~ a + b*y2 + c*y2^2, fitted on named single-chicken boxes."""
    area = (named.x2 - named.x1) * (named.y2 - named.y1)
    y = named.y2.to_numpy()
    deg = 2 if np.ptp(y) > 50 and len(np.unique(y)) > 3 else 0  # no height spread: constant size
    return np.polyfit(y, np.log(area.to_numpy()), deg)


def size_ratio(rows: pd.DataFrame, model: np.ndarray) -> np.ndarray:
    area = ((rows.x2 - rows.x1) * (rows.y2 - rows.y1)).to_numpy()
    expected = np.exp(np.polyval(model, rows.y2.to_numpy()))
    r = pd.Series(area / expected).rolling(SMOOTH, center=True, min_periods=1).median()
    return r.to_numpy()


def _runs(mask: np.ndarray, frames: np.ndarray, fps: float, min_s: float) -> list[tuple[int, int]]:
    """(first_frame, last_frame) of runs where mask is True for at least min_s."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (frames[i - 1] - frames[start] + 1) / fps >= min_s:
                out.append((int(frames[start]), int(frames[i - 1])))
            start = None
    return out


def _best_chicken(feats: np.ndarray | None, centroids: dict[int, np.ndarray], names: dict[int, str]):
    if feats is None or not len(feats) or not centroids:
        return None, None
    e = feats.mean(0)
    e = e / (np.linalg.norm(e) + 1e-9)
    sims = {cid: float(c @ e) for cid, c in centroids.items()}
    cid = max(sims, key=sims.get)
    return cid, round(sims[cid], 2)


def run_checks(data: dict, det: pd.DataFrame, named: pd.DataFrame, picks: pd.DataFrame,
               feats: np.ndarray, frags: dict, fps: float) -> list[dict]:
    from .identity import _centroid, confirmed  # local: avoid import cycle

    dismissed = set(data.get("dismissed_checks", []))
    names = {c["id"]: c["name"] or f"Chicken {c['id'] + 1}" for c in data["chickens"]}
    owner = {t: c["id"] for c in data["chickens"] for t in c["tracks"]}
    sure = confirmed(data)
    centroids = {}
    for c in data["chickens"]:
        ts = [t for t in c["tracks"] if t in frags]
        ref = [t for t in ts if t in sure] or ts
        if ref:
            centroids[c["id"]] = _centroid([frags[t] for t in ref])
    model = size_model(named)
    feats_of = {t: (picks.frame.to_numpy()[i], feats[i]) for t, i in
                ((t, np.flatnonzero(picks.track_id.to_numpy() == t)) for t in picks.track_id.unique())}

    def seg_feats(t, f0, f1):
        if t not in feats_of:
            return None
        fr, ft = feats_of[t]
        sel = (fr >= f0) & (fr <= f1)
        return ft[sel] if sel.any() else None

    clock = lambda f: f"{int(f / fps // 60)}:{int(f / fps % 60):02d}"
    out = []
    aside = {t: "several chickens" for t in data.get("multiple", [])} | \
            {t: "not a chicken" for t in data.get("rejected", [])}

    for t, g in det.groupby("track_id"):
        t = int(t)
        g = g.sort_values("frame")
        if len(g) < SMOOTH:
            continue
        frames = g.frame.to_numpy()
        ratio = size_ratio(g, model)

        if t in aside:   # 1. set aside, but single-bird sized for a long stretch
            for f0, f1 in _runs(ratio <= SINGLE_MAX, frames, fps, min_s=5.0):
                cid, sim = _best_chicken(seg_feats(t, f0, f1), centroids, names)
                key = f"aside_single:{t}:{f0}:{f1}"
                if key in dismissed:
                    continue
                looks = f", and it looks like {names[cid]}" if cid is not None and sim and sim >= 0.85 else ""
                out.append({"key": key, "kind": "aside_single", "track": t, "from_frame": f0, "to_frame": f1,
                            "seconds": round((f1 - f0 + 1) / fps, 1), "chicken": cid if looks else None,
                            "title": f"Piece #{t} is set aside as \"{aside[t]}\", but {clock(f0)}–{clock(f1)} "
                                     f"its box is the size of one bird{looks}.",
                            "fix": f"Cut {clock(f0)}–{clock(f1)} out" +
                                   (f" and give it to {names[cid]}" if looks else " so it can be named")})
        elif t in owner:  # 2. merged box inside a named piece
            for f0, f1 in _runs(ratio >= MERGED_MIN, frames, fps, min_s=3.0):
                key = f"merged_inside:{t}:{f0}:{f1}"
                if key in dismissed:
                    continue
                out.append({"key": key, "kind": "merged_inside", "track": t, "from_frame": f0, "to_frame": f1,
                            "seconds": round((f1 - f0 + 1) / fps, 1), "chicken": owner[t],
                            "title": f"{names[owner[t]]}'s piece #{t}: {clock(f0)}–{clock(f1)} the box is about "
                                     f"{np.median(ratio[(frames >= f0) & (frames <= f1)]):.1f}× one bird — "
                                     f"probably two chickens in one box.",
                            "fix": f"Cut {clock(f0)}–{clock(f1)} out as \"several chickens\""})
            # 3. look changes part-way: compare consecutive crops along the piece
            if t in feats_of and len(feats_of[t][0]) >= 4:
                fr, ft = feats_of[t]
                order = np.argsort(fr)
                fr, ft = fr[order], ft[order]
                sims = np.einsum("ij,ij->i", ft[:-1], ft[1:])
                k = int(np.argmin(sims))
                before, after = ft[:k + 1].mean(0), ft[k + 1:].mean(0)
                split_sim = float(before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-9))
                if split_sim < 0.80:
                    cut = int((fr[k] + fr[k + 1]) // 2)
                    key = f"swap:{t}:{cut}"
                    if key in dismissed:
                        continue
                    a, _ = _best_chicken(ft[:k + 1], centroids, names)
                    b, _ = _best_chicken(ft[k + 1:], centroids, names)
                    who = f" (looks like {names[a]} before, {names[b]} after)" if a is not None and a != b else ""
                    out.append({"key": key, "kind": "swap", "track": t, "from_frame": cut, "to_frame": cut,
                                "seconds": round(min(cut - frames[0], frames[-1] - cut) / fps, 1),
                                "chicken": owner[t],
                                "title": f"{names[owner[t]]}'s piece #{t} looks different after ~{clock(cut)}"
                                         f"{who} — the tracker may have switched birds.",
                                "fix": f"Cut the piece at {clock(cut)}; both halves get re-checked"})
    return sorted(out, key=lambda c: -c["seconds"])


def apply_check(check: dict, data: dict, det: pd.DataFrame, run_dir, split_piece) -> tuple[pd.DataFrame, list[int]]:
    """Carry out a check's suggested fix.

    Returns (updated detections, pieces that still need a chicken - the caller places them
    once their fresh appearance features exist).
    """
    from .identity import apply_answer

    t, f0, f1 = check["track"], check["from_frame"], check["to_frame"]
    frames = np.sort(det.loc[det.track_id == t, "frame"].to_numpy())
    if check["kind"] == "swap":
        return split_piece(run_dir, data, det, t, int(frames[frames >= f0][0])), []
    # isolate [f0, f1] of piece t: cut at the frame after f1 first, then at f0
    after = frames[frames > f1]
    if len(after):
        det = split_piece(run_dir, data, det, t, int(after[0]))
    piece = t
    if frames[0] < f0:
        det = split_piece(run_dir, data, det, t, int(f0))
        piece = data["splits"][-1]["new"]
    if check["kind"] == "aside_single":
        if check.get("chicken") is not None:
            apply_answer(data, piece, check["chicken"], "moved")
            return det, []
        for key in ("multiple", "rejected"):  # unknown bird: out of the pile, caller places it
            if piece in data.get(key, []):
                data[key].remove(piece)
        return det, [piece]
    apply_answer(data, piece, None, "multiple")  # merged_inside
    return det, []
