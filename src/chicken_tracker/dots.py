"""One dot per hen in every frame, and the moves no hen can make.

Where a hen is detected and named, her dot sits at her feet (bottom middle of the box). Where the
detector loses her (behind something, sitting still in shade, too small), the dot follows the
*pixels*: points on her body in the last box are tracked with optical flow, frame to frame, so the
dot moves the way she moves. If those points are lost too, the dot stays where she was last seen
and is marked lost.

With every hen's dot known, some moves are impossible - each is a likely naming or tracking mistake:

* ``teleport``     her dot jumps between two frames faster than any hen runs (box jumped to another bird)
* ``two_places``   the same hen boxed twice, far apart, at the same moment (one of the names is wrong)
* ``gap_jump``     she vanishes and reappears further away than she could have walked in that time
* ``flow_split``   the pixels followed her one way, but "she" reappears somewhere else (likely a swap:
                   the dot usually ends up next to the hen that was really her)
* ``stacked``      two hens' dots on top of each other for a while (one bird named twice)

Results: dots.json (for the player) and illegal_moves.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibration import load_calibration, to_ground

FLOW_MAX_S = 30.0      # follow pixels at most this long after she's lost
FLOW_SCALE = 0.5       # optical flow on half-size frames
MIN_POINTS = 4         # fewer tracked points than this: lost
MAX_SPEED = 4.0        # m/s - no hen runs faster (a sprint is ~2-3 m/s)
WALK_SPEED = 2.0       # m/s - fastest plausible average across a gap
HEN_M = 0.35           # a hen's box diagonal in metres, used when the run has no calibration
TWO_PLACES_BODIES = 1.5
FLOW_SPLIT_BODIES = 2.0
MOVED_BODIES = 0.5     # followed dot counts as moving with her once it's this far from where she was lost
STACK_BODIES = 0.25
MIN_EVENT_S = 0.5
VERSION = 1


def _feet(r) -> np.ndarray:
    return np.array([(r.x1 + r.x2) / 2, r.y2], float)


class _Metric:
    """Distances in metres (ground plane) when calibrated, else in hen body lengths x HEN_M."""

    def __init__(self, H, body_px: float):
        self.H, self.body = H, max(body_px, 1.0)

    def m(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.H is not None:
            g = to_ground(self.H, [a[0], b[0]], [a[1], b[1]])
            return float(np.linalg.norm(g[0] - g[1]))
        return float(np.linalg.norm(a - b)) / self.body * HEN_M


def _seen(named: pd.DataFrame, hen: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per frame: her best box, all her full boxes) - full boxes beat partial ones."""
    d = named[named.chicken == hen].copy()
    d["partial"] = d.partial.astype(bool) if "partial" in d else False
    best = d.sort_values(["frame", "partial", "conf"], ascending=[True, True, False]).drop_duplicates("frame")
    return best.set_index("frame"), d[~d.partial]


def _runs(frames: list[int], fps: float, min_s: float, join: int = 3) -> list[tuple[int, int]]:
    out = []
    for f in sorted(frames):
        if out and f - out[-1][1] <= join:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [(a, b) for a, b in out if (b - a + 1) / fps >= min_s]


# ------------------------------------------------------------------ following pixels
#
# Two kinds of pixel-following, both independent of the names (so they are worked out once per clip
# and saved in identity/flow.json; renaming never needs the video again):
#   "after":  from every place a piece stops being seen (its last frame, or a gap inside it), follow
#             the bird's pixels for up to FLOW_MAX_S - this is where her dot goes while she's lost.
#   "window": every WIN_S along every piece, follow the pixels in the box for WIN_S and note where
#             they end up - if the box ends up somewhere else, the box left the bird (switched birds).

WIN_S = 1.0
RESEED_BELOW = 12      # "after": top up the points from the dot's box when fewer than this remain
SPLIT_BODIES = 1.5     # "window": box this far (body lengths) from where her pixels went in 1 s = left the bird
FLOW_VERSION = 2
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def _box_key(f, r) -> str:
    return f"{int(f)}:{int(round(r.x1))}:{int(round(r.y1))}"


def _seed(g, box, off=(0.0, 0.0), n=40):
    x1, y1, x2, y2 = np.array(box, float) * FLOW_SCALE
    ox, oy = np.array(off) * FLOW_SCALE
    x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
    dx, dy = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    H, W = g.shape
    ya, yb, xa, xb = int(max(0, y1 + dy)), int(min(H, y2 - dy)), int(max(0, x1 + dx)), int(min(W, x2 - dx))
    if yb - ya < 3 or xb - xa < 3:
        return None
    mask = np.zeros_like(g)
    mask[ya:yb, xa:xb] = 255
    return cv2.goodFeaturesToTrack(g, n, 0.01, 3, mask=mask)


def _flow_pass(video: str, jobs: list[dict]) -> None:
    """Run all jobs {kind, start, end, box} through the video in one pass. All points of all active
    jobs go through ONE optical-flow call per frame (forward + backward check), which keeps it fast.
    "after" jobs get .path [(f, x, y)] and .lost_at; "window" jobs get .end_xy or .lost_at."""
    if not jobs:
        return
    jobs.sort(key=lambda j: j["start"])
    cap = cv2.VideoCapture(video)
    pending, active, prev, pos = list(jobs), [], None, None
    last = max(j["end"] for j in jobs)
    f = jobs[0]["start"]
    while f <= last and (active or pending):
        if not active and pending[0]["start"] > f:
            f, prev = pending[0]["start"], None
        if pos != f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        pos = f + 1
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(img, None, fx=FLOW_SCALE, fy=FLOW_SCALE), cv2.COLOR_BGR2GRAY)
        if prev is not None and active:
            p0 = np.concatenate([j["pts"] for j in active]).astype(np.float32)
            idx = np.concatenate([np.full(len(j["pts"]), k) for k, j in enumerate(active)])
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **LK)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **LK)
            ok_all = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) < 1.0)
            still = []
            for k, j in enumerate(active):
                sel = idx == k
                good = ok_all[sel]
                if good.sum() < MIN_POINTS:
                    j["lost_at"] = f
                    continue
                d = (p1[sel] - p0[sel]).reshape(-1, 2)[good]
                if j["kind"] == "after":
                    # background caught in the box doesn't move: follow the moving points when enough of her moves
                    mv = np.linalg.norm(d, axis=1) > 0.3
                    step = np.median(d[mv], axis=0) if mv.sum() >= max(MIN_POINTS, 0.3 * len(d)) else np.zeros(2)
                else:
                    step = np.median(d, axis=0)
                j["off"] = j["off"] + step / FLOW_SCALE
                j["pts"] = p1[sel][good].reshape(-1, 1, 2)
                if j["kind"] == "after":
                    x, y = j["feet0"] + j["off"]
                    j["path"].append((f, int(round(x)), int(round(y))))
                    if len(j["pts"]) < RESEED_BELOW:  # top up from where the dot's box is now
                        new = _seed(g, j["box"], j["off"], n=30)
                        if new is not None:
                            j["pts"] = np.concatenate([j["pts"], new.astype(np.float32)])
                if f >= j["end"]:
                    if j["kind"] == "window":
                        j["end_xy"] = (j["feet0"] + j["off"]).round().astype(int).tolist()
                    continue
                still.append(j)
            active = still
        while pending and pending[0]["start"] <= f:
            j = pending.pop(0)
            j.update(off=np.zeros(2), path=[], lost_at=None,
                     feet0=np.array([(j["box"][0] + j["box"][2]) / 2, j["box"][3]], float))
            pts = _seed(g, j["box"]) if j["start"] == f else None
            if pts is None or len(pts) < MIN_POINTS:
                j["lost_at"] = j["start"]
                continue
            j["pts"] = pts.astype(np.float32)
            active.append(j)
        prev = g
        f += 1
    cap.release()
    for j in jobs:
        j.setdefault("path", [])
        j.setdefault("lost_at", None)
        if j["kind"] == "window" and "end_xy" not in j and j["lost_at"] is None:
            j["lost_at"] = j["end"]


def _flow_jobs(det: pd.DataFrame, fps: float, have: dict) -> list[dict]:
    """Jobs for every piece end / gap inside a piece ("after") and every WIN_S of every piece
    ("window") not already in `have`."""
    jobs, win = [], max(2, int(round(WIN_S * fps)))
    n_frames = int(det.frame.max()) + 1 + int(FLOW_MAX_S * fps)
    for t, g in det.sort_values("frame").groupby("track_id"):
        fr = g.frame.to_numpy()
        rows = list(g.itertuples())
        for i, r in enumerate(rows):  # ends: last frame, or a gap of > 3 frames inside the piece
            if i + 1 == len(rows) or fr[i + 1] - fr[i] > 3:
                key = f"{int(t)}:{int(fr[i])}"
                if key not in have["after"]:
                    jobs.append({"kind": "after", "key": key, "start": int(fr[i]),
                                 "end": int(min(fr[i] + FLOW_MAX_S * fps, n_frames)), "box": [r.x1, r.y1, r.x2, r.y2]})
        pos = {int(f): k for k, f in enumerate(fr)}
        nxt = int(fr[0])
        for k, f in enumerate(fr):  # windows, one every WIN_S
            if f < nxt:
                continue
            nxt = int(f) + win
            if int(f) + win not in pos:
                continue
            r = rows[k]
            key = _box_key(f, r)
            if key not in have["window"]:
                jobs.append({"kind": "window", "key": key, "start": int(f), "end": int(f) + win,
                             "box": [r.x1, r.y1, r.x2, r.y2]})
    return jobs


def flow_cache(run_dir: Path, det: pd.DataFrame, video: str, fps: float) -> dict:
    """identity/flow.json, with anything missing (new pieces after cuts, or everything the first time)
    worked out now."""
    p = Path(run_dir) / "identity" / "flow.json"
    have = {"version": FLOW_VERSION, "after": {}, "window": {}}
    if p.exists():
        try:
            c = json.loads(p.read_text())
            if c.get("version") == FLOW_VERSION:
                have = c
        except ValueError:
            pass
    jobs = _flow_jobs(det, fps, have)
    if jobs:
        _flow_pass(video, jobs)
        for j in jobs:
            if j["kind"] == "after":
                pth = j["path"]
                have["after"][j["key"]] = {"f0": pth[0][0] if pth else j["start"] + 1,
                                           "x": [q[1] for q in pth], "y": [q[2] for q in pth], "lost": j["lost_at"]}
            else:
                have["window"][j["key"]] = None if j.get("lost_at") is not None else [j["end"], *j["end_xy"]]
        p.parent.mkdir(exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(have, separators=(",", ":")))
        tmp.replace(p)
    return have


def box_left_bird(det: pd.DataFrame, track: int, flow: dict, fps: float) -> tuple[list[dict], int]:
    """Where piece `track`'s box moved away from the bird's own pixels (it switched birds there):
    ([{from_frame, to_frame, cut_frame, bodies}], number of one-second checks made)."""
    g = det[det.track_id == track].sort_values("frame")
    by_f = {int(r.frame): r for r in g.itertuples()}
    bad, checked = [], 0
    for f, r in by_f.items():
        v = flow["window"].get(_box_key(f, r))
        if not v:
            continue
        fb, px, py = v
        rb = by_f.get(int(fb))
        if rb is None:
            continue
        checked += 1
        body = max(1.0, float(np.hypot(r.x2 - r.x1, r.y2 - r.y1)))
        off = float(np.hypot((rb.x1 + rb.x2) / 2 - px, rb.y2 - py)) / body
        if off > SPLIT_BODIES:
            bad.append((f, int(fb), off))
    out = []
    for f, fb, off in bad:  # neighbouring bad windows: one event
        if out and f <= out[-1]["to_frame"]:
            out[-1]["to_frame"], out[-1]["bodies"] = fb, max(out[-1]["bodies"], round(off, 1))
        else:
            out.append({"from_frame": f, "to_frame": fb, "bodies": round(off, 1)})
    for e in out:
        e["cut_frame"] = (e["from_frame"] + e["to_frame"]) // 2
    return out, checked


def compute(run_dir: Path, video: str | None = None) -> dict:
    run_dir = Path(run_dir)
    named = pd.read_csv(run_dir / "detections_named.csv")
    meta = json.loads((run_dir / "meta.json").read_text())
    fps, n_frames = float(meta["video"]["fps"]), int(meta["video"]["frame_count"])
    video = video or meta["video"]["path"]
    H = load_calibration(run_dir)
    flow = flow_cache(run_dir, named, video, fps)  # names don't matter here: worked out once, reused
    named = named[named.chicken.notna()]
    hens = sorted(named.chicken.unique())
    body = float(np.median(np.hypot(named.x2 - named.x1, named.y2 - named.y1)))
    M = _Metric(H, body)
    events: list[dict] = []
    best, jobs = {}, []
    for hen in hens:
        b, full = _seen(named, hen)
        best[hen] = b
        # two places at once: two full boxes of hers in one frame, far apart
        far = []
        for f, g in full.groupby("frame"):
            if len(g) < 2:
                continue
            p = np.stack([(g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2], 1).to_numpy()
            sz = np.hypot(g.x2 - g.x1, g.y2 - g.y1).to_numpy()
            dmax = max(np.linalg.norm(p[i] - p[k]) / ((sz[i] + sz[k]) / 2)
                       for i in range(len(p)) for k in range(i + 1, len(p)))
            if dmax > TWO_PLACES_BODIES:
                far.append(int(f))
        for a, z in _runs(far, fps, MIN_EVENT_S):
            tr = sorted(full[(full.frame >= a) & (full.frame <= z)].track_id.unique().tolist())
            events.append({"kind": "two_places", "hen": hen, "from_frame": a, "to_frame": z,
                           "tracks": tr, "detail": f"{hen} boxed in two places at once (pieces {', '.join(f'#{t}' for t in tr)})"})
        fr = b.index.to_numpy()
        # teleports between nearly consecutive sightings
        for i in range(1, len(fr)):
            df = fr[i] - fr[i - 1]
            if df > 3:
                continue
            ra, rb = b.loc[fr[i - 1]], b.loc[fr[i]]
            dist = M.m(_feet(ra), _feet(rb))
            if dist > MAX_SPEED * df / fps + 0.3:
                events.append({"kind": "teleport", "hen": hen, "from_frame": int(fr[i - 1]), "to_frame": int(fr[i]),
                               "tracks": sorted({int(ra.track_id), int(rb.track_id)}),
                               "detail": f"{hen} jumps {dist:.1f} m in {df / fps:.2f} s (#{int(ra.track_id)} → #{int(rb.track_id)})"})
        # gaps: follow the pixels
        for i in range(len(fr)):
            nxt = fr[i + 1] if i + 1 < len(fr) else n_frames
            if nxt - fr[i] <= 3:
                continue
            r = b.loc[fr[i]]
            jobs.append({"hen": hen, "start": int(fr[i]), "end": int(min(nxt - 1, fr[i] + FLOW_MAX_S * fps, n_frames - 1)),
                         "reappear": int(nxt) if nxt < n_frames else None, "box": [r.x1, r.y1, r.x2, r.y2],
                         "track": int(r.track_id)})
    for j in jobs:  # her dot while she's lost: the saved pixel-following from where she was last seen
        a = flow["after"].get(f"{j['track']}:{j['start']}")
        if a is None:
            j["path"], j["lost_at"] = [], j["start"] + 1
            continue
        n = max(0, min(len(a["x"]), j["end"] - a["f0"] + 1))
        j["path"] = [(a["f0"] + k, a["x"][k], a["y"][k]) for k in range(n)]
        j["lost_at"] = a["lost"] if a["lost"] is not None and a["lost"] <= j["end"] else None

    # every hen's dot, every frame: x, y, and where it came from (S seen, F followed, L lost, - unknown)
    dots = {}
    for hen in hens:
        b = best[hen]
        x = np.full(n_frames, -1, np.int32)
        y = np.full(n_frames, -1, np.int32)
        src = np.full(n_frames, "-", "<U1")
        dots[hen] = (x, y, src)
    for j in jobs:
        x, y, src = dots[j["hen"]]
        last_xy = np.array([(j["box"][0] + j["box"][2]) / 2, j["box"][3]])
        start_xy, moved = last_xy.copy(), False
        for f, px, py in j["path"]:
            # "F" only once her pixels have really carried the dot somewhere; pixels that stay put
            # (she sits still, or went under a bush and the points are on leaves) = "W": waiting where last seen
            moved = moved or np.hypot(px - start_xy[0], py - start_xy[1]) > MOVED_BODIES * body
            x[f], y[f], src[f] = round(px), round(py), "F" if moved else "W"
            last_xy = np.array([px, py])
        lost_from = j["lost_at"] if j["lost_at"] is not None else (j["end"] + 1)
        stop = j["reappear"] if j["reappear"] is not None else n_frames
        if lost_from < stop:
            x[lost_from:stop], y[lost_from:stop], src[lost_from:stop] = round(last_xy[0]), round(last_xy[1]), "L"
        j["dot_end"] = last_xy
        j["followed_to_end"] = j["lost_at"] is None and j["end"] + 1 >= (j["reappear"] or 0)
    for hen in hens:  # sightings last: they always win over followed / lost dots
        b, (x, y, src) = best[hen], dots[hen]
        fr = b.index.to_numpy()
        x[fr] = ((b.x1 + b.x2) / 2).round().astype(int).to_numpy()
        y[fr] = b.y2.round().astype(int).to_numpy()
        src[fr] = "S"

    # reappearing: plausible distance? does it agree with where the pixels went?
    for j in jobs:
        if j["reappear"] is None:
            continue
        hen, b = j["hen"], best[j["hen"]]
        rb = b.loc[j["reappear"]]
        back = _feet(rb)
        gap_s = (j["reappear"] - j["start"]) / fps
        start = np.array([(j["box"][0] + j["box"][2]) / 2, j["box"][3]])
        far_m = M.m(start, back)
        tr = sorted({j["track"], int(rb.track_id)})
        if far_m > WALK_SPEED * gap_s + 0.5:
            events.append({"kind": "gap_jump", "hen": hen, "from_frame": j["start"], "to_frame": j["reappear"], "tracks": tr,
                           "detail": f"{hen} gone {gap_s:.1f} s, back {far_m:.1f} m away — too far to walk"})
        elif j["followed_to_end"] and len(j["path"]) >= 3:
            off = float(np.linalg.norm(j["dot_end"] - back)) / body
            if off > FLOW_SPLIT_BODIES:
                near = _nearest_other(best, hen, j["reappear"], j["dot_end"], body)
                hint = f" — the dot ended next to {near}" if near else ""
                events.append({"kind": "flow_split", "hen": hen, "from_frame": j["start"], "to_frame": j["reappear"],
                               "tracks": tr, "with": near,
                               "detail": f"{hen}'s pixels went one way, \"{hen}\" reappears {off:.1f} body lengths away{hint}"})

    # two hens on top of each other
    for i, a in enumerate(hens):
        for c in hens[i + 1:]:
            ba, bc = best[a], best[c]
            common = ba.index.intersection(bc.index)
            if not len(common):
                continue
            pa = np.stack([((ba.x1 + ba.x2) / 2).loc[common], ba.y2.loc[common]], 1)
            pc = np.stack([((bc.x1 + bc.x2) / 2).loc[common], bc.y2.loc[common]], 1)
            close = common[np.linalg.norm(pa - pc, axis=1) / body < STACK_BODIES]
            for s0, s1 in _runs(close.tolist(), fps, 1.0):
                tr = sorted(set(ba.loc[s0:s1].track_id.astype(int)) | set(bc.loc[s0:s1].track_id.astype(int)))
                events.append({"kind": "stacked", "hen": a, "with": c, "from_frame": s0, "to_frame": s1, "tracks": tr,
                               "detail": f"{a} and {c} on the same spot for {(s1 - s0) / fps:.1f} s — one bird named twice?"})

    events = _merge_flips(sorted(events, key=lambda e: e["from_frame"]), fps)
    for e in events:
        e["time_s"] = round(e["from_frame"] / fps, 1)
    out = {"version": VERSION, "fps": fps, "frames": n_frames, "hens": hens,
           "dots": {h: {"x": dots[h][0].tolist(), "y": dots[h][1].tolist(), "src": "".join(dots[h][2])} for h in hens},
           "illegal": events}
    (run_dir / "dots.json").write_text(json.dumps(out, separators=(",", ":")))
    pd.DataFrame([{k: e.get(k) for k in ("time_s", "kind", "hen", "with", "from_frame", "to_frame", "tracks", "detail")}
                  for e in events]).to_csv(run_dir / "illegal_moves.csv", index=False)
    return out


def _merge_flips(events: list[dict], fps: float, within_s: float = 1.0) -> list[dict]:
    """Teleports of one hen close together are one problem (usually two boxes with her name taking turns)."""
    out, last = [], {}
    for e in events:
        prev = last.get(e["hen"])
        if e["kind"] == "teleport" and prev is not None and e["from_frame"] - prev["to_frame"] <= within_s * fps:
            prev["to_frame"] = max(prev["to_frame"], e["to_frame"])
            prev["tracks"] = sorted(set(prev["tracks"]) | set(e["tracks"]))
            prev["n"] = prev.get("n", 1) + 1
            prev["detail"] = (f"{e['hen']}'s dot jumps back and forth {prev['n']} times between pieces "
                              f"{', '.join(f'#{t}' for t in prev['tracks'])} — two boxes with her name?")
            continue
        out.append(e)
        if e["kind"] == "teleport":
            last[e["hen"]] = e
    return out


def _nearest_other(best: dict, hen: str, frame: int, p: np.ndarray, body: float) -> str | None:
    cand = []
    for h, b in best.items():
        if h == hen or frame not in b.index:
            continue
        cand.append((float(np.linalg.norm(_feet(b.loc[frame]) - p)) / body, h))
    cand.sort()
    return cand[0][1] if cand and cand[0][0] < 1.5 else None


# ------------------------------------------------------------------ for the Questions page

CLOSE_BODIES = 1.5     # a hen's dot this close to where the piece starts/ends: she fits
ELSEWHERE_SHARE = 0.2  # seen elsewhere for this share of the piece (and >= 1 s): can't be her
LOST_MAX_S = 60.0      # a lost dot older than this says nothing
STRONG_LOST_S = 3.0    # lost / waiting up to this long: still strong evidence she's there
STRONG_MOVED_S = 15.0  # dot carried by her moving pixels up to this long: strong evidence


def whereabouts(det: pd.DataFrame, owner: dict[int, int], names: dict[int, str], track: int,
                dots: dict | None, fps: float, asked: int | None = None, sure: set | None = None) -> dict:
    """Where was every hen while piece `track` was seen? Rules hens out (seen elsewhere at the same
    time - from the current naming) and in (her dot was right where the piece starts / ends).
    `dots` is dots.json (may be older than the latest answers, or None)."""
    d = det[det.track_id == track].sort_values("frame")
    if d.empty:
        return {"error": "gone"}
    f0, f1 = int(d.frame.iloc[0]), int(d.frame.iloc[-1])
    p0, p1 = _feet(d.iloc[0]), _feet(d.iloc[-1])
    body = float(np.median(np.hypot(d.x2 - d.x1, d.y2 - d.y1)))
    body = max(body, 0.6 * float(np.median(np.hypot(det.x2 - det.x1, det.y2 - det.y1))), 1.0)
    mine = d.set_index("frame")
    cx = ((mine.x1 + mine.x2) / 2).groupby(level=0).first()
    cy = ((mine.y1 + mine.y2) / 2).groupby(level=0).first()
    span = det[(det.frame >= f0) & (det.frame <= f1) & (det.track_id != track)]
    hens = []
    for cid, name in names.items():
        h = {"chicken": cid, "name": name, "asked": cid == asked}
        # ruled out: one of her other pieces is somewhere else at the same time
        o = span[span.track_id.map(owner) == cid]
        if len(o):
            oc = o.assign(ox=(o.x1 + o.x2) / 2, oy=(o.y1 + o.y2) / 2).groupby("frame")[["ox", "oy"]].first()
            oc = oc[oc.index.isin(cx.index)]
            far = np.hypot(oc.ox - cx.loc[oc.index], oc.oy - cy.loc[oc.index]) / body > TWO_PLACES_BODIES
            secs = float(far.sum()) / fps
            if secs >= 1.0 and far.sum() / max(1, len(cx)) >= ELSEWHERE_SHARE:
                h["elsewhere_s"] = round(secs, 1)
                h["elsewhere_tracks"] = sorted(o[o.frame.isin(oc.index[far.to_numpy()])].track_id.unique().astype(int).tolist())[:4]
        if dots and name in dots["dots"]:
            dd = dots["dots"][name]
            h["before"] = _dot_at(dd, f0 - 1, p0, body, fps, back=True)
            after = _dot_at(dd, f1 + 1, p1, body, fps, back=False)
            if cid == asked and after and after["src"] != "S":  # her own dot after this piece just follows this piece
                after = _next_seen(dd, f1 + 1, p1, body, fps)
            h["after"] = after
            for x in (h.get("before"), h.get("after")):  # a sighting: which piece, and did YOU name it?
                if x and x["src"] == "S":
                    rows = det[(det.frame == x["frame"]) & (det.track_id.map(owner) == cid) & (det.track_id != track)]
                    if len(rows):
                        k = np.argmin(np.hypot((rows.x1 + rows.x2) / 2 - dd["x"][x["frame"]], rows.y2 - dd["y"][x["frame"]]))
                        x["track"] = int(rows.track_id.iloc[k])
                        x["confirmed"] = sure is not None and x["track"] in sure
        ev = [x for x in (h.get("before"), h.get("after")) if x and x["bodies"] <= CLOSE_BODIES]
        h["fits"] = ("no" if "elsewhere_s" in h else "yes" if any(x["strength"] == "strong" for x in ev)
                     else "near" if any(x["strength"] == "weak" for x in ev) else "maybe")
        hens.append(h)
    order = {"yes": 0, "near": 1, "maybe": 2, "no": 3}
    hens.sort(key=lambda h: (order[h["fits"]], min([x["bodies"] for x in (h.get("before"), h.get("after")) if x] or [99])))
    illegal = [e for e in (dots or {}).get("illegal", []) if track in e.get("tracks", [])]
    return {"track": track, "hens": hens, "illegal": illegal, "has_dots": dots is not None,
            "fits": [h["name"] for h in hens if h["fits"] == "yes"],
            "near": [h["name"] for h in hens if h["fits"] == "near"],
            "ruled_out": [h["name"] for h in hens if h["fits"] == "no"]}


def _dot_at(dd: dict, f: int, p: np.ndarray, body: float, fps: float, back: bool) -> dict | None:
    src = dd["src"]
    if f < 0 or f >= len(src) or src[f] == "-":
        return None
    age = 0
    if src[f] != "S":  # how long since she was last really seen
        g = f
        while 0 <= g < len(src) and src[g] != "S" and age < LOST_MAX_S * fps:
            g += -1 if back else 1
            age += 1
    dist = float(np.hypot(dd["x"][f] - p[0], dd["y"][f] - p[1])) / body
    a, k = age / fps, src[f]
    strength = ("strong" if k == "S" or (k == "F" and a <= STRONG_MOVED_S) or a <= STRONG_LOST_S
                else "weak" if a < LOST_MAX_S else "none")
    return {"src": k, "bodies": round(dist, 1), "age_s": round(a, 1), "strength": strength,
            "counts": strength != "none", "frame": int(f)}


def _next_seen(dd: dict, f: int, p: np.ndarray, body: float, fps: float, within_s: float = 4.0) -> dict | None:
    src = dd["src"]
    for g in range(f, min(len(src), f + int(within_s * fps))):
        if src[g] == "S":
            dist = float(np.hypot(dd["x"][g] - p[0], dd["y"][g] - p[1])) / body
            return {"src": "S", "bodies": round(dist, 1), "age_s": round((g - f) / fps, 1), "counts": True,
                    "strength": "strong", "later": True, "frame": int(g)}
    return None
