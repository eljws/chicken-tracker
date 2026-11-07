"""Local web app: name chickens, merge groups, answer "is this Rocky?", fix conflicts, cut pieces."""

from __future__ import annotations

import functools
import json
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

from . import checks as K
from . import identity as I
from . import recognizer as R

STRIP_LEN = 12


def make_handler(run_dir: Path):
    run_dir = Path(run_dir)
    store = I.IdentityStore(run_dir)
    crops = store.root / "crops"
    page = (resources.files("chicken_tracker") / "static" / "identity.html").read_bytes()
    lock = threading.Lock()
    # mutable state shared by handlers; replaced wholesale after a cut
    st = {"det": I.load_detections(run_dir, store.data), "frags": I.load_fragments(run_dir), "version": 0}

    @functools.lru_cache(maxsize=512)
    def snapshot(a: int, b: int, k: int, version: int) -> bytes | None:
        return I.overlap_snapshot(store.data["video"], st["det"], a, b, k, fps=store.data["fps"])

    @functools.lru_cache(maxsize=2048)
    def strip_img(track: int, frame: int, version: int, wide: bool = False) -> bytes | None:
        return I.piece_frame(store.data["video"], st["det"], track, frame, store.data["fps"], wide=wide)

    @functools.lru_cache(maxsize=16)
    def clip_mp4(track: int, version: int, direction: int = 0, span: tuple[int, int] | None = None) -> bytes | None:
        if span:  # a checks stretch: from a moment before to a moment after
            pad = int(2 * store.data["fps"])
            return I.piece_clip(store.data["video"], st["det"], track, store.data["fps"], span=(span[0] - pad, span[1] + pad))
        if not direction:
            return I.piece_clip(store.data["video"], st["det"], track, store.data["fps"])
        tr = trace_view(track, version)["back" if direction < 0 else "forward"]
        if not tr:
            return None
        fps = store.data["fps"]
        mine = st["det"].loc[st["det"].track_id == track, "frame"]
        far = tr[-1]
        span = ((far["last"] if far.get("anchor") else far["first"]) - 3 * fps, mine.min() + 3 * fps) if direction < 0 \
            else (mine.max() - 3 * fps, (far["first"] if far.get("anchor") else far["last"]) + 3 * fps)
        chain = {h["track"]: f"#{h['track']} {h['name']}{' (sure)' if h['sure'] else ''}" for h in tr}
        return I.piece_clip(store.data["video"], st["det"], track, fps, chain=chain, span=span)

    dots_state = {"key": None, "data": None, "computing": False, "error": None}

    def dots_data():
        p = run_dir / "dots.json"
        if not p.exists():
            return None
        key = p.stat().st_mtime
        if dots_state["key"] != key:
            dots_state["key"], dots_state["data"] = key, json.loads(p.read_text())
        return dots_state["data"]

    def compute_dots():
        from . import dots as DOTS
        try:
            while True:
                dots_state["again"] = False
                try:
                    DOTS.compute(run_dir, video=store.data["video"])
                except (ValueError, KeyError, OSError):  # caught a file mid-write: try again shortly
                    dots_state["again"] = True
                if not dots_state["again"]:
                    break
                threading.Event().wait(1.0)
        except Exception as e:  # shown on the page
            dots_state["error"] = str(e)
        finally:
            dots_state["computing"] = False

    def schedule_dots(delay: float = 1.5):
        """Debounced: rebuild the dots 1.5 s after the last change (answer, cut, mark...). Uses the saved
        pixel-following, so it's a few seconds; only for clips whose dots were worked out once."""
        if not (run_dir / "dots.json").exists():
            return
        if dots_state["computing"]:
            dots_state["again"] = True
            return
        if dots_state.get("timer"):
            dots_state["timer"].cancel()

        def fire():
            dots_state.update(timer=None, computing=True, error=None)
            compute_dots()
        dots_state["timer"] = threading.Timer(delay, fire)
        dots_state["timer"].daemon = True
        dots_state["timer"].start()

    flow_state = {"key": None, "data": None}

    def flow_data():
        p = store.root / "flow.json"
        if not p.exists():
            return None
        if flow_state["key"] != p.stat().st_mtime:
            flow_state["key"], flow_state["data"] = p.stat().st_mtime, json.loads(p.read_text())
        return flow_state["data"]

    def where_view(t: int) -> dict:
        from . import dots as DOTS
        import datetime as dt
        names = {c["id"]: name_of(c["id"]) for c in store.data["chickens"]}
        dd = dots_data()
        out = DOTS.whereabouts(st["det"], I.chicken_of(store.data), names, t, dd, store.data["fps"],
                               asked=I.chicken_of(store.data).get(t), sure=I.confirmed(store.data))
        out["computing"], out["dots_error"] = dots_state["computing"], dots_state["error"]
        out["updating"] = bool(dots_state.get("timer")) or (dots_state["computing"] and dd is not None)
        if dd:
            out["dots_made"] = dt.datetime.fromtimestamp(dots_state["key"]).strftime("%d %b %H:%M:%S")
        fl = flow_data()
        if fl:  # does the box stay on the bird its pixels show? one check per second of the piece
            splits, checked = DOTS.box_left_bird(st["det"], t, fl, store.data["fps"])
            out["pixels"] = {"checked": checked, "splits": [{**e, "clock": clock(e["cut_frame"]),
                                                              "at_s": round(e["cut_frame"] / store.data["fps"], 1)}
                                                             for e in splits]}
        return out

    MARK_BGR = [(229, 135, 57), (38, 89, 217), (112, 158, 25), (0, 133, 201), (129, 81, 213),
                (233, 133, 144), (103, 103, 230), (0, 131, 0)]  # same colours as the page's hen brushes

    @functools.lru_cache(maxsize=16)
    def where_mp4(track: int, version: int, dots_key, wide: bool = False) -> bytes | None:
        """The piece from 3 s before to 3 s after, with every hen's dot in her colour.
        wide: 60 s before to 60 s after at 4x speed, each dot pulling a 6 s trail, timeline bar."""
        import cv2
        dd = dots_data()
        if not dd:
            return None
        fps = store.data["fps"]
        w = where_view(track)
        fr = st["det"].loc[st["det"].track_id == track, "frame"]
        pad = (60 if wide else 3) * fps
        f0, f1 = int(max(0, fr.min() - pad)), int(fr.max() + pad)
        if wide and f1 - f0 > 240 * fps:  # very long piece: keep it under 4 min of video time
            f1 = f0 + int(240 * fps)
        p0, p1 = int(fr.min()), int(fr.max())
        col = {name_of(c["id"]): MARK_BGR[i % len(MARK_BGR)] for i, c in enumerate(store.data["chickens"])}
        pd_ = st["det"][st["det"].track_id == track]
        cxy = (float(((pd_.x1 + pd_.x2) / 2).median()), float(pd_.y2.median()))
        body = float(I.np.hypot(pd_.x2 - pd_.x1, pd_.y2 - pd_.y1).median())
        show = [h["name"] for h in w["hens"] if h["fits"] in (("yes", "near") if wide else ("yes",)) or h["asked"]]
        pts = []
        for n in show:
            dn = dd["dots"].get(n)
            if dn:  # only where she was seen or moving near the piece, so the view isn't stretched to the whole yard
                pts += [(dn["x"][g], dn["y"][g]) for g in range(f0, min(f1 + 1, len(dn["src"])), 4)
                        if dn["src"][g] in ("SF" if wide else "SFWL")
                        and I.np.hypot(dn["x"][g] - cxy[0], dn["y"][g] - cxy[1]) < 8 * body]
        trail = int(6 * fps)

        mine = st["det"][st["det"].track_id == track].groupby("frame").first()
        asked = next((h["name"] for h in w["hens"] if h["asked"]), None)
        F = cv2.FONT_HERSHEY_SIMPLEX

        def text(out, s, xy, c, size=0.55):
            cv2.putText(out, s, xy, F, size, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, s, xy, F, size, c, 1 if size < 0.6 else 2, cv2.LINE_AA)

        def draw(out, f, to):  # drawn on the final picture: same size whatever the zoom
            H, W = out.shape[:2]
            for k, (n, dn) in enumerate(sorted(dd["dots"].items())):
                c = col.get(n, (200, 200, 200))
                if wide:  # her last 6 s as a line that fades out behind her: where she came from
                    prev = None
                    for g in range(max(0, f - trail), min(f + 1, len(dn["src"])), 2):
                        if dn["src"][g] in "SF":
                            p = to(dn["x"][g], dn["y"][g])
                            if prev is not None and abs(p[0] - prev[0]) + abs(p[1] - prev[1]) < W // 4:
                                a = 0.25 + 0.75 * (g - f + trail) / trail
                                cv2.line(out, prev, p, tuple(int(v * a) for v in c), 3, cv2.LINE_AA)
                            prev = p
                        else:
                            prev = None
                if f >= len(dn["src"]) or dn["src"][f] == "-":
                    continue
                src = dn["src"][f]
                x, y = to(dn["x"][f], dn["y"][f])
                if not (0 <= x < W and 0 <= y < H):
                    continue
                if n == asked and src == "S" and f in mine.index:  # that's this piece itself, already boxed
                    r = mine.loc[f]
                    if abs((r.x1 + r.x2) / 2 - dn["x"][f]) < 3 and abs(r.y2 - dn["y"][f]) < 3:
                        continue
                if src == "S":
                    cv2.circle(out, (x, y), 8, (0, 0, 0), -1)
                    cv2.circle(out, (x, y), 6, c, -1)
                    lab = n
                elif src == "F":
                    cv2.circle(out, (x, y), 9, (0, 0, 0), 4)
                    cv2.circle(out, (x, y), 9, c, 2)
                    lab = n + " (moving)"
                else:  # not seen, waiting where last seen: faint, no label (they pile up where hens hide)
                    cv2.circle(out, (x, y), 7, tuple(int(v * 0.6) for v in c), 1)
                    continue
                text(out, lab, (x + 12, y + 5 + (k % 3 - 1) * 16), c, 0.6)
            if wide:  # timeline: whole span grey, the piece yellow, white playhead
                y, X = H - 6, lambda g: int((g - f0) / max(1, f1 - f0) * (W - 1))
                cv2.rectangle(out, (0, y - 3), (W - 1, H - 1), (70, 70, 70), -1)
                cv2.rectangle(out, (X(p0), y - 3), (X(p1), H - 1), (0, 215, 255), -1)
                cv2.rectangle(out, (X(f) - 2, y - 9), (X(f) + 2, H - 1), (255, 255, 255), -1)
                inside = p0 <= f <= p1
                text(out, "4x  " + ("THIS PIECE" if inside else "before" if f < p0 else "after"), (W - 200, 28),
                     (0, 215, 255) if inside else (255, 255, 255), 0.7)

        return I.piece_clip(store.data["video"], st["det"], track, fps, span=(f0, f1), extra_pts=pts,
                            max_s=250.0 if wide else 90.0, step=4 if wide else 1, draw_after=draw)

    def belief(t: int) -> dict:
        """Who a piece is and how sure: confirmed by you, recogniser >= 0.9 over >= 5 crops, or a guess."""
        owner, name = I.chicken_of(store.data), name_of
        c = owner.get(t)
        out = {"name": name(c) if c is not None else "—", "chicken": c, "sure": False, "by": "guess", "p": None}
        for key, why in ((k, w) for w, k in I.EXCLUDED_KEYS.items()):
            if t in store.data.get(key, []):
                return {**out, "name": why, "by": "you", "sure": False}
        if t in I.confirmed(store.data):
            return {**out, "sure": True, "by": "you"}
        rec = recog()
        if rec:
            pred, classes = rec
            rows = pred[pred.track_id == t]
            if len(rows) >= 5:
                cols = [f"p_{k}" for k in classes]
                mean = rows[cols].mean()
                top = mean.idxmax()[2:]
                out.update(p=round(float(mean.max()), 2), rec=top)
                if mean.max() >= 0.9:
                    agree = c is not None and out["name"] == top
                    out.update(by="recogniser" if agree else "guess", sure=agree)
        return out

    @functools.lru_cache(maxsize=64)
    def trace_view(track: int, version: int) -> dict:
        fps = store.data["fps"]
        res = {}
        for key, d in (("back", -1), ("forward", 1)):
            hops = I.trace(st["det"], track, d, fps, is_anchor=lambda t: belief(t)["sure"])
            for h in hops:
                h.update(belief(h["track"]))
                h["clock"] = clock(h["join_frame"])
                if h["alt"] is not None:
                    h["alt_name"] = belief(h["alt"])["name"]
            res[key] = hops
        return res

    def current_checks() -> list[dict]:
        named_p = run_dir / "detections_named.csv"
        if not named_p.exists():
            return []
        picks = I.pd.read_csv(store.root / "crops.csv")
        feats = I.np.load(store.root / "features.npy")
        from . import sizes as SZ
        from .calibration import load_calibration
        cal, H = SZ.load(run_dir), load_calibration(run_dir)
        ck = K.run_checks(store.data, st["det"], I.pd.read_csv(named_p), picks, feats, st["frags"],
                          store.data["fps"], recog=recog(), sizes=(cal, H) if cal and H is not None else None)
        irt = ir_tracks()  # infrared: out of scope
        return [c for c in ck if c["track"] not in irt and not in_ir(c["from_frame"], c["to_frame"])]

    def recog():
        """Recogniser predictions joined to the current pieces; re-joined after cuts."""
        p = store.root / R.PRED_FILE
        key = (st["version"], p.stat().st_mtime if p.exists() else 0)
        if st.get("recog_key") != key:
            st["recog_key"], st["recog"] = key, R.load_predictions(run_dir, st["det"])
        return st["recog"]

    def _check_view(c: dict) -> dict:
        frames = I.np.sort(st["det"].loc[st["det"].track_id == c["track"], "frame"].to_numpy())
        pad = int(3 * store.data["fps"])
        win = frames[(frames >= c["from_frame"] - pad) & (frames <= c["to_frame"] + pad)]
        idx = I.np.unique(I.np.linspace(0, len(win) - 1, min(12, len(win))).round().astype(int)) if len(win) else []
        strip = [int(win[i]) for i in idx]
        return {**c, "strip": strip, "in_range": [c["from_frame"] <= f <= c["to_frame"] for f in strip]}

    def mid_crop(t: int) -> str:
        c = store.data["fragments"][str(t)]["crops"]
        return c[len(c) // 2] if c else ""

    def piece(t: int) -> dict:
        f = store.data["fragments"][str(t)]
        return {"track": t, "start": f["start"], "duration_s": round(f["end"] - f["start"], 1),
                "sure": t in I.confirmed(store.data), "strip": I.strip_frames(st["det"], t, STRIP_LEN)}

    # infrared (black-and-white) stretches are out of scope: measured once in the background, then the
    # pieces mostly inside them get no checks, questions or "?" marks (their names stay as they are)
    ir_state: dict = {"ir": None}

    def _measure_ir():
        from . import daylight as DL
        try:
            ir_state["ir"] = DL.infrared_s(run_dir)
        except Exception:  # never block naming on this
            ir_state["ir"] = []
    threading.Thread(target=_measure_ir, daemon=True).start()

    def ir_tracks() -> set[int]:
        from . import daylight as DL
        ir = ir_state["ir"] or []
        key = (st["version"], len(store.data["fragments"]), tuple(map(tuple, ir)))
        if ir_state.get("k") != key:
            ir_state["k"], ir_state["tracks"] = key, DL.infrared_tracks(store.data, ir)
        return ir_state["tracks"]

    def in_ir(f0: int, f1: int) -> bool:
        a, b = f0 / store.data["fps"], f1 / store.data["fps"]
        ir = ir_state["ir"] or []
        if b <= a:
            return any(x <= a <= y for x, y in ir)
        return sum(max(0.0, min(b, y) - max(a, x)) for x, y in ir) > 0.5 * (b - a)

    CHECK_SHORT_S, CHECK_P, CHECK_SEVERAL_P = 2.0, 0.8, 0.2
    check_cache: dict = {}

    def to_check() -> dict[int, str]:
        """{piece: why} for the computer's guesses worth a look: seen under 2 s, the recogniser under
        80% sure or naming another hen, or giving "several hens" 20%+. Marks only, never a choice."""
        rec = recog()
        key = (st["version"], st.get("recog_key"), ir_state.get("k"))
        ir_tracks()
        if check_cache.get("k") == key:
            return check_cache["v"]
        d, fr = store.data, store.data["fragments"]
        done = I.confirmed(d) | I.excluded(d)
        owner = {t: name_of(c) for t, c in I.chicken_of(d).items()}
        means = {}
        if rec:
            pred, classes = rec
            cols = [f"p_{k}" for k in classes]
            clear = pred[pred.crowd <= 0.3] if "crowd" in pred else pred
            means = {int(t): g.mean() for t, g in clear.groupby("track_id")[cols]}
            means |= {int(t): g.mean() for t, g in pred.groupby("track_id")[cols] if int(t) not in means}
        out = {}
        irt = ir_tracks()
        for t, name in owner.items():
            if t in done or t in irt:
                continue
            why = []
            dur = fr[str(t)]["end"] - fr[str(t)]["start"]
            if dur < CHECK_SHORT_S:
                why.append(f"very short ({dur:.2f} s)")
            m = means.get(t)
            if m is not None:
                hens = m.drop(f"p_{R.SEVERAL}", errors="ignore")
                top = hens.idxmax()[2:]
                own = float(m.get(f"p_{name}", float("nan")))
                if f"p_{name}" in m and top != name:
                    why.append(f"recogniser thinks {top} ({hens.max():.0%})")
                elif own == own and own < CHECK_P:
                    why.append(f"recogniser only {own:.0%} sure")
                sev = float(m.get(f"p_{R.SEVERAL}", 0))
                if sev >= CHECK_SEVERAL_P:
                    why.append(f"maybe several hens ({sev:.0%})")
            if why:
                out[t] = " · ".join(why)
        check_cache.update(k=key, v=out)
        return out

    status_last: dict = {}

    def write_status(s: dict) -> None:
        """identity/status.json: what's waiting on this clip (checks need the page to work them out),
        for the home page's Naming button. Written only when it changes."""
        if s == status_last.get("s"):
            p = store.root / "status.json"
            if p.exists() and p.stat().st_mtime >= (store.root / "identity.json").stat().st_mtime:
                return
        status_last["s"] = dict(s)
        p = store.root / "status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({**s, "at": time.time()}))
        tmp.replace(p)

    seen_cache: dict = {}

    def seen() -> dict:  # frames each piece has a box in; changes only when pieces are cut or undone
        if seen_cache.get("v") != (st["version"], id(st["det"])):
            seen_cache.update(v=(st["version"], id(st["det"])), frames=I.seen_frames(st["det"]))
        return seen_cache["frames"]

    def state() -> dict:
        d = store.data
        fr = d["fragments"]
        chk = to_check()
        by_claude = I.answered_by_claude(d)
        chickens = []
        for c in d["chickens"]:
            tr = sorted(c["tracks"], key=lambda t: -fr[str(t)]["n"])
            chickens.append({
                "id": c["id"], "name": c["name"],
                "minutes": round(sum(fr[str(t)]["end"] - fr[str(t)]["start"] for t in tr) / 60, 1),
                "fragments": [{"track": t, "crop": mid_crop(t), "sure": t in I.confirmed(d),
                               "flag": I.box_flag(d, t), **({"check": chk[t]} if t in chk else {}),
                               **({"claude": True} if t in by_claude else {})} for t in tr],
            })
        irt = ir_tracks()
        qs = [q for q in I.questions(d, extra=chk) if q["track"] not in irt]
        aside = {reason: [{"track": t, "crop": mid_crop(t), **({"claude": True} if t in by_claude else {})} for t in d.get(key, [])]
                 for reason, key in I.EXCLUDED_KEYS.items()}
        # pieces in no hen's row and no pile (automatic naming found every hen elsewhere at that time):
        # shown in their own row so they can be answered, and counted as questions
        placed = {t for c in d["chickens"] for t in c["tracks"]} | I.excluded(d)
        unplaced = [{"track": int(t), "crop": mid_crop(int(t))} for t in sorted(fr, key=lambda t: fr[t]["start"])
                    if int(t) not in placed and int(t) not in irt]
        cf = [c for c in I.conflicts(d, seen=seen()) if not (c["a"] in irt and c["b"] in irt)]
        ck = current_checks()
        write_status({"conflicts": len(cf), "questions": len(qs) + len(unplaced), "checks": len(ck), "to_check": len(chk)})
        return {"chickens": chickens, "aside": aside, "unplaced": unplaced, "fps": d["fps"], "version": st["version"],
                "checks": len(ck), "check": _check_view(ck[0]) if ck else None,
                "check_list": [{k: c[k] for k in ("key", "kind", "title", "seconds", "from_frame", "to_frame", "track")} for c in ck[:40]],
                "undo": store.history(),
                "conflicts": len(cf), "conflict": _conflict(cf[0]) if cf else None,
                "questions": len(qs), "next": _question(qs[0]) if qs else None, "to_check": len(chk),
                "answered": len(d["answers"]),
                "infrared": ir_state["ir"] or [], "infrared_pieces": len(irt)}

    def _question(q: dict) -> dict:
        c = next(c for c in store.data["chickens"] if c["id"] == q["chicken"])
        ref = sorted((t for t in c["tracks"] if t != q["track"]),
                     key=lambda t: -store.data["fragments"][str(t)]["n"])[:8]
        return {**q, **piece(q["track"]), "strip": I.strip_frames(st["det"], q["track"], 20), "ref_crops": [x for x in map(mid_crop, ref) if x],
                "name": c["name"] or f"Chicken {c['id'] + 1}"}

    def _conflict(cf: dict) -> dict:
        c = next(c for c in store.data["chickens"] if c["id"] == cf["chicken"])
        fr = store.data["fragments"]
        lo = cf.get("from_s", max(fr[str(cf["a"])]["start"], fr[str(cf["b"])]["start"]))
        hi = cf.get("to_s", min(fr[str(cf["a"])]["end"], fr[str(cf["b"])]["end"]))
        pieces = []
        for t, other in ((cf["a"], cf["b"]), (cf["b"], cf["a"])):
            p = piece(t)
            # the 12 pictures are spread over the whole piece and can all miss a short overlap:
            # add up to 6 from inside it, so it can be seen and painted
            fps = store.data["fps"]
            ov = I.gap_frames(st["det"], t, round(lo * fps), round(hi * fps), n=6)  # round: ends are frame times
            p["strip"] = sorted(set(p["strip"]) | set(ov))
            p["can_isolate"] = bool(I.overlap_cuts(st["det"], t, other))
            p["other"] = other
            pieces.append(p)
        return {**cf, "name": c["name"] or f"Chicken {c['id'] + 1}", "pieces": pieces,
                "overlap_from": lo, "overlap_to": hi}

    def name_of(cid) -> str:
        if cid is None:
            return ""
        c = next((c for c in store.data["chickens"] if c["id"] == int(cid)), None)
        return (c["name"] or f"Chicken {c['id'] + 1}") if c else f"chicken {cid}"

    def clock(frame: int) -> str:
        t = int(frame / store.data["fps"])
        return f"{t // 60:02d}:{t % 60:02d}"

    VERDICT_LABEL = {"yes": "is {name}", "moved": "→ {name}", "partial": "only part of {name}",
                     "loose": "mostly {name}, box too big",
                     "multiple": "→ several chickens", "not_chicken": "→ not a chicken",
                     "unsure": "→ a hen, can't tell which",
                     "new": "→ new chicken", "no": "is not {name}"}

    def mutate(fn, label: str):
        with lock:
            store.snapshot(label)
            fn(store.data)
            # keep "how sure is each piece" current, so questions reflect every edit
            I.refresh_confidence(store.data, st["frags"])
            store.save()
            I.export(run_dir)
            return state()

    def reload_pieces():
        st["det"] = I.load_detections(run_dir, store.data)
        st["frags"] = _frags_from(st["det"])
        st["version"] += 1

    def do_undo() -> dict:
        with lock:
            label = store.undo()
            if label is None:
                raise ValueError("nothing to undo")
            reload_pieces()  # an undone cut changes which detections belong to which piece
            I.export(run_dir)
            out = state()
            out["undone"] = label
            return out

    def do_isolate(track: int, other: int) -> dict:
        """Cut `track` so the part overlapping `other` becomes its own piece."""
        with lock:
            cuts = I.overlap_cuts(st["det"], track, other)
            if not cuts:
                raise ValueError("this piece is already just the overlap")
            store.snapshot(f"cut out overlap of piece #{track} with #{other}", with_features=True)
            for f in reversed(cuts):  # cut the later point first, so both cuts apply to `track`
                st["det"] = I.split_piece(run_dir, store.data, st["det"], track, f)
            st["frags"] = _frags_from(st["det"])
            I.refresh_confidence(store.data, st["frags"])
            st["version"] += 1
            store.save()
            I.export(run_dir)
            return state()

    def do_apply_check(key: str, variant: str | None = None) -> dict:
        with lock:
            check = next((c for c in current_checks() if c["key"] == key), None)
            if check is None:
                raise ValueError("that check no longer applies")
            fix = check["alt"]["fix"] if variant and check.get("alt", {}).get("variant") == variant else check["fix"]
            store.snapshot(f"fix: {fix[:80]}", with_features=True)
            st["det"], to_place = K.apply_check(check, store.data, st["det"], run_dir, I.split_piece, variant=variant)
            st["frags"] = _frags_from(st["det"])
            for t in to_place:
                I.place_piece(store.data, st["frags"], t)
            I.refresh_confidence(store.data, st["frags"])
            st["version"] += 1
            store.save()
            I.export(run_dir)
            return state()

    def do_mark(track: int, marks: list[dict]) -> dict:
        """Marked filmstrip pictures -> cut the piece where the marks change, answer each part."""
        return do_marks([(track, marks)])

    def do_marks(items: list[tuple[int, list[dict]]], dismiss: str | None = None) -> dict:
        """Several pieces marked at once (a conflict's A and B): cut each where its marks change and
        answer every part, as one undo step. `dismiss`: the check these answers settle (from the
        Checks tab) - closed in the same step, so it can't come back as it was."""
        with lock:
            segs = [(t, *I.mark_segments(st["det"], t, [(m["frame"], m["label"]) for m in marks])) for t, marks in items]
            if not segs:
                raise ValueError("nothing marked")
            names = {f"c{c['id']}": (c["name"] or f"Chicken {c['id'] + 1}") for c in store.data["chickens"]}
            words = {"unsure": "can't tell", "not_chicken": "not a chicken", "multiple": "several", "keep": "as it was", **names}
            box = {"partial": " (only part)", "loose": " (box too big)"}
            word = lambda l: words.get(l.split(":")[0], l) + box.get(l.partition(":")[2], "")
            store.snapshot("; ".join(f"marked piece #{t}: " + " | ".join(word(l) for l in labels) for t, _, labels in segs),
                           with_features=any(cuts for _, cuts, _ in segs))
            for track, cuts, labels in segs:
                pieces = [track]
                for f in reversed(cuts):  # last cut first: each new piece is the tail of `track`
                    st["det"] = I.split_piece(run_dir, store.data, st["det"], track, f)
                    pieces.insert(1, store.data["splits"][-1]["new"])
                if cuts:
                    st["frags"] = _frags_from(st["det"])
                owner = I.chicken_of(store.data)
                for t, lab in zip(pieces, labels):
                    lab, _, flag = lab.partition(":")  # "c3:partial" = hen 3, only part of her / box too big
                    if lab == "keep":  # unpainted part (checks): cut off, but left as it was - no answer
                        continue
                    if lab.startswith("c"):
                        c = int(lab[1:])
                        verdict = flag if flag in I.BOX_FLAGS else "yes" if owner.get(t) == c else "moved"
                        I.apply_answer(store.data, t, c, verdict)
                    else:
                        I.apply_answer(store.data, t, None, lab)
            if dismiss:
                store.data.setdefault("dismissed_checks", []).append(str(dismiss))
            I.refresh_confidence(store.data, st["frags"])
            st["version"] += 1
            store.save()
            I.export(run_dir)
            return state()

    def do_split(track: int, frame: int) -> dict:
        with lock:
            I.check_cut(st["det"], track, frame, I.MIN_CUT_FRAMES)  # before the undo step, so a refused cut leaves none
            store.snapshot(f"cut piece #{track} at {clock(frame)}", with_features=True)
            st["det"] = I.split_piece(run_dir, store.data, st["det"], track, frame)
            st["frags"] = _frags_from(st["det"])  # identity.json isn't saved yet, so build from memory
            I.refresh_confidence(store.data, st["frags"])
            st["version"] += 1
            store.save()
            I.export(run_dir)
            return state()

    def _frags_from(det):
        picks = I.pd.read_csv(store.root / "crops.csv")
        feats = I.np.load(store.root / "features.npy")
        return I.fragments(det[det.track_id.isin(picks.track_id.unique())], picks, feats)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if path == "/api/questions":  # every open question (not just the next), e.g. to answer in bulk
                d = store.data
                irt = ir_tracks()
                return self._json([_question(q) for q in I.questions(d, extra=to_check()) if q["track"] not in irt])
            if path == "/api/state":
                return self._json(state())
            m = re.fullmatch(r"/api/check/(.+)", path)
            if m:
                c = next((c for c in current_checks() if c["key"] == unquote(m[1])), None)
                return self._json(_check_view(c) if c else {"error": "gone"})
            m = re.fullmatch(r"/api/gap/(\d+)/(\d+)/(\d+)", path)
            if m:
                return self._json({"track": int(m[1]),
                                   "frames": I.gap_frames(st["det"], int(m[1]), int(m[2]), int(m[3]))})
            m = re.fullmatch(r"/api/piece/(\d+)", path)
            if m and m[1] in store.data["fragments"]:
                return self._json(piece(int(m[1])))
            v = st["version"]
            m = re.fullmatch(r"/snap/(\d+)/(\d+)/([0-2])\.jpg", path)
            if m and (jpg := snapshot(int(m[1]), int(m[2]), int(m[3]), v)):
                return self._send(jpg, "image/jpeg")
            m = re.fullmatch(r"/strip/(\d+)/(\d+)\.jpg", path)
            if m and (jpg := strip_img(int(m[1]), int(m[2]), v, "wide=1" in self.path)):
                return self._send(jpg, "image/jpeg")
            m = re.fullmatch(r"/api/where/(\d+)", path)
            if m:
                return self._json(where_view(int(m[1])))
            m = re.fullmatch(r"/api/trace/(\d+)", path)
            if m:
                return self._json(trace_view(int(m[1]), st["version"]))
            m = re.fullmatch(r"/where/(\d+)\.mp4", path)
            if m and (mp4 := where_mp4(int(m[1]), v, dots_state["key"] if dots_data() else None, "wide=1" in self.path)):
                return self._send(mp4, "video/mp4")
            m = re.fullmatch(r"/clip/(\d+)\.mp4", path)
            direction = -1 if "dir=back" in self.path else 1 if "dir=forward" in self.path else 0
            sp = re.search(r"[?&]from=(\d+)&to=(\d+)", self.path)
            if m and (mp4 := clip_mp4(int(m[1]), v, direction, (int(sp[1]), int(sp[2])) if sp else None)):
                return self._send(mp4, "video/mp4")
            if path.startswith("/crop/"):
                p = crops / Path(unquote(path[6:])).name  # strip any directories
                if p.is_file():
                    return self._send(p.read_bytes(), "image/jpeg", cache=True)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
            if body.get("by") == "claude":  # answers Claude gave from the pictures: tagged, to spot-check
                # by identity, not position: a cut drops the piece's older answers before adding new ones
                before = {id(a) for a in store.data["answers"]}
                ok = self._post(body) is not False  # False = refused (error already sent)
                if ok:
                    with lock:
                        for a in store.data["answers"]:
                            if id(a) not in before:
                                a["by"] = "claude"
                        store.save()
                return
            self._post(body)

        def _post(self, body) -> bool:
            try:
                if self.path == "/api/name":
                    label = f"named {name_of(body['chicken'])} \"{str(body['name']).strip()}\""
                    return self._json(mutate(lambda d: I.rename(d, int(body["chicken"]), str(body["name"])), label))
                if self.path == "/api/merge":
                    label = f"merged {name_of(body['src'])} into {name_of(body['dst'])}"
                    return self._json(mutate(lambda d: I.merge_chickens(d, int(body["src"]), int(body["dst"])), label))
                if self.path == "/api/confirm_all":
                    label = f"all remaining pieces of {name_of(body['chicken'])} confirmed"
                    return self._json(mutate(lambda d: I.confirm_all(d, int(body["chicken"])), label))
                if self.path == "/api/dots":  # (re)compute dots.json in the background, ~1-2 min
                    if not dots_state["computing"]:
                        dots_state.update(computing=True, error=None)
                        threading.Thread(target=compute_dots, daemon=True).start()
                    return self._json({"computing": True})
                if self.path == "/api/answer":
                    ch = body.get("chicken")
                    v = str(body["verdict"])
                    label = f"piece #{body['track']} " + VERDICT_LABEL.get(v, v).format(name=name_of(ch))
                    return self._json(mutate(lambda d: I.apply_answer(
                        d, int(body["track"]), int(ch) if ch is not None else None, v), label))
                if self.path == "/api/answers":  # palette: many pictures painted at once, one undo step
                    items = [(int(a["track"]), None if a.get("chicken") is None else int(a["chicken"]), str(a["verdict"]))
                             for a in body["answers"]]
                    if not items:
                        raise ValueError("nothing painted")

                    def paint_all(d):
                        new_id = None  # every picture painted "+ new chicken" goes into the same new one
                        for t, ch, v in items:
                            if v == "new" and new_id is not None:
                                ch, v = new_id, "moved"
                            I.apply_answer(d, t, ch, v)
                            if v == "new":
                                new_id = d["answers"][-1]["chicken"]
                    return self._json(mutate(paint_all, f"painted {len(items)} pictures"))
                if self.path == "/api/resort":
                    return self._json(mutate(lambda d: I.resort(d, st["frags"]), "re-sort pieces"))
                if self.path == "/api/isolate":
                    return self._json(do_isolate(int(body["track"]), int(body["other"])))
                if self.path == "/api/apply_check":
                    return self._json(do_apply_check(str(body["key"]), body.get("variant")))
                if self.path == "/api/dismiss_check":  # "looks fine": your answer for that piece, one undo step
                    k = str(body["key"])
                    cks = current_checks()
                    check = next((c for c in cks if c["key"] == k), None)
                    # another open check on the same piece: don't confirm it yet (that one may still change it)
                    others = check is not None and any(c["track"] == check["track"] and c["key"] != k for c in cks)
                    said = []

                    def fn(d):
                        d.setdefault("dismissed_checks", []).append(k)
                        if check is not None and not others:
                            said.append(K.confirm_on_dismiss(check, d))
                    label = "looks fine"
                    if check is not None:
                        label += f": {check['title'][:60]}"
                    return self._json(mutate(fn, label))
                if self.path == "/api/tag_claude":  # repair: mark given answers as Claude's (latest match each)
                    with lock:
                        n = 0
                        for want in body["answers"]:
                            for a in reversed(store.data["answers"]):
                                if a["track"] == int(want["track"]) and a["verdict"] == want["verdict"] and not a.get("by"):
                                    a["by"] = "claude"; n += 1
                                    break
                        store.save()
                    return self._json({"tagged": n})
                if self.path == "/api/undo":
                    return self._json(do_undo())
                if self.path == "/api/mark":
                    return self._json(do_mark(int(body["track"]), list(body["marks"])))
                if self.path == "/api/marks":  # conflicts: A and B painted, one Apply
                    return self._json(do_marks([(int(p["track"]), list(p["marks"])) for p in body["pieces"]],
                                               dismiss=body.get("dismiss")))
                if self.path == "/api/split":
                    return self._json(do_split(int(body["track"]), int(body["frame"])))
            except (KeyError, ValueError, StopIteration) as e:
                self._json({"error": f"{e}"}, 400)
                return False
            finally:
                if self.path != "/api/dots":
                    schedule_dots()  # any change: refresh every hen's dot 1.5 s after the last one
            self._json({"error": "not found"}, 404)

    def reguess(weights: str, version: str | None = None, device: str | None = None) -> dict:
        """The given recogniser re-guesses the pieces you haven't confirmed (one undo step).

        Slow part (a crop per second through the model) runs outside the lock; the naming page stays
        usable meanwhile. Answers given during that time are kept: only still-unconfirmed pieces move."""
        pred, classes = R.predict_run(run_dir, Path(weights), device=device, video=store.data.get("video"))
        pred = R.load_predictions(run_dir, st["det"])  # joined to the pieces as they are now
        res = {}

        def fn(d):
            res.update(R.reguess(d, st["frags"], pred[0], pred[1], weights=weights, version=version))
        mutate(fn, f"recogniser {version or Path(weights).stem} re-guessed the unconfirmed pieces")
        return res

    Handler.reguess = staticmethod(reguess)
    return Handler


def serve(run_dir: Path, port: int = 8766, open_browser: bool = True) -> None:
    run_dir = Path(run_dir)
    if not (run_dir / "identity" / "identity.json").exists():
        raise SystemExit(f"No identity data in {run_dir}. Run `chicken-tracker identify {run_dir}` first.")
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(run_dir))
    url = f"http://127.0.0.1:{port}/"
    print(f"Naming chickens for {run_dir.name} at {url}  - Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
