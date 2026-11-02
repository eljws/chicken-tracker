"""Local web app: name chickens, merge groups, answer "is this Rocky?", fix conflicts, cut pieces."""

from __future__ import annotations

import functools
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

from . import identity as I

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
    def strip_img(track: int, frame: int, version: int) -> bytes | None:
        return I.piece_frame(store.data["video"], st["det"], track, frame, store.data["fps"])

    def mid_crop(t: int) -> str:
        c = store.data["fragments"][str(t)]["crops"]
        return c[len(c) // 2] if c else ""

    def piece(t: int) -> dict:
        f = store.data["fragments"][str(t)]
        return {"track": t, "start": f["start"], "duration_s": round(f["end"] - f["start"], 1),
                "sure": t in I.confirmed(store.data), "strip": I.strip_frames(st["det"], t, STRIP_LEN)}

    def state() -> dict:
        d = store.data
        fr = d["fragments"]
        chickens = []
        for c in d["chickens"]:
            tr = sorted(c["tracks"], key=lambda t: -fr[str(t)]["n"])
            chickens.append({
                "id": c["id"], "name": c["name"],
                "minutes": round(sum(fr[str(t)]["end"] - fr[str(t)]["start"] for t in tr) / 60, 1),
                "fragments": [{"track": t, "crop": mid_crop(t), "sure": t in I.confirmed(d),
                               "flag": I.box_flag(d, t)} for t in tr],
            })
        qs = I.questions(d)
        aside = {reason: [{"track": t, "crop": mid_crop(t)} for t in d.get(key, [])]
                 for reason, key in I.EXCLUDED_KEYS.items()}
        cf = I.conflicts(d)
        return {"chickens": chickens, "aside": aside, "fps": d["fps"], "version": st["version"],
                "undo": store.history(),
                "conflicts": len(cf), "conflict": _conflict(cf[0]) if cf else None,
                "questions": len(qs), "next": _question(qs[0]) if qs else None,
                "answered": len(d["answers"])}

    def _question(q: dict) -> dict:
        c = next(c for c in store.data["chickens"] if c["id"] == q["chicken"])
        ref = sorted((t for t in c["tracks"] if t != q["track"]),
                     key=lambda t: -store.data["fragments"][str(t)]["n"])[:8]
        return {**q, **piece(q["track"]), "ref_crops": [x for x in map(mid_crop, ref) if x],
                "name": c["name"] or f"Chicken {c['id'] + 1}"}

    def _conflict(cf: dict) -> dict:
        c = next(c for c in store.data["chickens"] if c["id"] == cf["chicken"])
        fr = store.data["fragments"]
        lo = max(fr[str(cf["a"])]["start"], fr[str(cf["b"])]["start"])
        hi = min(fr[str(cf["a"])]["end"], fr[str(cf["b"])]["end"])
        pieces = []
        for t, other in ((cf["a"], cf["b"]), (cf["b"], cf["a"])):
            p = piece(t)
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
                     "new": "→ new chicken", "no": "is not {name}"}

    def mutate(fn, label: str):
        with lock:
            store.snapshot(label)
            fn(store.data)
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

    def do_split(track: int, frame: int) -> dict:
        with lock:
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
            if path == "/api/state":
                return self._json(state())
            m = re.fullmatch(r"/api/piece/(\d+)", path)
            if m and m[1] in store.data["fragments"]:
                return self._json(piece(int(m[1])))
            v = st["version"]
            m = re.fullmatch(r"/snap/(\d+)/(\d+)/([0-2])\.jpg", path)
            if m and (jpg := snapshot(int(m[1]), int(m[2]), int(m[3]), v)):
                return self._send(jpg, "image/jpeg")
            m = re.fullmatch(r"/strip/(\d+)/(\d+)\.jpg", path)
            if m and (jpg := strip_img(int(m[1]), int(m[2]), v)):
                return self._send(jpg, "image/jpeg")
            if path.startswith("/crop/"):
                p = crops / Path(unquote(path[6:])).name  # strip any directories
                if p.is_file():
                    return self._send(p.read_bytes(), "image/jpeg", cache=True)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
            try:
                if self.path == "/api/name":
                    label = f"named {name_of(body['chicken'])} \"{str(body['name']).strip()}\""
                    return self._json(mutate(lambda d: I.rename(d, int(body["chicken"]), str(body["name"])), label))
                if self.path == "/api/merge":
                    label = f"merged {name_of(body['src'])} into {name_of(body['dst'])}"
                    return self._json(mutate(lambda d: I.merge_chickens(d, int(body["src"]), int(body["dst"])), label))
                if self.path == "/api/answer":
                    ch = body.get("chicken")
                    v = str(body["verdict"])
                    label = f"piece #{body['track']} " + VERDICT_LABEL.get(v, v).format(name=name_of(ch))
                    return self._json(mutate(lambda d: I.apply_answer(
                        d, int(body["track"]), int(ch) if ch is not None else None, v), label))
                if self.path == "/api/resort":
                    return self._json(mutate(lambda d: I.resort(d, st["frags"]), "re-sort pieces"))
                if self.path == "/api/isolate":
                    return self._json(do_isolate(int(body["track"]), int(body["other"])))
                if self.path == "/api/undo":
                    return self._json(do_undo())
                if self.path == "/api/split":
                    return self._json(do_split(int(body["track"]), int(body["frame"])))
            except (KeyError, ValueError, StopIteration) as e:
                return self._json({"error": f"{e}"}, 400)
            self._json({"error": "not found"}, 404)

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
