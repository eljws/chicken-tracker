"""Tiny local web app for reviewing candidate chicken boxes (stdlib only)."""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

from .labeling import LabelStore


def make_handler(store: LabelStore, objects: bool = False, outliner=None, pose: bool = False):
    """objects=True: the feeder/water pass over frames whose chickens are already reviewed.

    outliner: an ``outline.Outliner`` for click-to-outline (None = off).
    """
    page = (resources.files("chicken_tracker") / "static" /
            ("pose.html" if pose else "objects.html" if objects else "review.html")).read_bytes()
    key = "pose_status" if pose else "obj_status" if objects else "status"
    auto_cache: dict[str, list] = {}  # frame name -> outlines of its suggestions (computed once)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _safe_name(self, raw: str) -> str | None:
            name = unquote(raw)
            return name if name in store.items else None  # only known images, no path tricks

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if self.path == "/api/config":
                from .labeling import KEYPOINTS, SKELETON
                return self._json({"outline": outliner is not None, "keypoints": KEYPOINTS, "skeleton": SKELETON})
            if self.path == "/api/items":
                with store._lock:
                    store._reload()  # another page may have saved meanwhile
                items = [{"name": k, **v} for k, v in sorted(
                    store.items.items(), key=lambda kv: (-kv[1].get("priority", 0), kv[1]["video"], kv[1]["frame"]))
                    if key in v]
                return self._json({"items": items, "counts": store.counts(key)})
            if self.path.startswith("/img/"):
                name = self._safe_name(self.path[5:])
                if name is None:
                    return self._json({"error": "not found"}, 404)
                return self._send((store.images / name).read_bytes(), "image/jpeg")
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path.startswith("/api/auto/"):
                # outline every suggested box of a frame up front, so the page opens already outlined
                name = self._safe_name(self.path[len("/api/auto/"):])
                if name is None or outliner is None:
                    return self._json({"error": "not available"}, 404)
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                boxes = [[float(v) for v in b[:4]] for b in body.get("boxes", [])][:60]
                ck = name + json.dumps(boxes)
                if ck not in auto_cache:
                    try:
                        auto_cache[ck] = outliner.outline_boxes(store.images / name, boxes)
                    except Exception as e:
                        return self._json({"error": f"outline failed: {e}"}, 500)
                return self._json({"outlines": auto_cache[ck]})
            if self.path.startswith("/api/outline/"):
                name = self._safe_name(self.path[len("/api/outline/"):])
                if name is None or outliner is None:
                    return self._json({"error": "not available"}, 404)
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                pts = [[float(x), float(y)] for x, y in body.get("points", [])][:20]
                labels = [1 if int(v) else 0 for v in body.get("labels", [])][:len(pts)]
                if not pts or len(labels) != len(pts) or labels[0] != 1:
                    return self._json({"error": "first point must be on the bird"}, 400)
                try:
                    out = outliner.outline(store.images / name, pts, labels)
                except Exception as e:  # keep the page usable; drag still works
                    return self._json({"error": f"outline failed: {e}"}, 500)
                return self._json(out or {"error": "nothing found there"}, 200 if out else 422)
            if not self.path.startswith("/api/item/"):
                return self._json({"error": "not found"}, 404)
            name = self._safe_name(self.path[len("/api/item/"):])
            if name is None:
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            status = body.get("status")
            if status not in ("done", "skipped", "pending"):
                return self._json({"error": "bad status"}, 400)
            if pose:
                store.update_pose(name, status, body.get("poses", []))
            elif objects:
                store.update_objects(name, status, body.get("objects", []), body.get("rejected", []))
            else:
                store.update(name, status, body.get("boxes", []), body.get("outlines"))
            self._json({"ok": True, "counts": store.counts(key)})

    return Handler


def serve(root: Path, port: int = 8765, open_browser: bool = True, objects: bool = False,
          outline: bool = True, device: str | None = None, pose: bool = False) -> None:
    store = LabelStore(root)
    if not store.items:
        raise SystemExit(f"No frames in {root}. Run `chicken-tracker mine` first.")
    if pose:
        store.init_pose()
        outline = False
    key = "pose_status" if pose else "obj_status" if objects else "status"
    c = store.counts(key)
    if objects and not sum(c.values()):
        raise SystemExit(f"No feeder/water suggestions yet. Run `chicken-tracker mine --objects -d {root}` first.")
    outliner = None
    if outline and not objects:
        import threading

        from .outline import Outliner

        outliner = Outliner(device=device)
        threading.Thread(target=outliner.warm_up, daemon=True).start()  # load SAM while you look
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, objects, outliner, pose))
    url = f"http://127.0.0.1:{port}/"
    what = "chicken skeletons in" if pose else "feeder/water in" if objects else "chickens in"
    print(f"Reviewing {what} {sum(c.values())} frames ({c['pending']} pending) at {url}  - Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"\nSaved. {store.counts(key)}")
