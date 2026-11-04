"""One web server for everything: a calendar of days, and each clip's player, naming page and
clips under one address, plus progress, frame review and skeleton labelling.

    /                                  home: calendar of days, clips per day, links
    /progress/                         live progress of every run
    /review/  /pose/                   label frames / skeletons (whole dataset)
    /run/<run>/player/                 a clip's player        (<run> = folder under output/,
    /run/<run>/naming/                 a clip's naming page     e.g. 2026-11-02/143113-150338_ch04)
    /run/<run>/clips/                  short clips of set-aside pieces, if made

The pages are the same ones the single-purpose servers show: each request is handed to that
page's own handler with the prefix taken off, so they work identically here and on their own.
A clip's handler is made the first time it's opened and then kept.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import quote, unquote

from . import dashboard as D

APPS = ("player", "naming", "clips")


def browser_video(run: Path) -> Path | None:
    """The browser copy the player plays (made by `player --encode-only`), newest if several."""
    vids = sorted((run / "browser").glob("video_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return vids[0] if vids else None


def _naming_counts(run: Path) -> dict | None:
    p = run / "identity" / "identity.json"
    if not p.exists():
        return None
    from . import identity as I

    try:
        d = json.loads(p.read_text())
        return {"conflicts": len(I.conflicts(d)), "questions": len(I.questions(d)),
                "confirmed": len(I.confirmed(d)), "pieces": len(d.get("fragments", {}))}
    except Exception:  # half-written or older format: just don't show counts
        return None


def run_card(root: Path, run: Path) -> dict:
    s = D.run_status(root, run)
    rel = s["run"]
    base = f"run/{quote(rel)}/"
    pic = next((p for p in (run / "preview" / "detect.jpg", run / "heatmap.png") if p.exists()), None)
    s.update({
        "links": {
            "player": base + "player/" if browser_video(run) else None,
            "naming": base + "naming/" if (run / "identity" / "identity.json").exists() else None,
            "clips": base + "clips/" if (run / "identity" / "clips" / "index.html").exists() else None,
        },
        "thumb": f"{base}thumb?v={int(pic.stat().st_mtime)}" if pic else None,
        "counts": _naming_counts(run),
    })
    return s


def make_handler(root: Path, dataset: Path | None = None, device: str | None = None):
    root = Path(root).resolve()
    home = lambda: (resources.files("chicken_tracker") / "static" / "home.html").read_bytes()  # fresh each load
    lock = threading.Lock()
    apps: dict = {}  # (run, app) or "progress"/"review"/"pose" -> handler class

    def find_run(rel: str) -> Path | None:
        p = (root / rel).resolve()
        return p if root in p.parents and (p / "meta.json").exists() else None

    def app_handler(key, make):
        with lock:
            if key not in apps:
                apps[key] = make()
            return apps[key]

    def run_app(run: Path, app: str):
        def make():
            if app == "player":
                from . import player

                meta = json.loads((run / "meta.json").read_text())
                return player.make_handler(run, browser_video(run), meta["video"]["path"])
            if app == "naming":
                from . import identity_server

                return identity_server.make_handler(run)
            return None
        return app_handler((str(run), app), make)

    def dataset_app(name: str):
        def make():
            from . import review_server
            from .labeling import LabelStore

            store = LabelStore(dataset)
            if name == "pose":
                store.init_pose()
                return review_server.make_handler(store, pose=True)
            from .outline import Outliner

            outliner = Outliner(device=device)
            threading.Thread(target=outliner.warm_up, daemon=True).start()
            return review_server.make_handler(store, outliner=outliner)
        return app_handler(name, make)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, to: str):
            self.send_response(301)
            self.send_header("Location", to)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _delegate(self, cls, subpath: str):
            """Answer this request with another page's handler, as if it were asked for `subpath`."""
            h = cls.__new__(cls)
            h.__dict__.update(self.__dict__)  # same connection, headers and body
            h.path = subpath
            method = getattr(h, "do_" + self.command, None)
            if method is None:
                return self._send(b"not supported", "text/plain", 405)
            method()
            self.close_connection = h.close_connection

        def _file(self, p: Path):
            """A file with byte ranges (Safari won't play a video without them)."""
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
            if m and (m[1] or m[2]):
                start, end = (int(m[1]), int(m[2]) if m[2] else size - 1) if m[1] else (max(0, size - int(m[2])), size - 1)
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(p, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0 and (chunk := f.read(min(1 << 20, left))):
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _route(self):
            raw = self.path
            path, _, query = raw.partition("?")
            q = f"?{query}" if query else ""
            if path in ("/", "/index.html"):
                return self._send(home(), "text/html; charset=utf-8")
            if path == "/api/home":
                runs = [run_card(root, r) for r in D.find_runs(root)]
                extra = {}
                if dataset is not None and (Path(dataset) / "labels.json").exists():
                    items = json.loads((Path(dataset) / "labels.json").read_text())
                    extra = {"review_pending": sum(v.get("status") == "pending" for v in items.values()),
                             "pose_pending": sum(v.get("pose_status") == "pending" for v in items.values())}
                return self._send(json.dumps({"runs": runs, "now": time.time(), **extra}).encode(), "application/json")
            # shared pages: /progress/ /review/ /pose/
            for name in ("progress", "review", "pose"):
                if path == f"/{name}":
                    return self._redirect(f"/{name}/{q}")
                if path.startswith(f"/{name}/"):
                    if name != "progress" and dataset is None:
                        return self._send(b"no dataset folder given", "text/plain", 404)
                    cls = app_handler("progress", lambda: D.make_handler(root)) if name == "progress" \
                        else dataset_app(name)
                    return self._delegate(cls, path[len(name) + 1:] + q)
            # one clip: /run/<run>/<app>/...
            m = re.fullmatch(r"/run/(.+?)/(player|naming|clips|thumb)(/.*)?", path)
            if m:
                run = find_run(unquote(m[1]))
                if run is None:
                    return self._send(b"no such run", "text/plain", 404)
                app, rest = m[2], m[3]
                if app == "thumb":
                    p = next((p for p in (run / "preview" / "detect.jpg", run / "heatmap.png") if p.exists()), None)
                    return self._file(p) if p else self._send(b"", "image/jpeg", 404)
                if rest is None:
                    return self._redirect(f"{path}/{q}")
                if app == "clips":
                    d = (run / "identity" / "clips").resolve()
                    p = (d / unquote(rest.lstrip("/") or "index.html")).resolve()
                    if d not in p.parents or not p.is_file():
                        return self._send(b"not found", "text/plain", 404)
                    return self._file(p)
                if app == "player" and browser_video(run) is None:
                    return self._send(b"No browser copy of this video yet: run "
                                      b"`chicken-tracker player RUN --encode-only`.", "text/plain", 404)
                if app == "naming" and not (run / "identity" / "identity.json").exists():
                    return self._send(b"Not named yet: run `chicken-tracker identify RUN`.", "text/plain", 404)
                return self._delegate(run_app(run, app), rest + q)
            self._send(b'{"error":"not found"}', "application/json", 404)

        def do_GET(self):
            self._route()

        def do_HEAD(self):
            self._route()

        def do_POST(self):
            self._route()

    return Handler


def serve(root: Path = Path("output"), dataset: Path | None = Path("dataset"), port: int = 8760,
          open_browser: bool = True, device: str | None = None) -> None:
    if dataset is not None and not (Path(dataset) / "labels.json").exists():
        dataset = None
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(root), dataset, device))
    url = f"http://127.0.0.1:{port}/"
    print(f"Chicken tracker home for {root}/ at {url}  - Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
