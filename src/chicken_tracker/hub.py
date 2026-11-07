"""One web server for everything: a calendar of days, and each clip's player, naming page and
clips under one address, plus progress, frame review and skeleton labelling.

    /                                  home: calendar of days, clips per day, links
    /progress/                         live progress of every run
    /review/  /pose/                   label frames / skeletons (whole dataset)
    /run/<run>/player/                 a clip's player        (<run> = folder under output/,
    /run/<run>/naming/                 a clip's naming page     e.g. 2026-11-02/143113-150338_ch04)
    /run/<run>/clips/                  short clips of set-aside pieces, if made
    /metrics/                          per-hen charts for a day: distance, activity, runs
    /hens/                             every hen's pictures from all clips, grouped, to look her over
    /timeline/                         all recordings as one video, dawn and dusk marked, day = dawn to dawn

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
from . import versions as V

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
        cf = run / "identity" / "conflicts.json"  # frame-exact count, written by every save
        n = json.loads(cf.read_text())["conflicts"] if cf.exists() and cf.stat().st_mtime >= p.stat().st_mtime \
            else len(I.conflicts(d))
        out = {"conflicts": n, "questions": len(I.questions(d)),
               # done = answered by you or set aside, as the naming page's Name & merge count
               "confirmed": len({int(t) for t in d.get("fragments", {})} & (I.confirmed(d) | I.excluded(d))),
               "pieces": len(d.get("fragments", {}))}
        st = run / "identity" / "status.json"  # checks and "?" questions: known once the naming page has run
        if st.exists() and st.stat().st_mtime >= p.stat().st_mtime - 1:
            s = json.loads(st.read_text())
            out.update(checks=s.get("checks"), questions=s.get("questions", out["questions"]), conflicts=s.get("conflicts", n))
        return out
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
    ver = V.of_run(run)
    for step in s["steps"]:
        if step["key"] in ver:
            step["ver"] = ver[step["key"]]
    return s


def make_handler(root: Path, dataset: Path | None = None, device: str | None = None):
    root = Path(root).resolve()
    home = lambda: (resources.files("chicken_tracker") / "static" / "home.html").read_bytes()  # fresh each load
    lock = threading.Lock()
    apps: dict = {}  # (run, app) or "progress"/"review"/"pose" -> handler class

    # recompute queue: one job at a time, in the order asked (they share the GPU)
    jobs: list[dict] = []          # waiting, and the one running first
    job_log: list[dict] = []       # finished, newest last (shown as the step's tooltip)
    jobs_cv = threading.Condition()

    def queued(rel: str) -> dict[str, str]:
        with jobs_cv:
            return {j["step"]: j["state"] for j in jobs if j["run"] == rel}

    def enqueue(rel: str, step: str) -> bool:
        with jobs_cv:
            if any(j["run"] == rel and j["step"] == step for j in jobs):
                return False
            jobs.append({"run": rel, "step": step, "state": "queued", "asked": time.time()})
            jobs_cv.notify()
            return True

    def run_job(j: dict) -> None:
        import subprocess
        import sys

        run = root / j["run"]
        exe = str(Path(sys.executable).parent / "chicken-tracker")
        dev = ["--device", device] if device else []
        cwd = str(Path.cwd())
        if j["step"] == "naming":
            from . import records

            cur = next((m for m in records.models("recognizer") if m["status"] == "current"), None)
            if cur is None:
                raise RuntimeError("no recogniser in use")
            cls = run_app(run, "naming")
            j["result"] = cls.reguess(cur["path"], cur["version"], device)
            return
        steps = [[exe, "pose", str(run), *dev], [exe, "pecks", str(run)]] if j["step"] == "skeletons" \
            else [[exe, "pecks", str(run)]]
        for cmd in steps:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else f"exit {r.returncode}")

    def worker():
        while True:
            with jobs_cv:
                while not jobs:
                    jobs_cv.wait()
                j = jobs[0]
                j["state"] = "running"
            try:
                run_job(j)
                j["state"] = "done"
            except Exception as e:  # noqa: BLE001 - shown on the page
                j.update(state="failed", error=f"{type(e).__name__}: {e}")
            with jobs_cv:
                jobs.pop(0)
                job_log.append({**j, "finished": time.time()})
                del job_log[:-50]

    threading.Thread(target=worker, daemon=True).start()

    def find_run(rel: str) -> Path | None:
        p = (root / rel).resolve()
        return p if root in p.parents and ((p / "meta.json").exists() or (p / "progress.json").exists()) else None  # running: live picture

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
            if path == "/api/recompute" and self.command == "POST":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
                added = 0
                for item in body.get("items", []):
                    rel, step = str(item.get("run", "")), str(item.get("step", ""))
                    run = find_run(rel)
                    if run is None or step not in V.UPDATABLE:
                        return self._send(json.dumps({"error": f"can't recompute {step} of {rel}"}).encode(),
                                          "application/json", 400)
                    added += enqueue(str(run.relative_to(root)), step)
                return self._send(json.dumps({"queued": added}).encode(), "application/json")
            if path == "/api/home":
                runs = [run_card(root, r) for r in D.find_runs(root)]
                last = {}
                for j in job_log:
                    last[(j["run"], j["step"])] = j
                for r in runs:
                    q = queued(r["run"])
                    for st_ in r["steps"]:
                        if st_["key"] in q:
                            st_["job"] = q[st_["key"]]
                        elif (r["run"], st_["key"]) in last and last[(r["run"], st_["key"])]["state"] == "failed":
                            st_["job_error"] = last[(r["run"], st_["key"])].get("error")
                extra = {}
                if dataset is not None and (Path(dataset) / "labels.json").exists():
                    items = json.loads((Path(dataset) / "labels.json").read_text())
                    extra = {"review_pending": sum(v.get("status") == "pending" for v in items.values()),
                             "pose_pending": sum(v.get("pose_status") == "pending" for v in items.values())}
                return self._send(json.dumps({"runs": runs, "now": time.time(), **extra}).encode(), "application/json")
            if path == "/metrics":
                return self._redirect(f"/metrics/{q}")
            if path == "/metrics/":
                return self._send((resources.files("chicken_tracker") / "static" / "metrics.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if path == "/metrics/api":
                from urllib.parse import parse_qs
                from . import metrics as Mx
                all_runs = D.find_runs(root)
                days = sorted({str(r.relative_to(root)).split("/")[0] for r in all_runs
                               if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.relative_to(root)).split("/")[0])})
                day = (parse_qs(query).get("day") or [days[-1] if days else ""])[0]
                body = Mx.day_metrics(root, all_runs, day) if day else {}
                return self._send(json.dumps({"days": days, **body}).encode(), "application/json")
            if path == "/timeline":
                return self._redirect(f"/timeline/{q}")
            if path == "/timeline/":
                return self._send((resources.files("chicken_tracker") / "static" / "timeline.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if path == "/timeline/api":  # GET: the timeline + scan state; POST: scan the recordings again
                from . import timeline as TL
                if self.command == "POST":
                    TL.scan_in_background(root)
                return self._send(json.dumps({"timeline": TL.load(root), "scan": TL.scan_status()}).encode(), "application/json")
            if path == "/timeline/overlay.js":
                return self._send((resources.files("chicken_tracker") / "static" / "overlay.js").read_bytes(),
                                  "text/javascript; charset=utf-8")
            if path == "/timeline/hens":  # the hens of every processed clip of one dawn-to-dawn day
                from urllib.parse import parse_qs
                from . import timeline as TL
                day = (parse_qs(query).get("day") or [""])[0]
                return self._send(json.dumps(TL.day_hens(root, day)).encode(), "application/json")
            m = re.fullmatch(r"/timeline/video/(\d{4}-\d{2}-\d{2})/([\w\-]+\.mp4)", path)
            if m:  # only recordings listed in the timeline are served
                from . import timeline as TL
                tl = TL.load(root) or {"clips": []}
                c = next((c for c in tl["clips"] if c["rel"] == f"{m[1]}/{m[2]}"), None)
                return self._file(Path(c["path"])) if c and Path(c["path"]).is_file() else self._send(b"not found", "text/plain", 404)
            if path == "/hens":
                return self._redirect(f"/hens/{q}")
            if path == "/hens/":
                return self._send((resources.files("chicken_tracker") / "static" / "hens.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if path == "/hens/sizes":  # GET: the saved size calibration (+ stale?); POST: learn it again
                from . import sizes as SZ
                runs = D.find_runs(root)
                try:
                    cal = SZ.build(root, runs) if self.command == "POST" or SZ.load(root) is None else SZ.load(root)
                except ValueError as e:
                    return self._send(json.dumps({"error": str(e)}).encode(), "application/json")
                newest = max(((r / "identity" / "identity.json").stat().st_mtime for r in runs
                              if (r / "identity" / "identity.json").exists()), default=0)
                return self._send(json.dumps({"cal": cal, "stale": newest > cal["built"] + 1}).encode(), "application/json")
            if path == "/hens/api":
                from . import hens as Hn
                return self._send(json.dumps(Hn.hens_data(root, D.find_runs(root))).encode(), "application/json")
            m = re.fullmatch(r"/hens/crop/(.+)/([^/]+)", path)
            if m:
                from . import hens as Hn
                p = Hn.crop_path(root, unquote(m[1]), unquote(m[2]))
                return self._send(p.read_bytes(), "image/jpeg", cache=True) if p else self._send(b"", "text/plain", 404)
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

        def _allowed(self) -> bool:
            """The outside port (server.gate set) needs the login; the local one doesn't."""
            gate = getattr(self.server, "gate", None)
            if gate is None:
                return True
            ip = self.client_address[0]
            if gate.locked(ip):
                self._send(b"Too many wrong passwords - try again in 10 minutes.", "text/plain", 429)
                return False
            if gate.check(self.headers.get("Authorization"), ip):
                return True
            body = b"Login needed."
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Chicken tracker", charset="UTF-8"')
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def do_GET(self):
            if self._allowed():
                self._route()

        def do_HEAD(self):
            if self._allowed():
                self._route()

        def do_POST(self):
            if self._allowed():
                self._route()

    return Handler


def serve(root: Path = Path("output"), dataset: Path | None = Path("dataset"), port: int = 8760,
          open_browser: bool = True, device: str | None = None, public_port: int | None = None) -> None:
    """Local page on 127.0.0.1:port. With public_port, the same pages (same handler, so the same
    naming data and undo history) also on every network address at public_port, behind the login."""
    if dataset is not None and not (Path(dataset) / "labels.json").exists():
        dataset = None
    handler = make_handler(Path(root), dataset, device)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Chicken tracker home for {root}/ at {url}  - Ctrl+C to stop")
    if public_port:
        from .auth import AUTH_FILE, Gate

        gate = Gate()  # with no password set yet it lets nobody in; setting one takes effect at once
        outside = ThreadingHTTPServer(("0.0.0.0", public_port), handler)
        outside.gate = gate
        threading.Thread(target=outside.serve_forever, daemon=True).start()
        print(f"Also on port {public_port} for other devices / the router, login needed" +
              ("" if gate.ready else f" - NO PASSWORD SET YET, nobody can log in: run `chicken-tracker password` ({AUTH_FILE})"))
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
