"""Progress dashboard: every analysed run, which steps are done, live bars for running ones.

Reads only files in the output folder (``progress.json`` written by the steps, plus the
results each step leaves behind), so it shows work started from any terminal.
"""

from __future__ import annotations

import json
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

import pandas as pd

from . import progress as P

DONE_IF = {  # a step counts as done if it left this behind (for runs made before progress.json)
    "detect": "detections.csv",
    "pieces": "identity/identity.json",
    "skeletons": "poses.csv",
    "pecks": "pecks.csv",
}
STALE_S = 120  # a "running" step not heard from for this long probably stopped


def find_runs(root: Path) -> list[Path]:
    root = Path(root)
    runs = {p.parent for pat in ("*/meta.json", "*/*/meta.json", "*/progress.json", "*/*/progress.json")
            for p in root.glob(pat)}
    return sorted(runs, key=lambda r: str(r.relative_to(root)))


def _named(run: Path) -> dict | None:
    p = run / "identity" / "identity.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    names = [c["name"] for c in d.get("chickens", []) if c.get("name")]
    auto = d.get("auto_named")
    return {"names": names, "auto": auto, "answers": len(d.get("answers", []))} if names or auto else None


def run_status(root: Path, run: Path) -> dict:
    now = time.time()
    prog = P.read(run)
    rel = str(run.relative_to(root))
    m = re.search(r"(\d{4}-\d{2}-\d{2})", rel)
    clip = re.search(r"(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", run.name)
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    if not m and meta:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", meta.get("video", {}).get("path", ""))
    steps = []
    for key, label in P.STEPS.items():
        rec = dict(prog.get(key, {}))
        if not rec:
            done = (_named(run) is not None) if key == "naming" else (run / DONE_IF[key]).exists()
            rec = {"label": label, "state": "done" if done else "todo"}
        elif rec.get("state") == "running" and now - rec.get("updated", now) > STALE_S:
            rec["state"] = "stalled"
        rec["key"] = key
        pic = run / "preview" / f"{key}.jpg"
        if pic.exists():  # mtime in the URL: the browser fetches a fresh picture only when it changed
            rec["picture"] = f"preview/{rel}/preview/{key}.jpg?v={int(pic.stat().st_mtime * 1000)}"
        steps.append(rec)
    hens = []
    if (run / "chickens.csv").exists():
        try:
            c = pd.read_csv(run / "chickens.csv")
            for _, r in c.iterrows():
                hens.append({"name": r.chicken, "visible_min": round(float(r.get("visible_s", 0)) / 60, 1),
                             "pecks": int(r["pecks_at_feeder"]) if "pecks_at_feeder" in c else None})
        except Exception:
            pass
    return {"run": rel, "date": m[1] if m else "", "from": f"{clip[1]}:{clip[2]}" if clip else "",
            "to": f"{clip[4]}:{clip[5]}" if clip else "", "steps": steps, "hens": hens, "naming": _named(run),
            "minutes": round(meta["video"]["frame_count"] / meta["video"]["fps"] / 60, 1) if meta else None}


def make_handler(root: Path):
    root = Path(root)
    page = lambda: (resources.files("chicken_tracker") / "static" / "dashboard.html").read_bytes()  # fresh each load

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # pictures are versioned in their URL, so the browser may keep them (no flicker on refresh)
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(page(), "text/html; charset=utf-8")
            if path.startswith("/preview/") and path.endswith(".jpg"):
                from urllib.parse import unquote
                p = (root / unquote(path[len("/preview/"):])).resolve()
                rel = p.parent.relative_to(root.resolve()) if root.resolve() in p.parents else None
                if rel is None or p.parent.name != "preview" or p.stem not in P.STEPS or not p.exists():
                    return self._send(b"not found", "text/plain", 404)
                return self._send(p.read_bytes(), "image/jpeg", cache=True)
            if path == "/api/runs":
                runs = [run_status(root, r) for r in find_runs(root)]
                return self._send(json.dumps({"runs": runs, "now": time.time()}).encode(), "application/json")
            self._send(b'{"error":"not found"}', "application/json", 404)

    return Handler


def serve(root: Path = Path("output"), port: int = 8760, open_browser: bool = True) -> None:
    root = Path(root)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    url = f"http://127.0.0.1:{port}/"
    print(f"Progress dashboard for {root}/ at {url}  - Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
