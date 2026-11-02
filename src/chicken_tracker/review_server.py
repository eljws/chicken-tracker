"""Tiny local web app for reviewing candidate chicken boxes (stdlib only)."""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

from .labeling import LabelStore


def make_handler(store: LabelStore):
    page = (resources.files("chicken_tracker") / "static" / "review.html").read_bytes()

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
            if self.path == "/api/items":
                items = [{"name": k, **v} for k, v in sorted(
                    store.items.items(), key=lambda kv: (kv[1]["video"], kv[1]["frame"]))]
                return self._json({"items": items, "counts": store.counts()})
            if self.path.startswith("/img/"):
                name = self._safe_name(self.path[5:])
                if name is None:
                    return self._json({"error": "not found"}, 404)
                return self._send((store.images / name).read_bytes(), "image/jpeg")
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.startswith("/api/item/"):
                return self._json({"error": "not found"}, 404)
            name = self._safe_name(self.path[len("/api/item/"):])
            if name is None:
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            status = body.get("status")
            if status not in ("done", "skipped", "pending"):
                return self._json({"error": "bad status"}, 400)
            store.update(name, status, body.get("boxes", []))
            self._json({"ok": True, "counts": store.counts()})

    return Handler


def serve(root: Path, port: int = 8765, open_browser: bool = True) -> None:
    store = LabelStore(root)
    if not store.items:
        raise SystemExit(f"No frames in {root}. Run `chicken-tracker mine` first.")
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    url = f"http://127.0.0.1:{port}/"
    c = store.counts()
    print(f"Reviewing {len(store.items)} frames ({c['pending']} pending) at {url}  - Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"\nSaved. {store.counts()}")
