"""Live progress of long steps, written to ``<run>/progress.json`` for the dashboard.

    with Progress(run_dir, "skeletons", total=frames, label="Skeletons") as p:
        for i, ... in enumerate(...):
            p.update(i + 1)

Writes are throttled (at most once a second) and atomic, so a page polling the file never
reads half a write. An exception marks the step as failed with its message.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

STEPS = {  # pipeline order, with names shown on the dashboard
    "detect": "Detect & track",
    "pieces": "Build pieces",
    "naming": "Name hens",
    "skeletons": "Skeletons",
    "pecks": "Pecks",
}
# Steps with no model of their own carry a method number: raise it when a change to the code
# should make older results count as out of date (the home page then marks them yellow).
METHOD_VERSIONS = {"pieces": 1, "pecks": 1}
_lock = threading.Lock()


def read(run_dir: Path) -> dict:
    p = Path(run_dir) / "progress.json"
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(run_dir: Path, step: str, rec: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        state = read(run_dir)
        state[step] = rec
        fd, tmp = tempfile.mkstemp(dir=run_dir, suffix=".progress")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))  # numpy numbers
            os.replace(tmp, run_dir / "progress.json")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class Progress:
    def __init__(self, run_dir: Path, step: str, total: int | None = None, label: str | None = None,
                 every_s: float = 1.0):
        self.run_dir, self.step, self.total = Path(run_dir), step, total
        self.label = label or STEPS.get(step, step)
        self.every_s, self._last, self.done_n = every_s, 0.0, 0
        self.started = time.time()
        self.phase = ""

    def _rec(self, state: str, **extra) -> dict:
        v = {"version": METHOD_VERSIONS[self.step]} if self.step in METHOD_VERSIONS else {}
        return {"label": self.label, "state": state, "done": self.done_n, "total": self.total,
                "phase": self.phase, "started": self.started, "updated": time.time(), **v, **extra}

    def preview_due(self, every_s: float = 4.0) -> bool:
        """True about every ``every_s`` seconds: time to draw a fresh picture (drawing costs time)."""
        now = time.time()
        if now - getattr(self, "_last_pic", 0.0) >= every_s:
            self._last_pic = now
            return True
        return False

    def preview(self, img, width: int = 640) -> None:
        """Save a picture of the work in progress (BGR image) as preview/<step>.jpg for the dashboard."""
        import cv2

        if img is None or not getattr(img, "size", 0):
            return
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        d = self.run_dir / "preview"
        d.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            tmp = d / f".{self.step}.jpg"
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, d / f"{self.step}.jpg")

    def __enter__(self) -> "Progress":
        _write(self.run_dir, self.step, self._rec("running"))
        return self

    def update(self, done: int, total: int | None = None, phase: str | None = None, force: bool = False) -> None:
        self.done_n = int(done)
        if total is not None:
            self.total = int(total)
        if phase is not None and phase != self.phase:
            self.phase, force = phase, True
        now = time.time()
        if force or now - self._last >= self.every_s:
            self._last = now
            _write(self.run_dir, self.step, self._rec("running"))

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            if self.total:
                self.done_n = self.total
            _write(self.run_dir, self.step, self._rec("done", finished=time.time()))
        else:
            _write(self.run_dir, self.step, self._rec("failed", finished=time.time(), error=f"{exc_type.__name__}: {exc}"))
        return False
