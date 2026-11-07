"""Which version made each step of a clip, and whether that's the latest.

Model steps name their model (detect: detector, naming: recogniser, skeletons: skeleton model);
pieces and pecks carry a method number (progress.METHOD_VERSIONS). The home page shows these on
each step: green = made with what's in use now, yellow = made with something older.

Recomputing is offered only where it can't lose your work:
  skeletons  run again with the current skeleton model (then pecks, which are counted from them)
  pecks      counted again
  naming     the current recogniser re-guesses only the pieces you haven't confirmed; your answers
             and set-asides stay, and it's one undo step on the naming page
Detection and pieces are not offered: redoing them rebuilds the pieces, which throws away the
clip's naming.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import records
from .progress import METHOD_VERSIONS, read as read_progress

UPDATABLE = {"naming", "skeletons", "pecks"}
KIND = {"detect": "detector", "naming": "recognizer", "skeletons": "pose"}
_vcache: dict = {}


def short(version: str | None) -> str:
    """all-r4-20261107-0209 -> all-r4 (the date is in the tooltip)."""
    if not version:
        return "?"
    return re.sub(r"-\d{8}(-\d{4})?$", "", version)


def model_version(kind: str, weights: str | None) -> str | None:
    """Registered version of a weights file (by its name in the store, else by checksum, cached)."""
    if not weights:
        return None
    p = Path(weights)
    if p.parent.name == kind and p.parent.parent.name == "models":
        return p.stem
    try:
        st = p.stat()
    except OSError:
        return p.stem
    key = (kind, str(p), st.st_mtime, st.st_size)
    if key not in _vcache:
        _vcache[key] = records.version_of(kind, p) or p.stem
    return _vcache[key]


def numbers() -> dict[tuple[str, str], int]:
    """(kind, version name) -> its plain number (detector v3 ...)."""
    try:
        return {(m["kind"], m["version"]): m["number"] for m in records.models() if m.get("number")}
    except Exception:
        return {}


def latest() -> dict[str, str | None]:
    out = {}
    try:
        for m in records.models():
            if m["status"] == "current":
                out[m["kind"]] = m["version"]
    except Exception:
        pass
    return {"detect": out.get("detector"), "pieces": f"v{METHOD_VERSIONS['pieces']}",
            "naming": out.get("recognizer"), "skeletons": out.get("pose"), "pecks": f"v{METHOD_VERSIONS['pecks']}"}


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def of_run(run: Path, prog: dict | None = None) -> dict[str, dict]:
    """{step: {"version", "latest", "current": bool|None, "note"}} for the steps this clip has done."""
    run = Path(run)
    prog = prog if prog is not None else read_progress(run)
    new = latest()
    nums = numbers()
    out: dict[str, dict] = {}

    def label(step, version):
        n = nums.get((KIND.get(step, ""), version)) if version else None
        return f"v{n}" if n else short(version)

    def put(step, version, note=""):
        lv = new.get(step)
        full = lambda v: f"{label(step, v)} ({v})" if v and label(step, v) != v else v
        out[step] = {"version": full(version), "short": label(step, version), "latest": full(lv),
                     "latest_short": label(step, lv), "current": (version == lv) if version and lv else None,
                     "note": note}

    meta = _json(run / "meta.json")
    if meta:
        put("detect", model_version("detector", meta.get("track_config", {}).get("model")))
    if (run / "identity" / "identity.json").exists():
        put("pieces", f"v{prog.get('pieces', {}).get('version', 1)}")
        d = _json(run / "identity" / "identity.json")
        w = (d.get("auto_named") or {}).get("weights") or _json(run / "identity" / "recognizer.json").get("weights")
        if w:
            put("naming", model_version("recognizer", w))
        else:
            out["naming"] = {"version": "by you", "short": "by you", "latest": None, "current": None, "note": ""}
        if d:  # everything confirmed or set aside: a newer recogniser would change nothing
            from . import identity as I
            pieces = {int(t) for t in d.get("fragments", {})}
            if pieces and pieces <= (I.confirmed(d) | I.excluded(d)) and "naming" in out:
                out["naming"].update(current=True, short="all yours",
                                     note="every piece confirmed by you: nothing left to re-guess")
    if (run / "poses.csv").exists():
        put("skeletons", model_version("pose", _json(run / "poses.json").get("weights")))
    if (run / "pecks.csv").exists():
        put("pecks", f"v{prog.get('pecks', {}).get('version', 1)}")
        if (run / "poses.csv").exists() and (run / "poses.csv").stat().st_mtime > (run / "pecks.csv").stat().st_mtime + 1:
            out["pecks"].update(current=False, note="counted from older skeletons")
    for step, v in out.items():
        v["can_update"] = step in UPDATABLE and v["current"] is False
    return out
