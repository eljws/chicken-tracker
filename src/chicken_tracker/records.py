"""Project records: model versions, training rounds, the frozen test set, and backups.

Everything lives in ``records/`` next to ``dataset/`` and ``output/``::

    records/records.db               SQLite: models, rounds, splits, backups
    records/models/<kind>/<version>.pt   a copy of every registered model, never overwritten

* **Models** - every trained model is registered with its checksum, parent and what it was
  trained on. Each kind (detector, recognizer, pose, seg) has one ``current`` version, which
  the commands use by default; ``chicken-tracker models --use KIND VERSION`` switches (or goes
  back). The training folders (``dataset/runs/...``) may be reused; the copy here never is.
* **Rounds** - each training round's candidate, the model it was compared with, the scores and
  whether it was promoted, so you can see whether the loop is getting better.
* **Frozen test set** - once a frame is held back for testing it stays held back, so every model
  is judged on the same frames however many new ones are added.
* **Backups** - ``backup()`` copies what can't be recreated (your labels and frames, naming,
  stations, calibration, these records and models) into a dated folder. Files unchanged since
  the previous backup are hard links, so each backup only costs what changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

KINDS = ("detector", "recognizer", "pose", "seg")
ROOT = Path("records")
BACKUP_ROOT = Path.home() / "chicken-tracker-backups"


def _override(root):
    """CHICKEN_TRACKER_RECORDS points everything elsewhere (the tests use it)."""
    env = os.environ.get("CHICKEN_TRACKER_RECORDS")
    return Path(env) if env else Path(root)

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    kind TEXT NOT NULL, version TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
    parent TEXT, created REAL NOT NULL, status TEXT NOT NULL DEFAULT 'candidate',
    trained_on TEXT, metrics TEXT, note TEXT,
    PRIMARY KEY (kind, version));
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, at REAL NOT NULL,
    candidate TEXT, baseline TEXT, metrics TEXT, promoted INTEGER NOT NULL DEFAULT 0, note TEXT);
CREATE TABLE IF NOT EXISTS splits (
    frame TEXT PRIMARY KEY, split TEXT NOT NULL CHECK (split IN ('train', 'val')), frozen_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, path TEXT NOT NULL,
    files INTEGER, copied INTEGER, bytes_copied INTEGER, reason TEXT);
"""


def connect(root: Path = ROOT) -> sqlite3.Connection:
    root = _override(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "records.db", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")  # pages can read while a training round writes
    db.executescript(SCHEMA)
    return db


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- models

def register(kind: str, weights: Path, version: str | None = None, parent: str | None = None,
             trained_on: dict | None = None, metrics: dict | None = None, note: str = "",
             root: Path = ROOT, make_current_if_first: bool = True) -> str:
    """Copy a trained model into the store and record it; returns its version.

    The same file registered twice (same checksum) returns the existing version.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    weights = Path(weights)
    digest = sha256(weights)
    with connect(root) as db:
        row = db.execute("SELECT version FROM models WHERE kind=? AND sha256=?", (kind, digest)).fetchone()
        if row:
            return row["version"]
        version = version or time.strftime("%Y%m%d-%H%M%S")
        if db.execute("SELECT 1 FROM models WHERE kind=? AND version=?", (kind, version)).fetchone():
            version = f"{version}-{digest[:6]}"
        dst = _override(root) / "models" / kind / f"{version}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights, dst)
        first = db.execute("SELECT COUNT(*) FROM models WHERE kind=?", (kind,)).fetchone()[0] == 0
        db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (kind, version, str(dst), digest, parent, time.time(),
                    "current" if first and make_current_if_first else "candidate",
                    json.dumps(trained_on or {}), json.dumps(metrics or {}), note))
    return version


def use(kind: str, version: str, root: Path = ROOT) -> None:
    """Make `version` the current model of its kind (the old current one is kept, as 'previous')."""
    with connect(root) as db:
        if not db.execute("SELECT 1 FROM models WHERE kind=? AND version=?", (kind, version)).fetchone():
            raise ValueError(f"no {kind} model {version!r}")
        db.execute("UPDATE models SET status='previous' WHERE kind=? AND status='current'", (kind,))
        db.execute("UPDATE models SET status='current' WHERE kind=? AND version=?", (kind, version))


def current(kind: str, root: Path = ROOT) -> Path | None:
    """The current model of a kind, checked against its checksum (None if there's none)."""
    if not (_override(root) / "records.db").exists():
        return None
    with connect(root) as db:
        row = db.execute("SELECT path, sha256 FROM models WHERE kind=? AND status='current'", (kind,)).fetchone()
    if row is None or not Path(row["path"]).exists():
        return None
    return Path(row["path"])


def model_or(kind: str, given, fallback: str | Path) -> str:
    """What a command should load: what you passed, else the current registered model, else the old default."""
    if given:
        return str(given)
    cur = current(kind)
    return str(cur) if cur else str(fallback)


def version_of(kind: str, weights, root: Path = ROOT) -> str | None:
    """The registered version with the same file contents, if any."""
    p = Path(str(weights))
    if not p.exists() or not (_override(root) / "records.db").exists():
        return None
    with connect(root) as db:
        row = db.execute("SELECT version FROM models WHERE kind=? AND sha256=?", (kind, sha256(p))).fetchone()
    return row["version"] if row else None


def models(kind: str | None = None, root: Path = ROOT) -> list[dict]:
    with connect(root) as db:
        q = "SELECT * FROM models" + (" WHERE kind=?" if kind else "") + " ORDER BY kind, created"
        return [dict(r) for r in db.execute(q, (kind,) if kind else ())]


def verify(root: Path = ROOT) -> list[str]:
    """Problems with the stored models (missing or changed files)."""
    bad = []
    for m in models(root=root):
        p = Path(m["path"])
        if not p.exists():
            bad.append(f"{m['kind']} {m['version']}: file missing ({p})")
        elif sha256(p) != m["sha256"]:
            bad.append(f"{m['kind']} {m['version']}: file changed since it was registered")
    return bad


# ---------------------------------------------------------------------------- rounds

def record_round(kind: str, candidate: str | None, baseline: str | None, metrics: dict,
                 promoted: bool, note: str = "", root: Path = ROOT) -> int:
    with connect(root) as db:
        cur = db.execute("INSERT INTO rounds (kind, at, candidate, baseline, metrics, promoted, note) "
                         "VALUES (?,?,?,?,?,?,?)",
                         (kind, time.time(), candidate, baseline, json.dumps(metrics), int(promoted), note))
        return cur.lastrowid


def rounds(kind: str | None = None, root: Path = ROOT) -> list[dict]:
    with connect(root) as db:
        q = "SELECT * FROM rounds" + (" WHERE kind=?" if kind else "") + " ORDER BY id"
        return [{**dict(r), "metrics": json.loads(r["metrics"] or "{}")}
                for r in db.execute(q, (kind,) if kind else ())]


# ---------------------------------------------------------------------------- frozen test set

def frozen_splits(items: dict[str, dict], rule, root: Path = ROOT) -> dict[str, str]:
    """train/val for each frame: the recorded one if it has one, else `rule`'s answer, recorded now.

    `rule(items) -> {frame: split}` is only consulted for frames never seen before, so adding
    frames can't move an existing frame between train and val.
    """
    with connect(root) as db:
        known = {r["frame"]: r["split"] for r in db.execute("SELECT frame, split FROM splits")}
        new = [k for k in items if k not in known]
        if new:
            proposed = rule(items)
            now = time.time()
            db.executemany("INSERT OR IGNORE INTO splits VALUES (?,?,?)",
                           [(k, proposed[k], now) for k in new])
            known.update({k: proposed[k] for k in new})
    return {k: known[k] for k in items}


def seed_splits(train: list[str], val: list[str], root: Path = ROOT) -> int:
    """Record an existing split (e.g. the one the current detector was tested with)."""
    now = time.time()
    with connect(root) as db:
        before = db.execute("SELECT COUNT(*) FROM splits").fetchone()[0]
        db.executemany("INSERT OR IGNORE INTO splits VALUES (?,?,?)",
                       [(k, "train", now) for k in train] + [(k, "val", now) for k in val])
        return db.execute("SELECT COUNT(*) FROM splits").fetchone()[0] - before


# ---------------------------------------------------------------------------- backups

def _backup_files(project: Path) -> list[Path]:
    """What can't be recreated from the videos: your decisions, your frames, the records."""
    project = Path(project)
    pats = ["dataset/labels.json", "dataset/images/*",
            "output/**/identity/identity.json", "output/**/stations.json",
            "output/**/calibration.json", "output/**/meta.json",
            "records/models/**/*.pt", "zones*.json"]
    files = {p for pat in pats for p in project.glob(pat) if p.is_file()}
    return sorted(files)


def backup(project: Path = Path("."), dest_root: Path = BACKUP_ROOT, reason: str = "",
           root: Path | None = None) -> dict:
    """A dated copy of everything irreplaceable. Unchanged files are hard links to the previous
    backup, so a backup after a few clicks costs a few kilobytes."""
    project = Path(project).resolve()
    root = _override(root if root else project / ROOT)
    dest_root = Path(os.environ.get("CHICKEN_TRACKER_BACKUPS", dest_root))
    dest_root.mkdir(parents=True, exist_ok=True)
    prev = sorted((d for d in dest_root.iterdir() if d.is_dir() and d.name[:1].isdigit()), reverse=True)
    prev = prev[0] if prev else None
    dest = dest_root / time.strftime("%Y-%m-%d_%H%M%S")
    n = 1
    while dest.exists():
        dest = dest_root / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{n}"
        n += 1
    files = _backup_files(project)
    copied = linked = nbytes = 0
    for src in files:
        rel = src.relative_to(project)
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        old = prev / rel if prev else None
        st = src.stat()
        if old is not None and old.exists():
            ost = old.stat()
            if ost.st_size == st.st_size and int(ost.st_mtime) == int(st.st_mtime):
                try:
                    os.link(old, dst)
                    linked += 1
                    continue
                except OSError:
                    pass
        shutil.copy2(src, dst)
        copied += 1
        nbytes += st.st_size
    # the records database: SQLite's own online copy, safe while pages are running
    if (root / "records.db").exists():
        (dest / "records").mkdir(parents=True, exist_ok=True)
        with connect(root) as db, sqlite3.connect(dest / "records" / "records.db") as out:
            db.backup(out)
        copied += 1
    info = {"path": str(dest), "files": len(files) + 1, "copied": copied, "linked": linked,
            "bytes_copied": nbytes}
    (dest / "BACKUP.json").write_text(json.dumps({**info, "reason": reason, "at": time.time(),
                                                  "project": str(project)}, indent=1))
    if (root / "records.db").exists():
        with connect(root) as db:
            db.execute("INSERT INTO backups (at, path, files, copied, bytes_copied, reason) VALUES (?,?,?,?,?,?)",
                       (time.time(), str(dest), info["files"], copied, nbytes, reason))
    return info
