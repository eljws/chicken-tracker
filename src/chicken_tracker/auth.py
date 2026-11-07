"""Login for reaching the home server from outside (the router's port forward).

The password is set with `chicken-tracker password` and kept only as a salted PBKDF2 hash in
~/.config/chicken-tracker/auth.json (readable by you only), never in the project. The browser asks
for it once (HTTP Basic login) and remembers it for the session.

Guessing is slowed down: every wrong try waits a second, and an address with 10 wrong tries in
10 minutes is turned away for 10 minutes.

Plain http: the password and pages are not encrypted on the way. Fine to start; HTTPS (or a VPN)
is the step up.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

AUTH_FILE = Path.home() / ".config" / "chicken-tracker" / "auth.json"
ITERATIONS = 300_000
MAX_FAILS, WINDOW_S, LOCK_S = 10, 600, 600


def _hash(password: str, salt: bytes, iterations: int = ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)


def set_password(user: str, password: str, path: Path = AUTH_FILE) -> None:
    salt = secrets.token_bytes(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"user": user, "salt": salt.hex(), "iterations": ITERATIONS,
                               "hash": _hash(password, salt).hex()}))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def load(path: Path = AUTH_FILE) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Gate:
    """Checks a request's Authorization header; remembers good headers so each picture isn't re-hashed."""

    def __init__(self, path: Path | None = None):
        self.path = path or AUTH_FILE
        self.lock = threading.Lock()
        self.good: set[bytes] = set()  # sha256 of headers already checked, for this run of the server
        self.fails: dict[str, list[float]] = {}
        self.stamp = None

    def _cfg(self) -> dict | None:
        try:
            st = self.path.stat().st_mtime
        except OSError:
            return None
        if st != self.stamp:  # password changed: forget remembered logins
            with self.lock:
                self.stamp, self.good = st, set()
        return load(self.path)

    @property
    def ready(self) -> bool:
        return self._cfg() is not None

    def locked(self, ip: str) -> bool:
        now = time.time()
        with self.lock:
            f = [t for t in self.fails.get(ip, []) if now - t < max(WINDOW_S, LOCK_S)]
            self.fails[ip] = f
            recent = [t for t in f if now - t < WINDOW_S]
            return len(recent) >= MAX_FAILS and now - f[-1] < LOCK_S

    def check(self, header: str | None, ip: str) -> bool:
        cfg = self._cfg()
        if cfg is None or not header or not header.startswith("Basic "):
            return False
        key = hashlib.sha256(header.encode()).digest()
        if key in self.good:
            return True
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
        except (ValueError, UnicodeDecodeError):
            user, pw = "", ""
        ok = hmac.compare_digest(user, cfg["user"]) & hmac.compare_digest(
            _hash(pw, bytes.fromhex(cfg["salt"]), cfg["iterations"]).hex(), cfg["hash"])
        with self.lock:
            if ok:
                self.good.add(key)
                self.fails.pop(ip, None)
            else:
                self.fails.setdefault(ip, []).append(time.time())
        if not ok:
            time.sleep(1.0)
        return ok
