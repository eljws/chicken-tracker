"""Browser player: a small video plus a live overlay drawn from the analysis data.

Recognition runs on the full-quality original; the browser plays a compressed copy and
draws boxes, names, trails, stations and hidden-chicken markers on a canvas on top.

The copy keeps every frame and is re-timed to exactly ``frame / fps`` (the camera's own
timestamps are slightly irregular), so frame N in the data is frame N on screen. The
overlay JSON is rebuilt from the latest CSVs on each page load, so naming fixes show up
without re-encoding anything.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import re
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .stations import at_station, load_stations

DENOISE = "hqdn3d=0:0:20:20"  # strong temporal filter: ~26% instead of ~35% size, smears fast birds


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def compress_for_browser(src: Path, out: Path, fps: float, width: int = 1280, crf: int = 26,
                         denoise: bool = False, progress: bool = True) -> Path:
    """H.264 copy for browsers: every frame kept, timestamps = frame / fps, 10 s keyframes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"setpts=N/({fps!r}*TB)", f"scale={width}:-2"]
    if denoise:
        vf.insert(1, DENOISE)
    tmp = out.with_suffix(".partial.mp4")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-an",
           "-vf", ",".join(vf), "-fps_mode", "passthrough", "-video_track_timescale", "90000",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-g", str(int(round(fps * 10))), "-keyint_min", str(int(round(fps))),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    if progress:
        cmd[4:4] = ["-stats"]
    subprocess.run(cmd, check=True)
    tmp.replace(out)  # only a finished encode gets the real name
    return out


def count_frames(path: Path) -> int:
    """Exact number of decoded frames (ffmpeg's -f null pass)."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"frame=\s*(\d+)", r.stderr)
    return int(m[-1]) if m else -1


def _color(key: str) -> str:
    """Same colour per name as preview.py (md5 hue), as #rrggbb."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _clock_start(video: str) -> int | None:
    """Seconds after midnight the recording starts, from names like 072145-075411_ch04.mp4."""
    m = re.search(r"(?:^|[_/])(\d{2})(\d{2})(\d{2})-\d{6}", Path(video).name) or \
        re.search(r"T(\d{2})(\d{2})(\d{2})_", Path(video).name)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None


def build_overlay(run_dir: Path, video: str | None = None) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    fps = meta["video"]["fps"]
    named = pd.read_csv(run_dir / "detections_named.csv")
    names = sorted(named.chicken.unique())
    idx = {n: i for i, n in enumerate(names)}
    stations = load_stations(run_dir)
    station_code = np.zeros(len(named), int)
    for st in stations:
        hit = at_station(named, st["box"], 30.0) & (station_code == 0)
        station_code[hit] = {"feeder": 1, "water": 2}.get(st["type"], 0)

    frames: dict[str, list[int]] = {}
    cols = np.column_stack([named.chicken.map(idx).to_numpy(), named[["x1", "y1", "x2", "y2"]].to_numpy().round(),
                            station_code]).astype(int)
    for f, row in zip(named.frame.to_numpy(), cols):
        frames.setdefault(str(int(f)), []).extend(row.tolist())

    # boxes set aside as "several chickens": drawn as grey "2+ chickens" so birds inside a
    # merged box don't simply vanish from the overlay
    groups: dict[str, list[int]] = {}
    ident_p = run_dir / "identity" / "identity.json"
    if ident_p.exists():
        from .identity import load_detections
        ident = json.loads(ident_p.read_text())
        multi = set(ident.get("multiple", []))
        if multi:
            det = load_detections(run_dir, ident)
            m = det[det.track_id.isin(multi)]
            for f, row in zip(m.frame.to_numpy(), m[["x1", "y1", "x2", "y2"]].to_numpy().round().astype(int)):
                groups.setdefault(str(int(f)), []).extend(row.tolist())

    hidden = []
    tl_p = run_dir / "chicken_timeline.csv"
    if tl_p.exists():
        tl = pd.read_csv(tl_p)
        last = named.sort_values("frame").groupby("track_id").tail(1).set_index("track_id")
        for _, h in tl[tl.state == "hidden"].iterrows():
            if h.track_before in last.index and h.chicken in idx:
                r = last.loc[h.track_before]
                hidden.append({"c": idx[h.chicken], "from": round(h.start_s, 3), "to": round(h.end_s, 3),
                               "box": [round(r.x1), round(r.y1), round(r.x2), round(r.y2)]})

    summary = {}
    ch_p = run_dir / "chickens.csv"
    if ch_p.exists():
        for _, r in pd.read_csv(ch_p).iterrows():
            summary[r.chicken] = {k: (None if pd.isna(v) else (round(float(v), 2) if isinstance(v, float) else v))
                                  for k, v in r.items() if k in (
                                      "visible_s", "hidden_s", "distance_m", "speed_m_per_min", "top_speed_m_s",
                                      "brisk_speed_m_s", "top_accel_m_s2", "feeder_visits", "feeder_time_s",
                                      "water_visits", "water_time_s", "pecks_at_feeder", "pecks_ground")}
    pecks: dict[str, list[int]] = {}  # frame -> [chicken, head x, head y, ...]
    pk_p = run_dir / "pecks.csv"
    if pk_p.exists():
        for _, r in pd.read_csv(pk_p).iterrows():
            if r.chicken in idx:
                pecks.setdefault(str(int(r.frame)), []).extend([idx[r.chicken], round(r.head_x), round(r.head_y)])
    moments = []
    fm_p = run_dir / "fast_moments.csv"
    if fm_p.exists():
        moments = [{"c": idx[r.chicken], "at": round(r.at_s, 2), "speed": round(r.speed_m_s, 2)}
                   for _, r in pd.read_csv(fm_p).iterrows() if r.chicken in idx]

    return {
        "fps": fps, "width": meta["video"]["width"], "height": meta["video"]["height"],
        "frames_total": meta["video"]["frame_count"],
        "clock_start_s": _clock_start(video or meta["video"]["path"]),
        "chickens": [{"name": n, "color": _color(n), **({"summary": summary[n]} if n in summary else {})}
                     for n in names],
        "stations": stations, "hidden": hidden, "moments": moments,
        "frames": frames, "groups": groups, "pecks": pecks,
    }


def make_handler(run_dir: Path, video_path: Path, source_video: str | None):
    page = (resources.files("chicken_tracker") / "static" / "player.html").read_bytes()
    lock = threading.Lock()
    cache: dict = {}

    def overlay_bytes() -> bytes:
        deps = [run_dir / n for n in ("detections_named.csv", "chicken_timeline.csv", "chickens.csv",
                                      "fast_moments.csv", "stations.json", "identity/identity.json", "pecks.csv")]
        key = tuple(p.stat().st_mtime if p.exists() else 0 for p in deps)
        with lock:
            if cache.get("key") != key:
                cache["key"] = key
                cache["data"] = json.dumps(build_overlay(run_dir, source_video), separators=(",", ":")).encode()
            return cache["data"]

    def poses_bytes() -> bytes:
        from .pose import overlay_poses

        p = run_dir / "poses.csv"
        key = ("poses", p.stat().st_mtime if p.exists() else 0, cache.get("key"))
        with lock:
            if cache.get("poses_key") != key:
                names = [c["name"] for c in json.loads(cache["data"])["chickens"]] if "data" in cache else \
                    sorted(pd.read_csv(run_dir / "detections_named.csv").chicken.unique())
                cache["poses_key"] = key
                cache["poses"] = json.dumps(overlay_poses(run_dir, names) or {"frames": {}},
                                            separators=(",", ":")).encode()
            return cache["poses"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _video(self, head_only: bool = False):
            size = video_path.stat().st_size
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
            if m and (m[1] or m[2]):
                if m[1]:
                    start = int(m[1])
                    end = int(m[2]) if m[2] else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m[2]))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if head_only:
                return
            with open(video_path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser cancelled (seek): normal

        def do_HEAD(self):
            if self.path.split("?")[0] == "/video.mp4":
                return self._video(head_only=True)
            self.send_response(404)
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if path == "/overlay.json":
                return self._send(overlay_bytes(), "application/json")
            if path == "/poses.json":  # skeleton layer: fetched only when switched on
                return self._send(poses_bytes(), "application/json")
            if path == "/video.mp4":
                return self._video()
            self._send(b'{"error":"not found"}', "application/json", 404)

    return Handler


def serve(run_dir: Path, video_path: Path, source_video: str | None, port: int = 8770,
          open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(run_dir), Path(video_path), source_video))
    url = f"http://127.0.0.1:{port}/"
    print(f"Player for {Path(run_dir).name} at {url}  - Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
