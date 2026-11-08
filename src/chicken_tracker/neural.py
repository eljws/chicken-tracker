"""Run the hen detector on the Mac's Neural Engine (Core ML) instead of the GPU: same boxes, ~2x faster tracking.

The detector's .pt weights are converted once to a Core ML model with the exact input shape processing uses
(1920x1536 video at imgsz 1280 -> 1280x1024; a square input adds grey bars that cost hens at the top and
bottom edges). The conversion is kept next to the weights, keyed by the weights' contents, so a newly
promoted detector gets its own. Before first use it is checked against the original on frames of the video
being processed; if the boxes differ, or anything fails, processing simply stays on the GPU.

Converting needs a torch version coremltools supports, so it runs in its own throwaway uv environment;
running the converted model only needs coremltools.
"""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

EXPORT_ENV = ["uv", "run", "--no-project", "--python", "3.11", "--with", "torch==2.7.*",
              "--with", "torchvision==0.22.*", "--with", "ultralytics==8.4.197", "--with", "coremltools==9.0",
              "--with", "numpy<2.3"]
CHECK_FRAMES = 6
MIN_IOU = 0.9


def input_shape(width: int, height: int, imgsz: int, stride: int = 32) -> tuple[int, int]:
    """(h, w) the detector sees for a frame, as YOLO's rectangular letterbox makes it."""
    r = imgsz / max(width, height)
    w, h = round(width * r), round(height * r)
    return h + (imgsz - h) % stride, w + (imgsz - w) % stride


def available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def model_path(weights: Path, shape: tuple[int, int]) -> Path:
    from .records import sha256
    return weights.parent / "coreml" / f"{weights.stem}-{sha256(weights)[:12]}-{shape[1]}x{shape[0]}.mlpackage"


def _export(weights: Path, shape: tuple[int, int], dest: Path, log=print) -> None:
    log(f"  converting {weights.name} for the Neural Engine ({shape[1]}x{shape[0]}), once - about a minute")
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp) / "det.pt"
        shutil.copy(weights, w)
        code = ("import sys; from ultralytics import YOLO; "
                "YOLO(sys.argv[1]).export(format='coreml', imgsz=[int(sys.argv[2]), int(sys.argv[3])], half=True, nms=False)")
        subprocess.run([*EXPORT_ENV, "python", "-c", code, str(w), str(shape[0]), str(shape[1])],
                       check=True, capture_output=True, timeout=1800)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(tmp) / "det.mlpackage"), dest)


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def _check(weights: Path, cm: Path, shape, video, cfg) -> dict:
    """Both models on frames spread over the video: every confident box of one must be in the other."""
    import cv2
    from ultralytics import YOLO
    pt, ml = YOLO(str(weights)), YOLO(str(cm), task="detect")
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res = {"frames": 0, "boxes": 0, "matched": 0, "worst_iou": 1.0}
    try:
        for k in range(CHECK_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / CHECK_FRAMES))
            ok, img = cap.read()
            if not ok:
                continue
            a = pt.predict(img, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, device=cfg.device, verbose=False)[0]
            b = ml.predict(img, conf=cfg.conf, iou=cfg.iou, imgsz=list(shape), verbose=False)[0]
            res["frames"] += 1
            for x, y, ys in ((a, b, b.boxes.xyxy.tolist()), (b, a, a.boxes.xyxy.tolist())):
                for box, c in zip(x.boxes.xyxy.tolist(), x.boxes.conf.tolist()):
                    if c < cfg.conf + 0.1:   # boxes near the cut-off may fall either side of it
                        continue
                    best = max((_iou(box, o) for o in ys), default=0.0)
                    res["boxes"] += 1
                    res["matched"] += best >= MIN_IOU
                    res["worst_iou"] = round(min(res["worst_iou"], best), 3)
    finally:
        cap.release()
    res["ok"] = res["frames"] > 0 and res["matched"] == res["boxes"]
    return res


def prepare(cfg, video, width: int, height: int, log=print):
    """`cfg` with the engine chosen: 'neural' (+ the Core ML model and its input shape) or 'gpu'."""
    if cfg.engine not in ("auto", "neural"):
        return replace(cfg, engine="gpu")
    if cfg.device not in (None, "mps") or not available():
        return replace(cfg, engine="gpu")
    weights = Path(cfg.model)
    if weights.suffix != ".pt" or not weights.exists():
        return replace(cfg, engine="gpu")
    shape = input_shape(width, height, cfg.imgsz)
    cm = model_path(weights, shape)
    check = cm.with_suffix(".check.json")
    try:
        cm.parent.mkdir(parents=True, exist_ok=True)
        with open(cm.with_suffix(".lock"), "w") as lock:   # two processing jobs never convert at once
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not cm.exists():
                _export(weights, shape, cm, log)
            if not check.exists():
                res = _check(weights, cm, shape, video, cfg)
                check.write_text(json.dumps({"video": str(video), **res}, indent=1))
                log(f"  Neural Engine check: {res['matched']}/{res['boxes']} boxes the same "
                    f"(worst overlap {res['worst_iou']}) on {res['frames']} frames -> "
                    f"{'using it' if res['ok'] else 'staying on the GPU'}")
        if not json.loads(check.read_text()).get("ok"):
            return replace(cfg, engine="gpu")
    except Exception as e:  # noqa: BLE001 - any trouble: the GPU path is always there
        log(f"  Neural Engine not used ({type(e).__name__}: {e}); staying on the GPU")
        return replace(cfg, engine="gpu")
    return replace(cfg, engine="neural", engine_model=str(cm), engine_imgsz=list(shape))
