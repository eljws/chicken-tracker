"""Click-to-outline for the review page, using SAM 2 (Segment Anything).

SAM knows nothing about chickens: given a point on an image it returns the object that
point belongs to, by its edges. One click on a hen gives her outline; extra clicks marked
"not this part" (label 0) trim it where it spills into a neighbour or the feed tray. The
box drawn from the outline is tight and consistent, which is what the detector learns from.

The image is encoded once (0.1-0.5 s on Apple GPU); each click after that is ~0.05 s.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

MODEL = "sam2.1_b.pt"  # ~155 MB, auto-downloaded by Ultralytics on first use


class Outliner:
    def __init__(self, model: str = MODEL, device: str | None = None):
        self.model, self.device = model, device
        self._p = None
        self._image: str | None = None
        self._lock = threading.Lock()

    def _predictor(self):
        if self._p is None:
            from ultralytics.models.sam import SAM2Predictor

            self._p = SAM2Predictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=self.model,
                                                   device=self.device, verbose=False, save=False))
        return self._p

    def warm_up(self) -> None:
        with self._lock:
            self._predictor()

    def outline(self, image: Path, points: list[list[float]], labels: list[int]) -> dict | None:
        """points in image pixels; labels 1 = this bird, 0 = not this part.

        Returns {"box": [x1, y1, x2, y2], "poly": [[x, y], ...]} for the region containing the
        first click, or None if nothing was found.
        """
        with self._lock:
            p = self._predictor()
            if self._image != str(image):
                p.reset_image()
                p.set_image(str(image))
                self._image = str(image)
            r = p(points=[points], labels=[labels])[0]
        if r.masks is None or not len(r.masks.data):
            return None
        mask = r.masks.data[0].cpu().numpy().astype(np.uint8)
        return mask_to_outline(mask, points[0])

    def outline_boxes(self, image: Path, boxes: list[list[float]]) -> list[dict | None]:
        """One outline per suggested box (the box is SAM's prompt), in the same order.

        Used to show a frame already outlined, so nothing needs clicking unless it's wrong.
        """
        if not boxes:
            return []
        with self._lock:
            p = self._predictor()
            if self._image != str(image):
                p.reset_image()
                p.set_image(str(image))
                self._image = str(image)
            r = p(bboxes=[[float(v) for v in b[:4]] for b in boxes])[0]
        masks = [] if r.masks is None else r.masks.data.cpu().numpy().astype(np.uint8)
        out = []
        for i, b in enumerate(boxes):
            o = mask_to_outline(masks[i], [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) if i < len(masks) else None
            # SAM sometimes grabs the whole yard or a sliver: then keep the suggested box as it was
            if o is not None and not (0.5 <= _area(o["box"]) / max(_area(b), 1.0) <= 1.6):
                o = None
            out.append(o)
        return out


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def mask_to_outline(mask: np.ndarray, seed: list[float]) -> dict | None:
    """Keep the connected piece of the mask under (or nearest to) the first click."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    x, y = int(round(seed[0])), int(round(seed[1]))
    h, w = mask.shape
    k = lab[min(max(y, 0), h - 1), min(max(x, 0), w - 1)]
    if k == 0:  # click just outside the mask: take the biggest piece
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    piece = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
    bx, by, bw, bh = cv2.boundingRect(c)
    return {"box": [float(bx), float(by), float(bx + bw), float(by + bh)],
            "poly": [[int(a), int(b)] for a, b in poly]}
