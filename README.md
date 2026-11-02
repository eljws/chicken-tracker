# chicken-tracker

Detects, tracks and **names individual chickens** in CCTV video (MP4), then generates per-chicken analytics:
time visible, time hidden (inferred), distance walked, zone usage (feeder / water / nest), a flock timeline
and an occupancy heatmap.

**Pipeline:** YOLO detection (fine-tuned on your camera) → BoT-SORT tracking → appearance-based grouping of
track pieces into named chickens, with a human answering the doubtful cases in a local web page.

## Setup

```bash
uv sync
```

The `mine` step uses an open-vocabulary model (YOLOE) that needs the `clip` package (Ultralytics' CLIP fork on GitHub);
Ultralytics installs it automatically on first use.

## Workflow

### 1. Train a detector for your camera

Stock COCO weights barely see chickens in top-down CCTV, so train on a few hundred frames of your own footage:

```bash
uv run chicken-tracker mine recordings/cam1.mp4          # sample frames + suggested boxes -> dataset/
uv run chicken-tracker review                            # browser: Y/N per frame, click/drag to fix boxes
uv run chicken-tracker train --device mps                # -> dataset/runs/chicken/weights/best.pt
```

Run `mine` again with `--model dataset/runs/chicken/weights/best.pt` on new footage for much better suggestions.

### 2. Track a video

```bash
uv run chicken-tracker process recordings/cam1.mp4 \
    --model dataset/runs/chicken/weights/best.pt --imgsz 1280 --device mps [--zones zones.json]
```

`process` also accepts folders. Useful flags: `--stride 3` (every 3rd frame, ~3x faster), `--save-video`
(large annotated MP4), `--conf`, `--tracker my_tracker.yaml`. Run `--help` for everything.

### 3. Name the chickens

```bash
uv run chicken-tracker identify output/cam1 [--video path/if/moved.mp4]
```

Groups the tracker's pieces by appearance and opens a page with three tabs:

- **Name & merge** – type each chicken's name; merge groups that are the same bird; click a piece to move it.
- **Questions** – "Is this Rocky?" for doubtful pieces (keys: `Y` `N` `P` partial, `L` box too big,
  `M` several chickens, `X` not a chicken, `C` new chicken, `1-9` another chicken).
- **Conflicts** – two pieces with the same name on screen at once. Shows the real frame at the same moment
  with both boxes, a filmstrip of each piece, **✂ cut** (split a piece where it changes bird) and
  **✂ cut out just the overlap**.
- **Checks** – an automatic audit for mistakes the other tabs can't see: a piece set aside as
  "several chickens" whose box is really one bird, a named piece whose box grows to two birds,
  a piece whose look changes part-way (tracker switched birds). Each comes with frames and a
  one-click fix, or "looks fine" to dismiss. The header shows how many items are left to review.

Every change is saved immediately and can be undone (`↶ Undo` / ⌘Z, last 200 steps in `identity/history/`).

### 4. Feeders, water and distances

```bash
uv run chicken-tracker mine --objects                    # feeder/water suggestions on already-reviewed frames
uv run chicken-tracker review --objects                  # browser: 2 = feeder, 3 = water; Y/N like before
uv run chicken-tracker stations output/cam1              # station positions + per-chicken visits
```

The camera is static, so one frame with the feeder and water boxed is enough for the stations;
the decisions you make are remembered by location, so further frames are mostly `Y` (needed only
to train a detector that finds feeders/water on its own, e.g. for moving trays or new cameras).

Distances in metres need `output/<video>/calibration.json` (image → ground homography).
For camera 04 it was fitted to the 600 mm patio slabs (`calibration.refine_to_joints`) and
cross-checked against a 2 L bottle (0.33 m); copy it to other runs from the same camera.

### 5. Watch it in a browser

```bash
uv run chicken-tracker player output/cam1 [--denoise]
```

Makes a small browser copy of the video once (1280×1024 H.264, 10 s keyframes, every frame kept and
re-timed to exactly `frame / fps`; ~35% of the camera file, ~26% with `--denoise`) and serves a page
that draws boxes, names, trails, feeder/water and hidden-chicken markers **live on top** of it from
the analysis data. Recognition stays on the full-quality original; fixing a name in the naming page
shows up on reload without re-encoding. Click a chicken to highlight her, click the presence
timeline or a "fastest moment" to jump.

### 6. Check it (burned-in clip)

```bash
uv run chicken-tracker preview output/cam1 --start 17:30 --duration 3:00
```

Renders an H.264 clip with names, 4-second trails, and dashed "Rocky? 62%" markers where a chicken is hidden.

### Other commands

```bash
uv run chicken-tracker snapshot cam1.mp4 frame.png       # a frame to read zone coordinates from
uv run chicken-tracker analyze output/cam1 --zones zones.json --px-per-meter 120   # recompute, no re-detection
```

## Outputs (`output/<video>/`)

| File | Content |
|---|---|
| `chickens.csv` | **per named chicken:** visible / hidden seconds, distance (px and m), pace, top speed, brisk (95th pct) speed, top acceleration, feeder / water visits and time |
| `station_visits.csv` | every visit to a feeder / water station with start, end, duration |
| `fast_moments.csv` | each chicken's 3 fastest moments (time, speed, piece) – check them in the preview |
| `chicken_timeline.csv` | per chicken: visible and hidden segments; hidden ones carry a confidence that decays with gap length |
| `detections_named.csv` | one row per chicken per frame (full box preferred over partial/duplicate ones) |
| `flagged_pieces.csv` | pieces marked several chickens / not a chicken / partial / box too big – material for retraining |
| `detections.csv` | raw tracker output: `frame, time_s, track_id, bbox, centroid, conf` |
| `tracks.csv`, `zone_visits.csv`, `timeline.csv`, `summary.json`, `heatmap.png` | per-track and flock-level analytics |
| `identity/` | naming state (`identity.json`), crops, appearance features, undo history |

Per-chicken distances and speeds are in metres when `calibration.json` exists. Speed is the net
movement over 0.5 s of median-smoothed feet positions; windows with impossible jumps (> 5 m/s
frame-to-frame) or missing frames are dropped, and partial / too-big boxes are left out. A tracker
swapping between two nearby birds can still look like a short dash, so eyeball `fast_moments.csv`.

## Notes

- Training uses SGD: Ultralytics' automatic optimizer choice (AdamW) diverged on small datasets here.
- A hidden segment's confidence is `exp(-gap / 20 s)`; gaps over 120 s are not bridged.
- `--zone-anchor bottom` (default) tests the chicken's feet against zones, which suits angled cameras;
  use `center` for top-down cameras.
- Tracker tuning: `src/chicken_tracker/trackers/chicken_botsort.yaml`.
