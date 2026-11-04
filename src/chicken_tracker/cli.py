"""Command-line interface: ``chicken-tracker {process,analyze,snapshot,mine,review,train,identify,recognizer,preview,stations,player}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .pipeline import process_video, reanalyze
from .tracking import TrackConfig, read_frame
from .zones import load_zones

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _add_analytics_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("analytics")
    g.add_argument("--zones", type=Path, help="JSON file of named polygons, e.g. feeder/water/nest")
    g.add_argument("--px-per-meter", type=float, help="calibration; report distances in metres")
    g.add_argument("--min-track-s", type=float, default=1.0)
    g.add_argument("--active-speed", type=float,
                   help="speed above which a chicken is 'active' (default 15 px/s or 0.05 m/s)")
    g.add_argument("--zone-anchor", choices=["bottom", "center"], default="bottom",
                   help="point of the bbox used for zone tests (bottom = feet)")
    g.add_argument("--zone-min-visit-s", type=float, default=1.0)
    g.add_argument("--timeline-bin-s", type=float, default=10.0)


def _analytics_kwargs(a: argparse.Namespace) -> dict:
    return dict(
        px_per_meter=a.px_per_meter, min_track_s=a.min_track_s, active_speed=a.active_speed,
        zone_anchor=a.zone_anchor, zone_min_visit_s=a.zone_min_visit_s,
        timeline_bin_s=a.timeline_bin_s,
    )


def _collect_videos(inputs: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for p in inputs:
        if p.is_dir():
            videos += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        elif p.exists():
            videos.append(p)
        else:
            sys.exit(f"Not found: {p}")
    return videos


def cmd_process(a: argparse.Namespace) -> None:
    cfg = TrackConfig(model=a.model, conf=a.conf, imgsz=a.imgsz, vid_stride=a.stride,
                      device=a.device, classes=a.classes, skip_empty=a.skip_empty)
    if a.tracker:
        cfg.tracker = a.tracker
    zones = load_zones(a.zones)
    videos = _collect_videos(a.inputs)
    if not videos:
        sys.exit("No video files found")
    for v in videos:
        summary = process_video(v, a.out, cfg, zones, _analytics_kwargs(a), save_video=a.save_video)
        _print_summary(v.stem, summary, a.out / v.stem)


def cmd_analyze(a: argparse.Namespace) -> None:
    summary = reanalyze(a.run_dir, load_zones(a.zones), _analytics_kwargs(a))
    _print_summary(a.run_dir.name, summary, a.run_dir)


def cmd_snapshot(a: argparse.Namespace) -> None:
    frame = read_frame(a.video, a.frame)
    cv2.imwrite(str(a.output), frame)
    h, w = frame.shape[:2]
    print(f"Wrote {a.output} ({w}x{h}). Use pixel coordinates from this image in zones.json.")


def cmd_mine(a: argparse.Namespace) -> None:
    from .labeling import mine

    if a.crowded:
        from .labeling import mine_crowded

        stats = mine_crowded(a.crowded, a.dataset, model=a.model if a.model != "yoloe-11l-seg.pt" else
                             "dataset/runs/chicken/weights/best.pt", n=a.max_frames, device=a.device,
                             video=str(a.video) if a.video else None)
        print(f"\n{stats}\nNext: chicken-tracker review {a.dataset}  (crowded frames come first)")
        return
    if a.objects:
        from .labeling import mine_objects

        stats = mine_objects(a.dataset, model=a.model, conf=a.conf, imgsz=a.imgsz, device=a.device)
        print(f"\n{stats}\nNext: chicken-tracker review {a.dataset} --objects")
        return
    videos = _collect_videos(a.inputs or [])
    if not videos:
        sys.exit("Give video files/folders to mine (or --objects to add feeder/water suggestions).")
    stats = mine(videos, a.dataset, model=a.model, prompts=a.prompt, every_s=a.every,
                 conf=a.conf, imgsz=a.imgsz, max_frames=a.max_frames, device=a.device)
    print(f"\n{stats}\nNext: chicken-tracker review {a.dataset}")


def cmd_review(a: argparse.Namespace) -> None:
    from .review_server import serve

    serve(a.dataset, port=a.port, open_browser=not a.no_browser, objects=a.objects,
          outline=not a.no_outline, device=a.device, pose=a.pose)


def cmd_outline(a: argparse.Namespace) -> None:
    from .labeling import auto_outline

    r = auto_outline(a.dataset, device=a.device, redo=a.redo)
    print(f"{r['frames']} frames: {r['outlined']} hens outlined, {r['failed']} left as boxes")


def cmd_train(a: argparse.Namespace) -> None:
    from .labeling import train, train_pose, train_seg

    if a.seg:
        train_seg(a.dataset, model=a.model if a.model != "yolo11s.pt" else "yolo11s-seg.pt", epochs=a.epochs,
                  imgsz=a.imgsz, batch=a.batch, device=a.device, name=a.name if a.name != "chicken" else "seg",
                  exclude_source=a.exclude_source)
        return
    if a.pose:
        train_pose(a.dataset, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device)
        return

    train(a.dataset, model=a.model, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
          with_objects=a.with_objects, name=a.name)


def cmd_identify(a: argparse.Namespace) -> None:
    from . import identity
    from .identity_server import serve

    if a.rebuild or not (a.run_dir / "identity" / "identity.json").exists():
        identity.build(a.run_dir, flock_size=a.flock_size, device=a.device,
                       video=str(a.video) if a.video else None)
        identity.export(a.run_dir)
        if a.auto_name:
            from .recognizer import auto_name

            r = auto_name(a.run_dir, a.recognizer, device=a.device, video=str(a.video) if a.video else None)
            print(f"Named by the recogniser: {r}")
    if not a.build_only:
        serve(a.run_dir, port=a.port, open_browser=not a.no_browser)


def cmd_recognizer(a: argparse.Namespace) -> None:
    from .recognizer import build_id_dataset, cross_fit, evaluate, train_recognizer

    if a.audit:
        r = cross_fit(a.run_dir, a.out, folds=a.folds, device=a.device, video=str(a.video) if a.video else None)
        print(f"{r['crops']} crops judged by models that never saw their piece; fold accuracy "
              + ", ".join(f"{v:.1%}" for v in r["fold_accuracy"].values()) + "\nSee the Checks tab.")
        return
    if not a.skip_build:
        info = build_id_dataset(a.run_dir, a.out, video=str(a.video) if a.video else None)
        print(f"{info['crops']} crops\n{info['counts']}")
    weights = train_recognizer(a.out, epochs=a.epochs, device=a.device)
    r = evaluate(a.out, weights, device=a.device)
    print(f"\nHeld-back pieces: crops {r['crop_acc']:.1%}, pieces {r['piece_acc']:.1%}\n{r['confusion']}")
    print(f"Weights: {weights}")


def cmd_pose(a: argparse.Namespace) -> None:
    from .pose import run_pose

    run_pose(a.run_dir, a.model, video=str(a.video) if a.video else None, device=a.device, conf=a.conf)
    print("Switch on \"Skeletons\" in the player (chicken-tracker player RUN_DIR).")


def cmd_pecks(a: argparse.Namespace) -> None:
    from .behaviour import write_pecking

    print(write_pecking(a.run_dir, video=str(a.video) if a.video else None))


def cmd_dashboard(a: argparse.Namespace) -> None:
    from .dashboard import serve

    serve(a.out, port=a.port, open_browser=not a.no_browser)


def cmd_preview(a: argparse.Namespace) -> None:
    from .preview import parse_time, render

    out = render(a.run_dir, parse_time(a.start), parse_time(a.duration), out=a.output,
                 video=str(a.video) if a.video else None, width=a.width)
    print(f"Wrote {out}")
    if not a.no_open:
        import subprocess
        subprocess.run(["open", str(out)], check=False)


def cmd_stations(a: argparse.Namespace) -> None:
    from . import identity
    from .stations import save_stations, stations_from_labels, write_station_analytics

    stations = stations_from_labels(a.dataset)
    if not stations:
        sys.exit(f"No feeder/water boxes labelled in {a.dataset}. Run `chicken-tracker review --objects` first.")
    print(f"Wrote {save_stations(a.run_dir, stations)}")
    for s in stations:
        print(f"  {s['id']:10s} box {s['box']}  (labelled in {s['labelled_in_frames']} frames)")
    identity.export(a.run_dir)  # rewrites chickens.csv, then adds feeder/water columns
    meta = json.loads((a.run_dir / "meta.json").read_text())
    print(write_station_analytics(a.run_dir, meta["video"]["fps"], margin=a.margin,
                                  min_visit_s=a.min_visit_s))


def cmd_player(a: argparse.Namespace) -> None:
    from . import player

    meta = json.loads((a.run_dir / "meta.json").read_text())
    src = Path(a.video) if a.video else Path(meta["video"]["path"])
    fps = meta["video"]["fps"]
    out = a.run_dir / "browser" / f"video_{a.width}{'_dn' if a.denoise else ''}.mp4"
    if a.reencode or not out.exists():
        if not src.exists():
            sys.exit(f"Source video not found: {src} - pass --video if it moved.")
        print(f"Encoding browser copy -> {out} (one-off; ~1/7 of the video's length)")
        player.compress_for_browser(src, out, fps, width=a.width, denoise=a.denoise)
        n_src, n_out = player.count_frames(src), player.count_frames(out)
        status = "OK" if n_src == n_out else "MISMATCH - overlay may drift"
        print(f"Frames: original {n_src}, browser copy {n_out} -> {status}")
        print(f"Size: {src.stat().st_size / 1e6:.0f} MB -> {out.stat().st_size / 1e6:.0f} MB "
              f"({100 * out.stat().st_size / src.stat().st_size:.0f}%)")
    if not a.encode_only:
        player.serve(a.run_dir, out, str(src), port=a.port, open_browser=not a.no_browser)


def _print_summary(name: str, summary: dict, out_dir: Path) -> None:
    brief = {k: v for k, v in summary.items() if k != "video"}
    print(f"\n== {name} ==\n{json.dumps(brief, indent=2)}\nOutputs: {out_dir}/\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chicken-tracker", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="detect + track chickens in video file(s) and compute analytics")
    p.add_argument("inputs", nargs="+", type=Path, help="video files or directories")
    p.add_argument("-o", "--out", type=Path, default=Path("output"))
    p.add_argument("--model", default="yolo11s.pt",
                   help="YOLO weights; use your fine-tuned chicken model for best results")
    p.add_argument("--classes", type=int, nargs="+",
                   help="class ids to track (default: auto-detect chicken/bird classes)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame (faster)")
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.add_argument("--tracker", help="custom tracker YAML (default: bundled chicken_botsort.yaml)")
    p.add_argument("--save-video", action="store_true", help="write annotated.mp4 with IDs")
    p.add_argument("--skip-empty", action="store_true",
                   help="quick scan once a second first; track only the stretches with hens in view")
    _add_analytics_args(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("analyze", help="recompute analytics from a previous run (no re-detection)")
    p.add_argument("run_dir", type=Path, help="e.g. output/<video-name>")
    _add_analytics_args(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("snapshot", help="save a frame as PNG to help draw zones")
    p.add_argument("video", type=Path)
    p.add_argument("output", type=Path, nargs="?", default=Path("frame.png"))
    p.add_argument("--frame", type=int, default=0)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("mine", help="find candidate chicken frames for yes/no review")
    p.add_argument("inputs", nargs="*", type=Path, help="video files or directories")
    p.add_argument("-d", "--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--objects", action="store_true",
                   help="add feeder/water suggestions to frames already in the dataset (no videos needed)")
    p.add_argument("--crowded", type=Path, metavar="RUN_DIR",
                   help="add frames where hens bunch up in this analysed run (merged '2+' boxes)")
    p.add_argument("--video", type=Path, help="with --crowded: source video, if it moved")
    p.add_argument("--model", default="yoloe-11l-seg.pt",
                   help="proposal model: open-vocabulary (yoloe/world) or your trained best.pt")
    p.add_argument("--prompt", nargs="+", default=["chicken"], help="text prompt(s) for open-vocab models")
    p.add_argument("--every", type=float, default=2.0, help="sample one frame every N seconds")
    p.add_argument("--conf", type=float, default=0.03, help="low on purpose: you reject the junk")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--max-frames", type=int, default=300, help="frames to add per run")
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("review", help="open the yes/no review page in your browser")
    p.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--objects", action="store_true", help="review feeder/water boxes instead of chickens")
    p.add_argument("--no-outline", action="store_true", help="turn off click-to-outline (SAM)")
    p.add_argument("--pose", action="store_true", help="label chicken skeletons (beak, head, neck, back, tail)")
    p.add_argument("--device", help="for click-to-outline: cpu, mps, 0, ...")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("train", help="train a chicken detector on the reviewed frames")
    p.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    p.add_argument("--model", default="yolo11s.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.add_argument("--name", default="chicken", help="run name: weights go to <dataset>/runs/<name>/")
    p.add_argument("--pose", action="store_true", help="train the chicken skeleton model instead")
    p.add_argument("--seg", action="store_true", help="train the hen outline model instead (run `outline` first)")
    p.add_argument("--exclude-source", help="leave out frames whose source starts with this (test on an unseen day)")
    p.add_argument("--with-objects", action="store_true",
                   help="also learn feeder/water (needs the objects pass on most frames)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("outline", help="SAM outlines for reviewed hens that only have boxes (for `train --seg`)")
    p.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.add_argument("--redo", action="store_true", help="redo frames that already have automatic outlines")
    p.set_defaults(func=cmd_outline)

    p = sub.add_parser("identify", help="group track pieces into named chickens (opens naming page)")
    p.add_argument("run_dir", type=Path, help="e.g. output/<video-name>")
    p.add_argument("--video", type=Path, help="source video, if it moved since `process`")
    p.add_argument("--flock-size", type=int, help="stop grouping at this many chickens")
    p.add_argument("--rebuild", action="store_true", help="redo grouping (discards your answers)")
    p.add_argument("--auto-name", action="store_true",
                   help="name the hens with the recogniser (for a new day; needs a fresh build)")
    p.add_argument("--recognizer", type=Path, default=Path("recognizer_runs/hens/weights/best.pt"))
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("recognizer", help="train a model on your named hens (crop -> which hen)")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("dataset_id"))
    p.add_argument("--video", type=Path, help="source video, if it moved since `process`")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--skip-build", action="store_true", help="reuse the existing crop set")
    p.add_argument("--audit", action="store_true",
                   help="train K models each without some pieces; feeds the Checks tab (~12 min)")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.set_defaults(func=cmd_recognizer)

    p = sub.add_parser("pose", help="chicken skeletons over the whole video, for the player overlay")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--model", type=Path, default=Path("dataset/runs/pose/weights/best.pt"))
    p.add_argument("--video", type=Path, help="source video, if it moved since `process`")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", help="cpu, mps, 0, ...")
    p.set_defaults(func=cmd_pose)

    p = sub.add_parser("pecks", help="count pecks per hen (head moves, body still) - needs `pose` first")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--video", type=Path, help="source video, if it moved since `process`")
    p.set_defaults(func=cmd_pecks)

    p = sub.add_parser("dashboard", help="web page: progress of every analysed clip, live")
    p.add_argument("-o", "--out", type=Path, default=Path("output"))
    p.add_argument("--port", type=int, default=8760)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("stations", help="feeder/water positions from labelled boxes + per-chicken visits")
    p.add_argument("run_dir", type=Path)
    p.add_argument("-d", "--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--margin", type=float, default=30.0, help="px around a station that still counts as at it")
    p.add_argument("--min-visit-s", type=float, default=2.0)
    p.set_defaults(func=cmd_stations)

    p = sub.add_parser("player", help="browser player: small video + live overlay from the analysis")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--video", type=Path, help="source video, if it moved since `process`")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--denoise", action="store_true", help="strong temporal filter: ~26%% instead of ~35%% size")
    p.add_argument("--reencode", action="store_true", help="re-make the browser copy")
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_player)

    p = sub.add_parser("preview", help="render a short video with boxes, names, trails")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--start", default="0", help="seconds or mm:ss into the video")
    p.add_argument("--duration", default="3:00", help="seconds or mm:ss")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--video", type=Path, help="source video, if it moved since `process`")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=cmd_preview)

    a = parser.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
