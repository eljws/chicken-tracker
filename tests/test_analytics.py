import numpy as np
import pandas as pd
import pytest

from chicken_tracker import analytics as A
from chicken_tracker.tracking import DETECTION_COLUMNS, resolve_classes
from chicken_tracker.zones import points_in_polygon

FPS = 10.0


def make_track(tid, xs, ys, start_frame=0, size=20):
    rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        f = start_frame + i
        rows.append((f, f / FPS, tid, x - size / 2, y - size / 2, x + size / 2, y + size / 2,
                     x, y, 0.9, 14))
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


@pytest.fixture
def det():
    # chicken 1 walks right at 10 px/frame (=100 px/s) for 5 s
    walker = make_track(1, np.arange(50) * 10.0 + 100, np.full(50, 100.0))
    # chicken 2 sits still inside the feeder for 5 s
    sitter = make_track(2, np.full(50, 500.0), np.full(50, 500.0))
    # chicken 3 is a 3-frame false positive
    blip = make_track(3, [50, 51, 52], [50, 50, 50])
    return pd.concat([walker, sitter, blip], ignore_index=True)


ZONES = {"feeder": np.array([[450, 450], [550, 450], [550, 550], [450, 550]], float)}


def cfg(**kw):
    return A.AnalyticsConfig(fps=FPS, total_frames=50, smooth_window=1, **kw)


def test_point_in_polygon():
    poly = ZONES["feeder"]
    inside = points_in_polygon(np.array([500, 400, 549]), np.array([500, 500, 451]), poly)
    assert inside.tolist() == [True, False, True]


def test_short_tracks_filtered(det):
    df = A.prepare(det, cfg())
    assert set(df.track_id) == {1, 2}


def test_track_summary(det):
    c = cfg()
    df = A.prepare(det, c)
    visits = A.zone_visits(df, ZONES, c)
    s = A.track_summary(df, visits, ZONES, c).set_index("track_id")

    assert s.loc[1, "distance_px"] == pytest.approx(490.0)
    assert s.loc[1, "p95_speed_px_s"] == pytest.approx(100.0)
    assert s.loc[1, "active_fraction"] > 0.95
    assert s.loc[1, "feeder_visits"] == 0

    assert s.loc[2, "distance_px"] == 0
    assert s.loc[2, "active_fraction"] == 0
    assert s.loc[2, "feeder_visits"] == 1
    assert s.loc[2, "feeder_time_s"] == pytest.approx(5.0)


def test_zone_visits_split_on_gap():
    xs = [500] * 20 + [100] * 30 + [500] * 20  # in, out 3 s, back in
    df = A.prepare(make_track(7, xs, [500] * 70), cfg())
    v = A.zone_visits(df, ZONES, cfg())
    assert len(v) == 2
    assert v.duration_s.tolist() == pytest.approx([2.0, 2.0])


def test_metres_calibration(det):
    c = cfg(px_per_meter=100.0)
    df = A.prepare(det, c)
    s = A.track_summary(df, A.zone_visits(df, {}, c), {}, c).set_index("track_id")
    assert s.loc[1, "distance_m"] == pytest.approx(4.9)


def test_timeline_and_summary(det):
    c = cfg(timeline_bin_s=1.0)
    df = A.prepare(det, c)
    visits = A.zone_visits(df, ZONES, c)
    tracks = A.track_summary(df, visits, ZONES, c)
    tl = A.flock_timeline(df, ZONES, c)
    assert len(tl) == 5
    assert tl.chickens_visible.tolist() == pytest.approx([2.0] * 5)
    assert tl.in_feeder.tolist() == pytest.approx([1.0] * 5)
    summary = A.flock_summary(df, tracks, tl, ZONES, c)
    assert summary["estimated_flock_size"] == 2
    assert summary["feeder_total_visits"] == 1


def test_empty_detections():
    c = cfg()
    df = A.prepare(pd.DataFrame(columns=DETECTION_COLUMNS), c)
    tracks = A.track_summary(df, A.zone_visits(df, ZONES, c), ZONES, c)
    tl = A.flock_timeline(df, ZONES, c)
    assert len(tracks) == 0
    assert A.flock_summary(df, tracks, tl, ZONES, c)["estimated_flock_size"] == 0


def test_resolve_classes():
    coco = {0: "person", 14: "bird", 15: "cat"}
    assert resolve_classes(coco, None) == [14]
    assert resolve_classes({0: "chicken", 1: "feeder"}, None) == [0]
    assert resolve_classes({0: "thing"}, None) is None
    assert resolve_classes(coco, [0]) == [0]


def test_split_has_chickens_in_val():
    from chicken_tracker.labeling import split_train_val

    items = {f"f{i}": {"video": "v", "frame": i, "boxes": [[0, 0, 1, 1]] if i >= 30 else []}
             for i in range(130)}
    s = split_train_val(items)
    val = [k for k, v in s.items() if v == "val"]
    assert any(items[k]["boxes"] for k in val) and any(not items[k]["boxes"] for k in val)
    assert 0.15 < len(val) / len(items) < 0.25


def test_export_with_objects(tmp_path):
    import json
    import cv2
    from chicken_tracker.labeling import LabelStore, export_yolo

    (tmp_path / "images").mkdir()
    items = {}
    for i in range(10):
        name = f"v__f{i:07d}.jpg"
        cv2.imwrite(str(tmp_path / "images" / name), np.zeros((100, 200, 3), np.uint8))
        items[name] = {"video": "v", "frame": i * 100, "time_s": i * 10, "proposals": [],
                       "boxes": [[10, 10, 30, 30]], "status": "done"}
    (tmp_path / "labels.json").write_text(json.dumps(items))
    data = export_yolo(tmp_path)
    assert "0: chicken" in data.read_text() and "feeder" not in data.read_text()

    st = LabelStore(tmp_path)
    for k in list(st.items)[:6]:
        st.update_objects(k, "done", [[50, 50, 90, 70, 1], [100, 20, 120, 60, 2]])
    data = export_yolo(tmp_path)
    assert "1: feeder" in data.read_text() and "2: water" in data.read_text()
    labels = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(labels) == 6                             # frames without the objects pass are left out
    classes = sorted(int(l.split()[0]) for f in labels for l in f.read_text().splitlines())
    assert classes.count(0) == 6 and classes.count(1) == 6 and classes.count(2) == 6
