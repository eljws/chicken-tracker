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
    import cv2, json
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
    assert "feeder" not in export_yolo(tmp_path).read_text()  # chickens only unless asked
    data = export_yolo(tmp_path, with_objects=True)
    assert "1: feeder" in data.read_text() and "2: water" in data.read_text()
    labels = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(labels) == 6                             # frames without the objects pass are left out
    classes = sorted(int(l.split()[0]) for f in labels for l in f.read_text().splitlines())
    assert classes.count(0) == 6 and classes.count(1) == 6 and classes.count(2) == 6


def test_player_server_byte_ranges(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from chicken_tracker.player import make_handler

    video = tmp_path / "v.mp4"
    video.write_bytes(bytes(range(256)) * 40)            # 10240 bytes
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, video, None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/video.mp4"
    try:
        def get(rng=None):
            req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers, r.read()
        st, h, body = get()
        assert st == 200 and len(body) == 10240 and h["Accept-Ranges"] == "bytes"
        st, h, body = get("bytes=0-1")                      # Safari's first probe
        assert st == 206 and body == bytes([0, 1]) and h["Content-Range"] == "bytes 0-1/10240"
        st, h, body = get("bytes=10000-")
        assert st == 206 and len(body) == 240 and body[0] == 10000 % 256
        st, h, body = get("bytes=-16")                      # last 16 bytes
        assert st == 206 and len(body) == 16 and h["Content-Range"] == "bytes 10224-10239/10240"
        try:
            get("bytes=20000-")
            raise AssertionError("expected 416")
        except urllib.error.HTTPError as e:
            assert e.code == 416
    finally:
        srv.shutdown()


def test_split_keeps_old_frames_when_crowded_frames_are_added():
    from chicken_tracker.labeling import split_train_val
    items = {f"a{i}": {"video": "v", "frame": i, "boxes": [[0, 0, 1, 1]]} for i in range(40)}
    before = split_train_val(items)
    items |= {f"c{i}": {"video": "w", "frame": i, "boxes": [[0, 0, 1, 1]], "source": "crowded"} for i in range(40)}
    after = split_train_val(items)
    assert all(after[k] == v for k, v in before.items())
    assert sum(after[f"c{i}"] == "val" for i in range(40)) == 10


def test_mask_to_outline_keeps_clicked_piece():
    from chicken_tracker.outline import mask_to_outline
    m = np.zeros((100, 200), np.uint8)
    m[10:40, 20:60] = 1      # the hen clicked
    m[60:90, 120:190] = 1    # a separate blob (neighbour) in the same mask
    out = mask_to_outline(m, [30, 20])
    assert out["box"] == [20.0, 10.0, 60.0, 40.0] and len(out["poly"]) >= 4
    assert mask_to_outline(m, [5, 95])["box"][0] == 120.0   # off-mask click: biggest piece
    assert mask_to_outline(np.zeros((10, 10), np.uint8), [1, 1]) is None


def test_review_server_outline_and_saved_outlines(tmp_path):
    import threading, urllib.request
    import cv2, json
    from http.server import ThreadingHTTPServer
    from chicken_tracker.labeling import LabelStore
    from chicken_tracker.review_server import make_handler
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.zeros((50, 50, 3), np.uint8))
    (tmp_path / "labels.json").write_text(json.dumps({"a.jpg": {"video": "v", "frame": 0, "time_s": 0,
                                                                "proposals": [], "boxes": [], "status": "pending"}}))
    store = LabelStore(tmp_path)

    class Fake:
        def outline(self, image, points, labels):
            assert labels == [1, 0]
            return {"box": [1.0, 2.0, 3.0, 4.0], "poly": [[1, 2], [3, 2], [3, 4]]}

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, outliner=Fake()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    post = lambda path, body: json.loads(urllib.request.urlopen(urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})).read())
    try:
        assert json.loads(urllib.request.urlopen(base + "/api/config").read())["outline"] is True
        assert post("/api/outline/a.jpg", {"points": [[5, 5], [9, 9]], "labels": [1, 0]})["box"] == [1.0, 2.0, 3.0, 4.0]
        post("/api/item/a.jpg", {"status": "done", "boxes": [[1, 2, 3, 4], [5, 5, 9, 9]],
                                 "outlines": [[[1, 2], [3, 2], [3, 4]], None]})
        it = LabelStore(tmp_path).items["a.jpg"]
        assert it["outlines"][0] == [[1, 2], [3, 2], [3, 4]] and it["outlines"][1] is None
    finally:
        srv.shutdown()


def test_pose_labels_export_and_shared_saves(tmp_path):
    import cv2, json
    from chicken_tracker.labeling import KEYPOINTS, LabelStore, export_pose
    (tmp_path / "images").mkdir()
    items = {}
    for i in range(6):
        n = f"v__f{i:07d}.jpg"
        cv2.imwrite(str(tmp_path / "images" / n), np.zeros((100, 200, 3), np.uint8))
        items[n] = {"video": "v", "frame": i * 100, "time_s": i, "proposals": [], "status": "done",
                    "boxes": [[10, 10, 50, 50], [100, 10, 140, 50]]}
    (tmp_path / "labels.json").write_text(json.dumps(items))
    pose_page, review_page = LabelStore(tmp_path), LabelStore(tmp_path)
    assert pose_page.init_pose() == 6
    skel = [[20, 20, 2], [25, 22, 2], [30, 30, 2], [35, 35, 2], [0, 0, 0]]  # tail hidden
    for n in items:
        pose_page.update_pose(n, "done", [skel, None])
    review_page.update("v__f0000000.jpg", "done", [[10, 10, 50, 50], [100, 10, 140, 50]])  # stale copy saves
    it = LabelStore(tmp_path).items["v__f0000000.jpg"]
    assert it["poses"][0] == skel and it["pose_status"] == "done"   # not wiped by the other page
    data = export_pose(tmp_path)
    assert f"kpt_shape: [{len(KEYPOINTS)}, 3]" in data.read_text()
    line = next((tmp_path / "yolo_pose" / "labels").rglob("*.txt")).read_text().splitlines()
    assert len(line) == 2 and len(line[0].split()) == 5 + 3 * len(KEYPOINTS)
    assert line[1].split()[5:] == ["0"] * (3 * len(KEYPOINTS))      # skipped hen: box kept, points hidden


def test_overlay_poses_format(tmp_path):
    import pandas as pd
    from chicken_tracker.labeling import KEYPOINTS
    from chicken_tracker.pose import overlay_poses
    assert overlay_poses(tmp_path, ["A"]) is None
    row = {"frame": 7, "track_id": 1, "chicken": "B"}
    for i, k in enumerate(KEYPOINTS):
        row |= {f"{k}_x": 10.4 + i, f"{k}_y": 20.6, f"{k}_c": 0.9 if i else 0.2}   # beak unsure
    pd.DataFrame([row, {**row, "chicken": "Z"}]).to_csv(tmp_path / "poses.csv", index=False)
    d = overlay_poses(tmp_path, ["A", "B"])
    assert list(d["frames"]) == ["7"]                     # unknown hen Z dropped
    f = d["frames"]["7"]
    assert len(f) == 1 + 3 * len(KEYPOINTS) and f[0] == 1  # B is index 1
    assert f[1:4] == [10, 21, 0] and f[4:7] == [11, 21, 1]  # beak hidden (low conf), head shown


def test_pecking_head_moves_body_still(tmp_path):
    import cv2, pandas as pd
    from chicken_tracker.behaviour import find_pecks, motion_signals
    from chicken_tracker.labeling import KEYPOINTS
    fps, n = 10.0, 60
    path = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (200, 120))
    rows = []
    for f in range(n):
        img = np.full((120, 200, 3), 90, np.uint8)
        walking = f >= 40
        bx = 60 + (f - 40) * 6 if walking else 60          # body: still, then walks right
        cv2.circle(img, (bx, 60), 14, (200, 200, 200), -1)
        head_down = (f % 4 == 0) and not walking               # a peck every 0.4 s while standing
        hx, hy = bx + 25, 60 + (10 if head_down else 0)
        cv2.circle(img, (hx, hy), 6, (20, 20, 20), -1)       # dark head on grey ground
        vw.write(img)
        row = {"frame": f, "chicken": "A", "track_id": 1}
        pts = {"beak": (hx + 6, hy), "head": (hx, 65), "neck": (bx + 15, 60), "back": (bx, 60), "tail": (bx - 15, 60)}
        for k in KEYPOINTS:
            row |= {f"{k}_x": pts[k][0], f"{k}_y": pts[k][1], f"{k}_c": 0.9}
        rows.append(row)
    vw.release()
    sig = motion_signals(path, pd.DataFrame(rows), progress=False)
    pecks = find_pecks(sig, fps, [{"id": "feeder 1", "type": "feeder", "box": [80, 55, 100, 80]}])
    standing = pecks[pecks.frame < 40]
    assert 8 <= len(standing) <= 12                            # ~10 pecks in 4 s
    assert len(pecks[pecks.frame >= 41]) == 0                  # walking: body moves, no pecks
    assert (standing["where"] == "feeder 1").all()
