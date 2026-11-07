import pytest
import numpy as np

from chicken_tracker import identity as I


def frag(tid, start, end, emb, xy0=(0, 0), xy1=(0, 0), n=None):
    e = np.asarray(emb, float)
    return I.Frag(tid, start, end, xy0, xy1, n or int((end - start) * 10) + 1, e / np.linalg.norm(e))


A, B = [1, 0, 0], [0, 1, 0]


def test_cluster_by_appearance():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 20, 30, A), 3: frag(3, 0, 10, B), 4: frag(4, 20, 30, B)}
    assert sorted(I.cluster(fr)) == [[1, 2], [3, 4]]


def test_time_overlap_never_merges():
    # identical appearance but visible at the same time -> two different chickens
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 5, 15, A)}
    assert len(I.cluster(fr)) == 2


def test_continuity_links_similar_fragments_across_short_gap():
    # slightly different looks (lighting), but B continues right where A vanished
    near = [0.75, 0.66, 0]
    fr = {1: frag(1, 0, 10, A, xy1=(100, 100)), 2: frag(2, 12, 20, near, xy0=(120, 100))}
    sim = float(fr[1].emb @ fr[2].emb)
    assert sim < 0.80  # would not merge on looks alone
    assert I.cluster(fr) == [[1, 2]]


def test_continuity_rejects_teleporting():
    fr = {1: frag(1, 0, 10, A, xy1=(0, 0)), 2: frag(2, 10.5, 20, A, xy0=(1500, 1000))}
    assert I._continuity([fr[1]], [fr[2]]) == 0.0


def _data(fr, clusters):
    return {"fps": 10, "answers": [],
            "chickens": [{"id": i, "name": "", "tracks": c} for i, c in enumerate(clusters)],
            "fragments": {str(t): {"start": f.start, "end": f.end, "n": f.n, "crops": [f"t{t}.jpg"],
                                   "margin": 0.0, "own_sim": 1.0, "alt_cluster": None} for t, f in fr.items()}}


def test_answers_and_merge():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 20, 30, A), 3: frag(3, 0, 10, B)}
    d = _data(fr, [[1], [2], [3]])
    I.rename(d, 0, "Rocky")
    I.merge_chickens(d, 1, 0)
    assert d["chickens"][0] == {"id": 0, "name": "Rocky", "tracks": [1, 2]}
    I.apply_answer(d, 3, None, "not_chicken")
    assert 3 in d["rejected"] and all(3 not in c["tracks"] for c in d["chickens"])
    I.apply_answer(d, 2, None, "new")
    assert [c["tracks"] for c in d["chickens"]] == [[1], [2]]
    assert 2 in I.confirmed(d)


def test_resort_moves_unconfirmed_to_best_match():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 20, 30, A), 3: frag(3, 0, 10, B), 4: frag(4, 40, 50, A)}
    d = _data(fr, [[1, 2], [3, 4]])  # 4 is misplaced
    I.apply_answer(d, 1, 0, "yes")
    moved = I.resort(d, fr)
    assert moved == 1 and sorted(d["chickens"][0]["tracks"]) == [1, 2, 4]


def test_multiple_and_not_chicken_are_distinct():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 20, 30, A), 3: frag(3, 40, 50, A)}
    d = _data(fr, [[1, 2, 3]])
    I.apply_answer(d, 1, None, "multiple")
    I.apply_answer(d, 2, None, "not_chicken")
    assert d["multiple"] == [1] and d["rejected"] == [2]
    assert I.excluded(d) == {1, 2} and d["chickens"][0]["tracks"] == [3]
    I.apply_answer(d, 1, 0, "moved")  # changed mind: it was one chicken after all
    assert d["multiple"] == [] and d["chickens"][0]["tracks"] == [1, 3]


def test_conflicts_found_after_manual_merge():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 5, 15, A), 3: frag(3, 20, 30, A)}
    d = _data(fr, [[1, 3], [2]])
    assert I.conflicts(d) == []
    I.merge_chickens(d, 1, 0)  # user says "same bird" although 1 and 2 overlap 5 s
    assert I.conflicts(d) == [{"chicken": 0, "a": 1, "b": 2, "overlap_s": 5.0}]
    I.apply_answer(d, 2, None, "not_chicken")  # e.g. it was the dog
    assert I.conflicts(d) == []



def test_conflict_needs_real_shared_frames():
    """A piece inside another piece's gap (same hen lost for a few seconds) isn't a conflict."""
    import numpy as np
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 4, 6, A)}
    d = _data(fr, [[1, 2]])
    d["fps"] = 10.0
    assert len(I.conflicts(d)) == 1  # by start..end alone they overlap 2 s
    gap = {1: np.r_[np.arange(0, 35), np.arange(70, 100)], 2: np.arange(40, 60)}
    assert I.conflicts(d, seen=gap) == []  # 2 sits in 1's gap: never on screen together
    both = {1: np.arange(0, 100), 2: np.arange(40, 60)}
    assert I.conflicts(d, seen=both) == [{"chicken": 0, "a": 1, "b": 2, "overlap_s": 2.0, "from_s": 4.0, "to_s": 5.9}]
    flicker = {1: np.arange(0, 100, 3), 2: np.arange(40, 60)}  # 1's box only every 3rd frame: still together
    assert len(I.conflicts(d, seen=flicker)) == 1

def _det(rows):
    import pandas as pd
    return pd.DataFrame([{"frame": f, "time_s": f / 10, "track_id": t, "x1": 0, "y1": 0, "x2": 1, "y2": 1,
                          "cx": 0, "cy": 0, "conf": 1, "cls": 0} for t, f in rows])


def test_cut_piece_in_two():
    det = _det([(5, f) for f in range(0, 100)] + [(6, f) for f in range(0, 50)])
    fr = {5: frag(5, 0, 9.9, A), 6: frag(6, 0, 4.9, B)}
    d = _data(fr, [[5], [6]])
    I.apply_answer(d, 5, 0, "yes")
    nid, det2 = I.register_split(d, det, 5, 60)
    assert nid >= I.SPLIT_ID_BASE
    assert det2[det2.track_id == 5].frame.max() == 59 and det2[det2.track_id == nid].frame.min() == 60
    assert d["chickens"][0]["tracks"] == [5, nid]
    assert 5 not in I.confirmed(d)                      # old answer no longer trusted
    assert d["fragments"][str(nid)]["start"] == 6.0
    # replaying recorded cuts on raw detections gives the same result
    assert I.apply_splits(det, d["splits"]).equals(det2)
    # a half can be cut again
    nid2, det3 = I.register_split(d, det2, nid, 80)
    assert nid2 == nid + 1 and set(det3.track_id) == {5, 6, nid, nid2}


def test_cut_outside_piece_rejected():
    import pytest
    det = _det([(5, f) for f in range(10, 20)])
    d = _data({5: frag(5, 1, 1.9, A)}, [[5]])
    with pytest.raises(ValueError):
        I.register_split(d, det, 5, 10)   # nothing before the cut
    with pytest.raises(ValueError):
        I.register_split(d, det, 5, 25)   # nothing after


def test_partial_piece_stays_with_chicken_and_is_not_a_conflict():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 2, 8, A)}  # 2 = only Monster's head, while 1 = whole Monster
    d = _data(fr, [[1, 2]])
    assert len(I.conflicts(d)) == 1
    I.apply_answer(d, 2, 0, "partial")
    assert d["partial"] == [2] and d["chickens"][0]["tracks"] == [1, 2]
    assert I.conflicts(d) == [] and 2 in I.confirmed(d)
    I.apply_answer(d, 2, None, "multiple")  # changed mind
    assert d["partial"] == [] and 2 not in d["chickens"][0]["tracks"]


def test_export_counts_each_chicken_once_per_frame(tmp_path):
    import json
    det = _det([(1, f) for f in range(0, 100)] + [(2, f) for f in range(20, 80)])
    det.to_csv(tmp_path / "detections.csv", index=False)
    fr = {1: frag(1, 0, 9.9, A), 2: frag(2, 2, 7.9, A)}
    d = _data(fr, [[1, 2]]); d["chickens"][0]["name"] = "Monster"; d["partial"] = [2]
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "identity.json").write_text(json.dumps(d))
    I.export(tmp_path)
    import pandas as pd
    s = pd.read_csv(tmp_path / "chickens.csv")
    assert s.visible_s.iloc[0] == 10.0           # 100 frames @10fps, not 160
    named = pd.read_csv(tmp_path / "detections_named.csv")
    assert set(named.track_id) == {1}             # full box preferred over the partial one
    flagged = pd.read_csv(tmp_path / "flagged_pieces.csv")
    assert flagged.to_dict("records") == [{"track_id": 2, "reason": "partial", "chicken": "Monster"}]


def test_undo_history(tmp_path):
    import json
    run = tmp_path
    (run / "identity").mkdir()
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 20, 30, B)}
    d = _data(fr, [[1], [2]])
    (run / "identity" / "identity.json").write_text(json.dumps(d))
    (run / "identity" / "crops.csv").write_text("v1")
    np.save(run / "identity" / "features.npy", np.zeros(3))
    st = I.IdentityStore(run)
    assert st.history() == {"count": 0, "label": None} and st.undo() is None

    st.snapshot("named Rocky"); I.rename(st.data, 0, "Rocky"); st.save()
    st.snapshot("cut piece #1", with_features=True)
    (run / "identity" / "crops.csv").write_text("v2"); np.save(run / "identity" / "features.npy", np.ones(3))
    I.apply_answer(st.data, 2, None, "multiple"); st.save()
    assert st.history() == {"count": 2, "label": "cut piece #1"}

    assert st.undo() == "cut piece #1"
    assert (run / "identity" / "crops.csv").read_text() == "v1"
    assert np.load(run / "identity" / "features.npy").sum() == 0
    assert I.IdentityStore(run).data["chickens"][1]["tracks"] == [2]   # persisted to disk
    assert st.data["chickens"][0]["name"] == "Rocky"
    assert st.undo() == "named Rocky" and st.data["chickens"][0]["name"] == ""
    assert st.history()["count"] == 0


def test_history_is_capped(tmp_path):
    import json
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "identity.json").write_text(json.dumps(_data({1: frag(1, 0, 1, A)}, [[1]])))
    st = I.IdentityStore(tmp_path)
    st.HISTORY_LIMIT = 5
    for i in range(8):
        st.snapshot(f"change {i}")
    assert st.history() == {"count": 5, "label": "change 7"}


def test_loose_box_flag():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 2, 8, A)}
    d = _data(fr, [[1, 2]])
    I.apply_answer(d, 2, 0, "loose")
    assert d["loose"] == [2] and I.box_flag(d, 2) == "loose" and I.conflicts(d) == []
    I.apply_answer(d, 2, 0, "partial")            # switching flag replaces the old one
    assert d["loose"] == [] and d["partial"] == [2] and I.box_flag(d, 2) == "partial"
    I.apply_answer(d, 2, 0, "yes")                # plain yes clears the flag
    assert I.box_flag(d, 2) is None and len(I.conflicts(d)) == 1


def test_overlap_cuts_isolate_the_overlap():
    det = _det([(1, f) for f in range(0, 100)] + [(2, f) for f in range(40, 60)])
    assert I.overlap_cuts(det, 1, 2) == [40, 60]       # cut before and after the overlap
    assert I.overlap_cuts(det, 2, 1) == []              # piece 2 is entirely inside: nothing to cut
    det = _det([(1, f) for f in range(0, 50)] + [(2, f) for f in range(40, 90)])
    assert I.overlap_cuts(det, 1, 2) == [40]            # overlap at the end of piece 1
    assert I.overlap_cuts(det, 2, 1) == [50]            # overlap at the start of piece 2
    d = _data({1: frag(1, 0, 4.9, A), 2: frag(2, 4, 8.9, A)}, [[1, 2]])
    for f in reversed(I.overlap_cuts(det, 1, 2)):       # applying them leaves exactly the overlap
        _, det = I.register_split(d, det, 1, f)
    new = d["splits"][-1]["new"]
    assert det[det.track_id == new].frame.tolist() == list(range(40, 50))


def test_station_visits():
    import pandas as pd
    from chicken_tracker.stations import station_visits
    fps = 10
    rows = []
    # Rocky at the feeder 0-5 s, away 5-6 s (short gap, merged), back 6-9 s; Salt walks past for 1 s
    for f in list(range(0, 50)) + list(range(60, 90)):
        rows.append({"chicken": "Rocky", "frame": f, "time_s": f / fps, "x1": 100, "y1": 100, "x2": 140, "y2": 140})
    for f in range(0, 10):
        rows.append({"chicken": "Salt", "frame": f, "time_s": f / fps, "x1": 150, "y1": 100, "x2": 190, "y2": 140})
    named = pd.DataFrame(rows)
    st = [{"id": "feeder 1", "type": "feeder", "box": [130, 110, 160, 130]}]
    v = station_visits(named, st, fps, margin=0, gap_s=2.0, min_visit_s=2.0)
    assert v.chicken.tolist() == ["Rocky"]                    # Salt's 1 s pass is too short
    assert v.duration_s.iloc[0] == pytest.approx(9.0)          # 0-8.9 s merged across the 1 s gap


def test_walked_metres_ignores_jitter_and_gaps():
    import pandas as pd
    from chicken_tracker.calibration import walked_metres
    H = np.diag([0.01, 0.01, 1.0])                      # 100 px = 1 m
    rng = np.random.default_rng(0)
    rows = [{"track_id": 1, "frame": f, "cx": 100 + f * 10 + rng.normal(0, 3), "y2": 500 + rng.normal(0, 3)}
            for f in range(50)]                          # walks 4.9 m with +-3 px jitter
    rows += [{"track_id": 2, "frame": 100 + f, "cx": 2000, "y2": 500} for f in range(20)]  # far away, standing
    d = pd.DataFrame(rows)
    m = walked_metres(d, H)
    assert 4.6 < m < 5.3                                  # jitter mostly removed, no jump to track 2


def test_kinematics_dash_and_glitch():
    import pandas as pd
    from chicken_tracker.calibration import kinematics
    H = np.diag([0.01, 0.01, 1.0])           # 100 px = 1 m
    fps = 15.625
    rows = []
    x = 100.0
    for f in range(160):                      # stand 3 s, dash at 2 m/s for 1.5 s, stand again
        t = f / fps
        if 47 <= f < 70:
            x += 200 / fps
        rows.append({"track_id": 1, "frame": f, "time_s": t, "cx": x, "y2": 500.0})
    k = kinematics(pd.DataFrame(rows), H)
    assert 1.8 < k["top_speed_m_s"] < 2.1
    assert 1.5 < k["top_speed_at_s"] < 4.6
    assert k["top_accel_m_s2"] > 2.0 and k["top_speed_piece"] == 1
    # a one-frame 3 m teleport (tracker glitch) must not become the top speed
    rows2 = [dict(r, track_id=2) for r in rows[:40]]
    rows2[20]["cx"] += 300
    k2 = kinematics(pd.DataFrame(rows2), H)
    assert np.isnan(k2["top_speed_m_s"]) or k2["top_speed_m_s"] < 0.5


def test_checks_find_single_bird_in_aside_and_merged_box():
    import pandas as pd
    from chicken_tracker.checks import run_checks
    fps = 10.0
    rows = []
    # Named reference birds (box 40x40 at y2=400) to learn the single-bird size
    for f in range(200):
        rows.append({"frame": f, "time_s": f / fps, "track_id": 1, "x1": 0, "y1": 360, "x2": 40, "y2": 400})
    # piece 2: named, box doubles in size between frames 50-100 (two birds)
    for f in range(150):
        w = 80 if 50 <= f < 100 else 40
        rows.append({"frame": f, "time_s": f / fps, "track_id": 2, "x1": 200, "y1": 360, "x2": 200 + w, "y2": 400})
    # piece 3: set aside as several chickens, but one bird's size for 10 s
    for f in range(100):
        rows.append({"frame": 300 + f, "time_s": (300 + f) / fps, "track_id": 3, "x1": 400, "y1": 360, "x2": 440, "y2": 400})
    det = pd.DataFrame(rows)
    named = det[det.track_id.isin([1, 2])].assign(chicken="X")
    fr = {1: frag(1, 0, 19.9, A), 2: frag(2, 0, 14.9, A), 3: frag(3, 30, 39.9, A)}
    d = _data(fr, [[1], [2]]); d["multiple"] = [3]
    picks = pd.DataFrame({"track_id": [1, 2, 3], "frame": [10, 10, 310]})
    feats = np.stack([fr[1].emb, fr[2].emb, fr[3].emb])
    checks = run_checks(d, det, named, picks, feats, fr, fps)
    kinds = {(c["kind"], c["track"]) for c in checks}
    assert ("merged_inside", 2) in kinds and ("aside_single", 3) in kinds
    merged = next(c for c in checks if c["kind"] == "merged_inside")
    assert 45 <= merged["from_frame"] <= 55 and 95 <= merged["to_frame"] <= 105
    d["dismissed_checks"] = [merged["key"]]
    assert all(c["key"] != merged["key"] for c in run_checks(d, det, named, picks, feats, fr, fps))


def test_recognizer_checks_mislabel_and_stretch():
    import pandas as pd
    from chicken_tracker.checks import apply_check, run_checks
    fps = 10.0
    rows = []
    for t, f0 in ((1, 0), (2, 300), (3, 600)):  # three named single-bird pieces, 20 s each
        for f in range(200):
            rows.append({"frame": f0 + f, "time_s": (f0 + f) / fps, "track_id": t,
                         "x1": 0, "y1": 360, "x2": 40, "y2": 400})
    det = pd.DataFrame(rows)
    fr = {1: frag(1, 0, 19.9, A), 2: frag(2, 30, 49.9, A), 3: frag(3, 60, 79.9, B)}
    d = _data(fr, [[1, 2], [3]])
    I.rename(d, 0, "Rocky"); I.rename(d, 1, "Monster")
    named = det.assign(chicken="Rocky")
    picks = pd.DataFrame({"track_id": [1, 2, 3], "frame": [10, 310, 610]})
    feats = np.stack([fr[1].emb, fr[2].emb, fr[3].emb])
    classes = ["Monster", "Rocky", "several"]
    # 1 crop/s: piece 1 Rocky throughout, piece 2 all Monster (mislabel), piece 3 Monster then Rocky (swap)
    pr = []
    for t, f0 in ((1, 0), (2, 300), (3, 600)):
        for k in range(20):
            who = {1: "Rocky", 2: "Monster", 3: "Monster" if k < 10 else "Rocky"}[t]
            p = [0.05, 0.05, 0.05]; p[classes.index(who)] = 0.9
            pr.append({"frame": f0 + k * 10, "track_id": t, "crowd": 0.0, **{f"p_{c}": v for c, v in zip(classes, p)}})
    recog = (pd.DataFrame(pr), classes)
    checks = run_checks(d, det, named, picks, feats, fr, fps, recog=recog)
    by = {(c["kind"], c["track"]): c for c in checks}
    assert ("recognizer", 2) in by and by[("recognizer", 2)]["chicken"] == 1
    st = by[("stretch", 3)]  # Monster's piece turns into Rocky half-way: cut from ~1:10 to the end, to Rocky
    assert 690 <= st["from_frame"] <= 700 and st["to_frame"] == 799 and st["chicken"] == 0
    assert not any(c["track"] == 1 for c in checks)
    apply_check(by[("recognizer", 2)], d, det, None, None)
    assert 2 in next(c for c in d["chickens"] if c["name"] == "Monster")["tracks"]


def test_rec_stretches_tolerate_one_right_crop():
    from chicken_tracker.checks import _rec_stretches
    w = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], bool)
    assert _rec_stretches(w) == [(2, 7)]
    assert _rec_stretches(np.array([1, 0, 1, 0, 1, 0], bool)) == []  # joined, but only 5 crops at 60%


def test_assign_pieces_one_hen_in_one_place():
    from chicken_tracker.recognizer import assign_pieces
    classes = ["Pepper", "Salt", "several"]
    fr = {1: frag(1, 0, 20, A), 2: frag(2, 5, 15, A), 3: frag(3, 30, 40, A), 4: frag(4, 50, 60, A)}
    verdict = {1: (np.array([0.05, 0.9, 0.05]), 20),    # sure Salt
               2: (np.array([0.3, 0.6, 0.1]), 10),      # looks Salt, but Salt is busy (piece 1) -> Pepper
               3: (np.array([0.1, 0.8, 0.1]), 10),      # Salt again, later: fine
               4: (np.array([0.1, 0.1, 0.8]), 10)}      # two birds
    tracks, several, rep = assign_pieces(verdict, fr, classes)
    assert tracks == [[2], [1, 3]] and several == [4]
    assert rep == {"named": 2, "second_choice": 1, "several": 1, "conflict": 0}


def test_clean_ghost_boxes_keeps_naming(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from chicken_tracker import identity as I
    rows = []
    for f in range(3):  # hens 1 and 2, plus ghost 3 around both in every frame
        rows += [(f, f / 10, 1, 10, 10, 30, 30, 20, 20, .9, 0), (f, f / 10, 2, 32, 10, 52, 30, 42, 20, .9, 0),
                 (f, f / 10, 3, 8, 8, 54, 32, 31, 20, .6, 0)]
    cols = ["frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "cls"]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "detections.csv", index=False)
    pd.DataFrame({"frame": [0, 0], "track_id": [1, 3], "chicken": ["Salt", "Salt"]}).to_csv(tmp_path / "poses.csv", index=False)
    pd.DataFrame({"frame": [1, 1], "track_id": [3, 1], "chicken": ["Rocky", "Salt"]}).to_csv(tmp_path / "detections_named.csv", index=False)
    pd.DataFrame({"chicken": ["Rocky", "Salt"], "frame": [1, 1]}).to_csv(tmp_path / "pecks.csv", index=False)
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "identity.json").write_text(json.dumps({"splits": []}))
    exported = []
    monkeypatch.setattr(I, "export", lambda run_dir: exported.append(run_dir))
    r = I.clean_ghost_boxes(tmp_path)
    assert r == {"ghost_boxes": 3, "skeletons_dropped": 1, "pecks_dropped": 1} and exported
    assert set(pd.read_csv(tmp_path / "detections.csv").track_id) == {1, 2}
    assert len(pd.read_csv(tmp_path / "detections_unfiltered.csv")) == 9
    assert list(pd.read_csv(tmp_path / "pecks.csv").chicken) == ["Salt"]


def test_confirm_all_confirms_only_the_rest_of_the_row():
    from chicken_tracker import identity as I
    data = {"answers": [{"track": 1, "chicken": 0, "verdict": "yes"}],
            "chickens": [{"id": 0, "name": "Salt", "tracks": [1, 2, 3]}, {"id": 1, "name": "Rocky", "tracks": [4]}]}
    assert I.confirm_all(data, 0) == 2
    assert I.confirmed(data) == {1, 2, 3}
    assert data["chickens"][0]["tracks"] == [1, 2, 3] and data["chickens"][1]["tracks"] == [4]
    assert I.confirm_all(data, 0) == 0


def test_cant_tell_which_hen_is_set_aside_and_not_asked_again():
    fr = {1: frag(1, 0, 10, A), 2: frag(2, 5, 15, A), 3: frag(3, 40, 50, A)}
    d = _data(fr, [[1, 2, 3]])  # 1 and 2 overlap: a conflict
    assert I.conflicts(d)
    I.apply_answer(d, 2, None, "unsure")
    assert d["unsure"] == [2] and 2 in I.excluded(d) and 2 in I.confirmed(d)
    assert d["chickens"][0]["tracks"] == [1, 3] and not I.conflicts(d)
    assert all(q["track"] != 2 for q in I.questions(d))
    I.apply_answer(d, 2, 0, "moved")  # spotted her after all
    assert d["unsure"] == [] and 2 in d["chickens"][0]["tracks"]


def test_gap_frames_zoom_between_two_strip_pictures():
    import pandas as pd
    det = pd.DataFrame({"track_id": [7] * 100, "frame": range(100)})
    strip = I.strip_frames(det, 7, 12)
    f0, f1 = strip[-2], strip[-1]  # the last gap: cutting at f1 would move just the last frame
    g = I.gap_frames(det, 7, f0, f1)
    assert g[0] == f0 and g[-1] == f1 and len(g) == min(10, f1 - f0 + 1)
    I.check_cut(det, 7, g[3])
    with pytest.raises(ValueError):
        I.check_cut(det, 7, 0)       # nothing before it
    with pytest.raises(ValueError):
        I.check_cut(det, 8, 50)      # no such piece
    with pytest.raises(ValueError, match="only 1 frame"):
        I.check_cut(det, 7, 99, I.MIN_CUT_FRAMES)  # just the last frame: the hen leaving the view, not a new bird


def test_run_events_from_speed_samples():
    from chicken_tracker.metrics import run_events
    dt = 0.064
    t = [i * dt for i in range(100)]
    v = [0.2] * 100
    for i in range(20, 35):          # ~1 s at 1.5 m/s with a one-frame dip: one run
        v[i] = 1.5
    v[27] = 0.6
    for i in range(60, 63):          # 0.19 s: too short to be a run
        v[i] = 2.0
    ev = run_events([(tt, vv, 7) for tt, vv in zip(t, v)])
    assert len(ev) == 1
    assert ev[0]["peak_m_s"] == 1.5 and 0.9 < ev[0]["duration_s"] < 1.0 and ev[0]["piece"] == 7


def test_mark_segments_cut_where_marks_change():
    import pandas as pd
    det = pd.DataFrame({"track_id": [3] * 200, "frame": range(100, 300)})
    strip = I.strip_frames(det, 3, 20)
    marks = [(f, "c0") for f in strip[:8]] + [(f, "unsure") for f in strip[8:12]] + [(f, "c1") for f in strip[12:]]
    cuts, labels = I.mark_segments(det, 3, marks)
    assert labels == ["c0", "unsure", "c1"] and len(cuts) == 2
    assert strip[7] < cuts[0] <= strip[8] and strip[11] < cuts[1] <= strip[12]   # in the gap where it changes
    assert I.mark_segments(det, 3, [(f, "c0") for f in strip]) == ([], ["c0"])  # all the same: no cut
    with pytest.raises(ValueError, match="few frames"):
        I.mark_segments(det, 3, [(f, "c0") for f in range(100, 299)] + [(299, "c2")])  # last part: 1 frame


def test_mark_segments_box_flag_is_its_own_part():
    import pandas as pd
    det = pd.DataFrame({"track_id": [3] * 200, "frame": range(100, 300)})
    strip = I.strip_frames(det, 3, 20)
    marks = [(f, "c0") for f in strip[:10]] + [(f, "c0:partial") for f in strip[10:]]
    cuts, labels = I.mark_segments(det, 3, marks)
    assert labels == ["c0", "c0:partial"] and len(cuts) == 1


def test_trace_follows_nearby_pieces_and_stops_at_anchor():
    import pandas as pd
    rows = []
    def piece(tid, f0, f1, x):
        rows.extend({"track_id": tid, "frame": f, "x1": x, "y1": 100, "x2": x + 50, "y2": 150} for f in range(f0, f1))
    piece(1, 0, 50, 100)      # anchor, far back
    piece(2, 60, 100, 110)    # starts 10 frames after 1 ended, same spot
    piece(3, 105, 150, 115)   # the piece in question
    piece(9, 95, 140, 900)    # another bird, far away: never a link
    det = pd.DataFrame(rows)
    hops = I.trace(det, 3, -1, fps=10, is_anchor=lambda t: t == 1)
    assert [h["track"] for h in hops] == [2, 1] and hops[-1].get("anchor") and not hops[0]["ambiguous"]
    assert hops[0]["gap_s"] == 0.6
    assert I.trace(det, 3, 1, fps=10) == []          # nothing after it
    fwd = I.trace(det, 1, 1, fps=10)
    assert [h["track"] for h in fwd] == [2, 3]


def test_trace_links_sitting_hen_across_long_gap():
    import pandas as pd
    rows = []
    def piece(tid, f0, f1, x):
        rows.extend({"track_id": tid, "frame": f, "x1": x, "y1": 100, "x2": x + 50, "y2": 150} for f in range(f0, f1))
    piece(1, 0, 50, 100)
    piece(2, 200, 300, 102)   # 15 s later, same spot: she sat there unseen
    piece(3, 450, 500, 300)   # 15 s later but 4 bodies away: not linked
    det = pd.DataFrame(rows)
    assert [h["track"] for h in I.trace(det, 2, -1, fps=10)] == [1]
    assert I.trace(det, 2, 1, fps=10) == []


def test_dots_merge_flips_into_one_event():
    from chicken_tracker.dots import _merge_flips
    ev = [{"kind": "teleport", "hen": "Pepper", "from_frame": f, "to_frame": f + 1, "tracks": [148, 157], "detail": ""}
          for f in (100, 102, 106)] + [{"kind": "teleport", "hen": "Pepper", "from_frame": 400, "to_frame": 401,
                                        "tracks": [1, 2], "detail": "x"}]
    out = _merge_flips(ev, fps=15)
    assert len(out) == 2 and out[0]["to_frame"] == 107 and "3 times" in out[0]["detail"]


def test_whereabouts_rules_out_hen_seen_elsewhere_and_finds_dot():
    import pandas as pd
    from chicken_tracker.dots import whereabouts
    rows = []
    def piece(tid, f0, f1, x):
        rows.extend({"track_id": tid, "frame": f, "x1": x, "y1": 100, "x2": x + 50, "y2": 150} for f in range(f0, f1))
    piece(5, 100, 200, 100)   # the piece in question
    piece(6, 90, 210, 800)    # Salt, far away the whole time
    det = pd.DataFrame(rows)
    n = 300
    rocky = {"x": [125] * n, "y": [150] * n, "src": "F" * n}          # her followed dot is right where #5 starts
    dots = {"dots": {"Rocky": rocky, "Salt": {"x": [825] * n, "y": [150] * n, "src": "S" * n}}, "illegal": []}
    w = whereabouts(det, {5: 0, 6: 1}, {0: "Rocky", 1: "Salt"}, 5, dots, fps=10, asked=0)
    assert w["ruled_out"] == ["Salt"] and w["fits"] == ["Rocky"]


def test_box_left_bird_flags_box_that_moves_away_from_pixels():
    import pandas as pd
    from chicken_tracker import dots as D
    rows = [{"track_id": 7, "frame": f, "x1": 100 if f < 20 else 400, "y1": 100, "x2": 150 if f < 20 else 450, "y2": 150}
            for f in range(0, 40)]
    det = pd.DataFrame(rows)
    flow = {"window": {}}
    for f in (0, 10, 20):  # pixels stay where the bird was (x ~125) the whole time
        r = det[det.frame == f].iloc[0]
        flow["window"][D._box_key(f, r)] = [f + 10, 125 if f < 20 else 425, 150]
    ev, checked = D.box_left_bird(det, 7, flow, fps=10)
    assert checked == 3 and len(ev) == 1 and ev[0]["from_frame"] == 10 and 10 <= ev[0]["cut_frame"] <= 20


def test_resize_handles_cutout_that_crashed_opencv():
    import numpy as np
    img = np.random.randint(0, 255, (1536, 1920, 3), np.uint8)
    out = I.resize(img[1269:1536, 556:1101], (544, 266))  # this exact cut-out segfaulted cv2.resize
    assert out.shape == (266, 544, 3)


def test_to_check_marks_are_asked_even_when_short():
    fr = lambda s, e, margin=0.5: {"start": s, "end": e, "margin": margin, "own_sim": 0.9, "alt_cluster": None}
    data = {"fragments": {"1": fr(0, 1), "2": fr(0, 30), "3": fr(40, 41), "4": fr(50, 60, margin=0.0)},
            "chickens": [{"id": 0, "name": "Salt", "tracks": [1, 2, 3, 4]}], "answers": []}
    assert [q["track"] for q in I.questions(data)] == [4]  # short pieces were never asked before
    qs = I.questions(data, extra={1: "seen under 2 s", 2: "recogniser thinks Pepper (90%)"})
    assert [q["track"] for q in qs] == [2, 4, 1] and qs[0]["why"].startswith("recogniser")
    data["answers"].append({"track": 1, "chicken": 0, "verdict": "yes"})  # answered: not asked again
    assert 1 not in [q["track"] for q in I.questions(data, extra={1: "seen under 2 s"})]


def test_looks_fine_confirms_the_piece():
    from chicken_tracker import checks as K
    data = {"chickens": [{"id": 0, "name": "Monster", "tracks": [5, 6]}], "answers": [], "multiple": [7], "loose": [6]}
    K.confirm_on_dismiss({"track": 5}, data)
    K.confirm_on_dismiss({"track": 6}, data)   # keeps her "box too big" flag
    K.confirm_on_dismiss({"track": 7}, data)   # set aside: stays in its pile, now your answer
    assert I.confirmed(data) == {5, 6, 7}
    assert data["loose"] == [6] and data["multiple"] == [7] and 5 in data["chickens"][0]["tracks"]
