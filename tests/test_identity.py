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
