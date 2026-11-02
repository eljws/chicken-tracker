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
