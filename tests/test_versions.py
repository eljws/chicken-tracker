import json

import numpy as np
import pandas as pd

from chicken_tracker import identity as I, recognizer as R, versions as V

CLASSES = ["Ginger", "Pepper", "several"]


def frag(tid, start, end):
    return I.Frag(tid, start, end, (0, 0), (0, 0), 10, np.array([1.0, 0, 0]))


def pred(rows):  # [(track, p_Ginger, p_Pepper, p_several)]
    return pd.DataFrame([{"track_id": t, "crowd": 0.0, "p_Ginger": g, "p_Pepper": p, "p_several": s} for t, g, p, s in rows])


def test_reguess_moves_only_unconfirmed_pieces():
    frags = {1: frag(1, 0, 10), 2: frag(2, 20, 30), 3: frag(3, 40, 50), 4: frag(4, 0, 10), 5: frag(5, 60, 61)}
    data = {"chickens": [{"id": 0, "name": "Ginger", "tracks": [1, 2, 3, 5]}, {"id": 1, "name": "Pepper", "tracks": [4]}],
            "answers": [{"track": 1, "chicken": 0, "verdict": "yes"}], "multiple": []}
    # the model now says: 1 is Pepper (but you confirmed Ginger), 2 Pepper, 3 several, 4 Ginger (but
    # Ginger is busy then with confirmed piece 1, so 4 stays Pepper); 5 has no crop
    p = pred([(1, .1, .9, 0), (2, .1, .9, 0), (3, .1, .2, .7), (4, .8, .2, 0)])
    r = R.reguess(data, frags, p, CLASSES, weights="w.pt", version="new")
    g, pe = data["chickens"][0]["tracks"], data["chickens"][1]["tracks"]
    assert 1 in g and 5 in g            # confirmed / no crop: untouched
    assert 2 in pe and 4 in pe          # re-guessed; 4 can't be Ginger at the time of confirmed 1
    assert data["multiple"] == [3]
    assert r["reguessed"] == 3 and data["auto_named"]["version"] == "new"
    assert I.confirmed(data) == {1}     # nothing became confirmed


def test_versions_of_a_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "identity").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"track_config": {"model": "records/models/detector/chicken_clean.pt"}}))
    (run / "identity" / "identity.json").write_text(json.dumps(
        {"fragments": {"1": {}, "2": {}}, "chickens": [], "answers": [{"track": 1, "chicken": 0, "verdict": "yes"}],
         "auto_named": {"weights": "records/models/recognizer/loop-r3-20261105-0249.pt"}}))
    (run / "poses.csv").write_text("x\n")
    (run / "poses.json").write_text(json.dumps({"weights": "records/models/pose/pose-20261103.pt"}))
    (run / "pecks.csv").write_text("x\n")
    monkeypatch.setattr(V, "latest", lambda: {"detect": "chicken_clean", "pieces": "v1", "naming": "all-r4-20261107-0209",
                                              "skeletons": "pose-20261103", "pecks": "v1"})
    v = V.of_run(run, prog={})
    assert v["detect"]["current"] and v["skeletons"]["current"] and v["pecks"]["current"]
    assert v["naming"]["current"] is False and v["naming"]["can_update"] and v["naming"]["latest_short"] == "all-r4"
    assert not v["detect"]["can_update"]
    # everything confirmed: nothing a newer recogniser could change
    d = json.loads((run / "identity" / "identity.json").read_text())
    d["answers"].append({"track": 2, "chicken": 0, "verdict": "yes"})
    (run / "identity" / "identity.json").write_text(json.dumps(d))
    assert V.of_run(run, prog={})["naming"]["current"] is True
