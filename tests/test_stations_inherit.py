import json, os, time

from chicken_tracker.stations import inherit_stations


def test_new_run_copies_latest_stations(tmp_path):
    old, new_src = tmp_path / "2026-11-02" / "a", tmp_path / "2026-11-03" / "b"
    for d, st in ((old, [{"id": "water 1"}]), (new_src, [{"id": "feeder 1"}, {"id": "water 1"}])):
        d.mkdir(parents=True)
        (d / "stations.json").write_text(json.dumps(st))
    os.utime(old / "stations.json", (time.time() - 100, time.time() - 100))
    run = tmp_path / "2026-11-04" / "c"
    run.mkdir(parents=True)
    assert inherit_stations(run, tmp_path / "2026-11-04")
    got = json.loads((run / "stations.json").read_text())
    assert [s["id"] for s in got] == ["feeder 1", "water 1"] and got[0]["copied_from"] == "b"
    assert inherit_stations(run, tmp_path / "2026-11-04") is None   # never overwrites
