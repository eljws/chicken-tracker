import os
from pathlib import Path

import pytest

from chicken_tracker import records as R


def _w(p: Path, data: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_registry_keeps_every_version_and_switches(tmp_path):
    a = _w(tmp_path / "runs/x/best.pt", b"model A")
    va = R.register("detector", a, version="clean")
    assert R.current("detector").read_bytes() == b"model A"      # first one becomes current
    a.write_bytes(b"model B")                                      # training folder reused ...
    vb = R.register("detector", a, version="next", parent=va)
    assert R.current("detector").read_bytes() == b"model A"      # ... the stored copy is untouched
    assert R.register("detector", a) == vb                         # same file again: same version
    R.use("detector", vb)
    assert R.current("detector").read_bytes() == b"model B"
    st = {m["version"]: m["status"] for m in R.models("detector")}
    assert st == {"clean": "previous", "next": "current"}
    assert R.version_of("detector", a) == "next"
    assert R.model_or("detector", None, "yolo11s.pt").endswith("next.pt")
    assert R.model_or("detector", "mine.pt", "yolo11s.pt") == "mine.pt"
    assert R.model_or("pose", None, "fallback.pt") == "fallback.pt"
    with pytest.raises(ValueError):
        R.use("detector", "nope")
    Path(R.current("detector")).write_bytes(b"tampered")
    assert any("changed" in p for p in R.verify())


def test_frozen_split_never_moves_a_frame():
    rule = lambda items: {k: ("val" if k.endswith("3") else "train") for k in items}
    first = R.frozen_splits({f"f{i}": {} for i in range(5)}, rule)
    assert first["f3"] == "val"
    # a later rule would put everything in val: known frames keep their split, new ones follow the rule
    later = R.frozen_splits({f"f{i}": {} for i in range(7)}, lambda items: {k: "val" for k in items})
    assert {k: later[k] for k in first} == first and later["f6"] == "val"
    assert R.seed_splits(["f1", "g1"], ["g2"]) == 2


def test_backup_copies_once_then_links(tmp_path):
    proj = tmp_path / "proj"
    _w(proj / "dataset/labels.json", b"{}")
    _w(proj / "dataset/images/a.jpg", b"jpg" * 100)
    _w(proj / "output/day/clip/identity/identity.json", b"{}")
    _w(proj / "output/day/clip/detections.csv", b"not backed up: can be recreated")
    R.register("pose", _w(tmp_path / "p.pt", b"pose"))
    one = R.backup(proj, reason="test")
    assert one["copied"] >= 4 and one["linked"] == 0
    b1 = Path(one["path"])
    assert (b1 / "dataset/images/a.jpg").exists() and not (b1 / "output/day/clip/detections.csv").exists()
    assert (b1 / "records/records.db").exists()
    (proj / "dataset/labels.json").write_text('{"x": 1}')
    os.utime(proj / "dataset/labels.json", (1e9, 1e9))
    two = R.backup(proj)
    b2 = Path(two["path"])
    assert two["linked"] >= 2 and (b2 / "dataset/labels.json").read_text() == '{"x": 1}'
    assert os.stat(b1 / "dataset/images/a.jpg").st_ino == os.stat(b2 / "dataset/images/a.jpg").st_ino


def test_models_get_plain_numbers_per_kind(tmp_path):
    R.register("detector", _w(tmp_path / "a.pt", b"a"), version="first")
    R.register("recognizer", _w(tmp_path / "r.pt", b"r"), version="hens")
    R.register("detector", _w(tmp_path / "b.pt", b"b"), version="second")
    num = {(m["kind"], m["version"]): m["number"] for m in R.models()}
    assert num == {("detector", "first"): 1, ("detector", "second"): 2, ("recognizer", "hens"): 1}
    R.use("detector", "v2")
    assert R.resolve("detector", "2") == "second" and R.current("detector").stem == "second"
    with pytest.raises(ValueError):
        R.use("detector", "v9")
