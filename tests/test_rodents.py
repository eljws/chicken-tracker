import json

import pytest

from chicken_tracker import rodents as RD


def test_event_id_is_stable_and_tells_movers_apart():
    a = RD.event_id("2026-11-02/220535-223801_ch04.mp4", {"t": 375.04, "box": [283, 170, 293, 179]})
    b = RD.event_id("2026-11-02/220535-223801_ch04.mp4", {"t": 375.04, "box": [100, 50, 110, 60]})
    assert a == "220535-223801_ch04@375.0:288,174" and a != b


def test_labels_are_saved_and_checked(tmp_path):
    RD.set_label(tmp_path, "x@1.0:1,1", "rat", by="claude")
    RD.set_label(tmp_path, "y@2.0:1,1", "nothing")
    lab = json.loads((tmp_path / "rodents" / "labels.json").read_text())
    assert lab["x@1.0:1,1"]["by"] == "claude" and lab["y@2.0:1,1"]["label"] == "nothing"
    RD.set_label(tmp_path, "x@1.0:1,1", None)
    assert "x@1.0:1,1" not in RD.labels(tmp_path)
    with pytest.raises(ValueError):
        RD.set_label(tmp_path, "z", "dragon")
