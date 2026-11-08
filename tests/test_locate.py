import json

from chicken_tracker import timeline as TL
from chicken_tracker.hub import player_video


def _rec(base, name="103150-110416_ch04.mp4"):
    p = base / "DEV" / "2026-11-03" / name
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    return p


def test_locate_as_saved(tmp_path):
    p = _rec(tmp_path / "anywhere")
    assert TL.locate(str(p), tmp_path / "output") == p


def test_locate_copied_output_finds_videos_beside_it(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    p = _rec(tmp_path / "videos")
    mac = "/Users/someone/downloads/DEV/2026-11-03/103150-110416_ch04.mp4"
    assert TL.locate(mac, out) == p
    assert TL.locate(mac.replace("1031", "0000"), out) is None
    assert TL.locate(None, out) is None


def test_download_roots_and_player_fall_back_to_the_recording(tmp_path):
    out = tmp_path / "output"
    run = out / "2026-11-03" / "103150-110416_ch04"
    run.mkdir(parents=True)
    p = _rec(tmp_path / "videos")
    (run / "meta.json").write_text(json.dumps({"video": {"path": "/elsewhere/DEV/2026-11-03/" + p.name}}))
    assert TL.download_roots(out) == [tmp_path / "videos" / "DEV"]
    assert player_video(run, out) == p
    b = run / "browser" / "video_1280.mp4"
    b.parent.mkdir()
    b.write_bytes(b"y")
    assert player_video(run, out) == b
