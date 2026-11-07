from chicken_tracker import timeline as T


def _clip(start, end):
    return {"start": start, "end": end}


def test_phases_join_files_and_fold_glitches():
    clips = [_clip("2026-11-02T05:00:00", "2026-11-02T05:30:00"), _clip("2026-11-02T05:30:00", "2026-11-02T06:00:00"),
             _clip("2026-11-02T06:00:00", "2026-11-02T06:30:00")]
    lights = [{"first": "night", "switches": [{"t": 600.0, "to": "day"}]},           # dawn at 05:10
              {"first": "day", "switches": [{"t": 60.0, "to": "night"}, {"t": 120.0, "to": "day"}]},  # 1 min glitch
              {"first": "day", "switches": []}]
    ph = T._phases(clips, lights)
    assert [(a.isoformat(), b.isoformat(), p) for a, b, p in ph] == [
        ("2026-11-02T05:00:00", "2026-11-02T05:10:00", "night"), ("2026-11-02T05:10:00", "2026-11-02T06:30:00", "day")]


def test_asking_the_timeline():
    tl = {"phases": [["2026-11-02T00:00:00", "2026-11-02T05:37:56", "night"], ["2026-11-02T05:37:56", "2026-11-02T15:18:05", "day"]],
          "days": [{"day": "2026-11-02", "from": "2026-11-02T05:37:56", "to": "2026-11-03T05:42:37"}],
          "clips": [{"name": "a.mp4", "first": "day", "switches": [{"t": 867.2, "to": "night"}], "duration_s": 1954.0}]}
    assert T.phase_at(tl, "2026-11-02T05:00:00") == "night"
    assert T.phase_at(tl, "2026-11-02T12:00:00") == "day"
    assert T.phase_at(tl, "2026-11-04T12:00:00") is None
    assert T.day_of(tl, "2026-11-03T02:00:00") == "2026-11-02"
    assert T.clip_phases(tl, "a.mp4") == [(0.0, 867.2, "day"), (867.2, 1954.0, "night")]
