from chicken_tracker import daylight as DL


def test_stretches_finds_infrared_and_ignores_short_dips():
    ts = [i * 10.0 for i in range(20)]
    sat = [45.0] * 5 + [0.0] + [45.0] * 4 + [0.0] * 10  # one 10 s dip, then infrared to the end
    assert DL.stretches(ts, sat) == [[95.0, 195.0]]


def test_infrared_tracks_by_share_of_time():
    data = {"fragments": {"1": {"start": 0, "end": 50}, "2": {"start": 90, "end": 150}, "3": {"start": 120, "end": 120}}}
    assert DL.infrared_tracks(data, [[100.0, 200.0]]) == {2, 3}
    assert DL.infrared_tracks(data, []) == set()


def test_in_infrared_per_frame():
    assert DL.in_infrared([0, 15, 30], 15.0, [[0.5, 1.5]]).tolist() == [False, True, False]
