import numpy as np

from chicken_tracker import social as S


def _walk(n=60):
    """Three hens: A and B side by side all along, B trailing A by 3 s on the same path; C far away."""
    t = np.arange(n)
    xy = np.full((n, 3, 2), np.nan)
    path = lambda s: np.stack([0.4 * s, np.zeros_like(s, float)], -1)  # 0.4 m/s along x
    xy[:, 0] = path(t)
    xy[:, 1] = path(t - 1.5) + [0, 0.1]
    xy[:, 2] = [[10.0, 10.0]] * n
    return t, xy, ["Monster", "Salt", "Pepper"]


def test_friends_follow_and_alone():
    t, xy, hens = _walk()
    r = S.analyse(t, xy, hens, start_s=8 * 3600)
    assert r["together"]["Monster"]["Salt"] == r["covisible"]["Monster"]["Salt"] == 60
    assert r["together"]["Monster"]["Pepper"] == 0
    assert max(r["nearest"]["Monster"], key=r["nearest"]["Monster"].get) == "Salt"
    assert r["alone"]["Pepper"] == 60 and r["alone"]["Monster"] == 0
    assert r["follows"]["Salt"]["Monster"] > 30 and r["follows"]["Monster"]["Salt"] == 0


def test_displacement():
    n = 20
    t = np.arange(n)
    xy = np.full((n, 2, 2), np.nan)
    xy[:, 1] = [2.0, 0.0]                      # B stands still...
    xy[:, 0] = [[max(0.0, 1.6 - 0.5 * max(0, 8 - s)), 0.0] for s in t]  # ...A walks up to 0.4 m by t=8 and stays
    for s in range(9, n):
        xy[s, 1] = [2.0 + 0.6 * (s - 8), 0.0]  # B then leaves
    r = S.analyse(t, xy, ["Monster", "Ginger"], start_s=0)
    assert r["displaces"]["Monster"]["Ginger"] == 1 and r["displaces"]["Ginger"]["Monster"] == 0
    assert r["events"][0]["a"] == "Monster"


def test_chase_one_way_only():
    """A runs 0.8 m behind B along the same line at 1 m/s for 2 s: A chases B, never B chases A.
    Two hens running side by side are not a chase."""
    step = S.CHASE_STEP_S
    ts = np.arange(0, 6, step)
    xy = np.full((len(ts), 3, 2), np.nan)
    run = np.clip(ts - 2, 0, 2)                       # running from t=2 to t=4
    xy[:, 1] = np.stack([run + 0.8, np.zeros_like(ts)], -1)
    xy[:, 0] = np.stack([run, np.zeros_like(ts)], -1)
    xy[:, 2] = np.stack([run + 0.8, np.full_like(ts, 0.5)], -1)  # beside B the whole way
    ev = S.chases(ts, xy, ["Monster", "Salt", "Pepper"])
    pairs = {(e["a"], e["b"]) for e in ev}
    assert ("Monster", "Salt") in pairs and ("Salt", "Monster") not in pairs
    assert ("Salt", "Pepper") not in pairs and ("Pepper", "Salt") not in pairs
