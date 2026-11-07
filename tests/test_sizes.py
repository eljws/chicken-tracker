import numpy as np
import pandas as pd

from chicken_tracker import checks as K, sizes as S


def test_pixels_per_metre_and_expected_area():
    H = np.diag([0.01, 0.01, 1.0])  # 100 px per metre everywhere
    assert np.allclose(S.pixels_per_metre(H, [10, 500], [10, 900]), 100)
    cal = {"a": np.log(50.0) - 2 * np.log(100.0), "b": 2.0, "hens": {"Big": {"factor": 1.5}, "Small": {"factor": 0.8}}}
    assert np.isclose(S.expected_area(cal, H, [0], [0])[0], 50.0)              # the average hen
    assert np.isclose(S.expected_area(cal, H, [0], [0], "Big")[0], 75.0)      # her own size
    assert np.isclose(S.expected_area(cal, H, [0], [0], "Nobody")[0], 50.0)   # unknown: average


def test_size_ratio_uses_the_hens_own_size():
    H = np.diag([0.01, 0.01, 1.0])
    cal = {"a": np.log(100.0) - 2 * np.log(100.0), "b": 2.0, "hens": {"Big": {"factor": 2.0}}}
    rows = pd.DataFrame({"x1": [0.0] * 20, "y1": [0.0] * 20, "x2": [14.0] * 20, "y2": [14.0] * 20})  # area 196
    assert np.allclose(K.size_ratio(rows, None, (cal, H)), 1.96)          # nearly two average birds ...
    assert np.allclose(K.size_ratio(rows, None, (cal, H), "Big"), 0.98)  # ... but just one Big
