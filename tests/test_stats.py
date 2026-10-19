import numpy as np
import pandas as pd

from memrot.stats import exceeds, mean_ci, paired_diff


def frame(vals):
    return pd.DataFrame([{"user": u, "attr": a, "v": x} for u, row in enumerate(vals) for a, x in enumerate(row)])


def test_mean_ci_constant():
    m, lo, hi = mean_ci(frame([[1, 1], [1, 1], [1, 1]]), "v")
    assert m == lo == hi == 1


def test_mean_ci_contains_mean_and_is_seeded():
    df = frame(np.random.default_rng(1).integers(0, 2, size=(20, 6)))
    m, lo, hi = mean_ci(df, "v")
    assert lo <= m <= hi
    assert mean_ci(df, "v") == (m, lo, hi)


def test_resamples_users_not_items():
    # two users, all-0 and all-1: any bootstrap draw is a mix of whole users
    df = frame([[0] * 6, [1] * 6])
    m, lo, hi = mean_ci(df, "v", n_boot=500)
    assert m == 0.5 and lo == 0 and hi == 1


def test_paired_diff():
    a = frame([[1, 0], [1, 1], [0, 0]])
    assert paired_diff(a, a, "v")["diff"] == 0
    b = frame([[0, 0], [0, 0], [0, 0]])
    d = paired_diff(a, b, "v")
    assert abs(d["diff"] - 0.5) < 1e-9 and d["n_items"] == 6


def test_exceeds():
    assert exceeds({"diff": 0.3, "lo": 0.1, "hi": 0.5}, {"diff": 0.05})
    assert not exceeds({"diff": 0.3, "lo": -0.1, "hi": 0.5}, {"diff": 0.05})
    assert not exceeds({"diff": 0.04, "lo": 0.01, "hi": 0.07}, {"diff": 0.05})
