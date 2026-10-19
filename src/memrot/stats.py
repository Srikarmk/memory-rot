"""Bootstrap statistics, resampling users (histories), never single answers.

Adapted from agent-interface-lab's task-level bootstrap and A/A gap: here the
resampling unit is the user, because a user's six questions share one history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def boot_idx(n: int, n_boot: int = C.BOOT, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def per_user(df: pd.DataFrame, value: str) -> pd.Series:
    return df.groupby("user")[value].mean().sort_index()


def mean_ci(df: pd.DataFrame, value: str, n_boot: int = C.BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a 0/1 (or numeric) column over items, with a 95% percentile interval
    from resampling users. Users have equal item counts, so the mean of user means
    equals the item mean."""
    u = per_user(df, value).values.astype(float)
    if len(u) == 0:
        return float("nan"), float("nan"), float("nan")
    b = u[boot_idx(len(u), n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(u.mean()), float(lo), float(hi)


def paired_diff(x: pd.DataFrame, ref: pd.DataFrame, value: str, n_boot: int = C.BOOT,
                seed: int = 0) -> dict:
    """x minus ref, paired by (user, attr), with a user-level bootstrap interval."""
    m = x[["user", "attr", value]].merge(ref[["user", "attr", value]], on=["user", "attr"], suffixes=("_x", "_r"))
    if m.empty:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_items": 0, "n_users": 0}
    m["d"] = m[f"{value}_x"].astype(float) - m[f"{value}_r"].astype(float)
    u = m.groupby("user")["d"].mean().values
    b = u[boot_idx(len(u), n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"diff": float(u.mean()), "lo": float(lo), "hi": float(hi), "n_items": int(len(m)),
            "n_users": int(len(u)), "abs95": float(np.percentile(np.abs(b), 95))}


def exceeds(effect: dict, aa: dict) -> bool:
    """An effect counts only if its interval excludes zero and it is larger than the A/A gap."""
    if not effect or not aa or np.isnan(effect["diff"]):
        return False
    excl = effect["lo"] > 0 or effect["hi"] < 0
    return bool(excl and abs(effect["diff"]) > max(abs(aa["diff"]), 0.0))
