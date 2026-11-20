"""Paired statistics over replication IDs (common random numbers)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BOOT_SEED = 30000


def paired_diff(df: pd.DataFrame, a: str, b: str, metric: str = "mean_delay",
                keys=("condition", "rho", "sigma")) -> pd.DataFrame:
    """Per-replication differences a - b within each cell (rows matched by rep)."""
    piv = df.pivot_table(index=list(keys) + ["rep"], columns="policy", values=metric)
    d = (piv[a] - piv[b]).rename("diff").reset_index()
    return d


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = BOOT_SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    bm = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(bm, alpha / 2)), float(np.quantile(bm, 1 - alpha / 2))


def bootstrap_cell_table(d: pd.DataFrame, keys=("condition", "rho", "sigma"), n_boot=10000, seed=BOOT_SEED):
    out = []
    for k, g in d.groupby(list(keys)):
        m, lo, hi = bootstrap_mean_ci(g["diff"].values, n_boot=n_boot, seed=seed)
        x = g["diff"].values
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        # paired sign-flip permutation p-value (two-sided) for mean difference
        p = signflip_p(x, seed=seed)
        out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), mean=m, lo=lo, hi=hi, se=se, n=len(x), p=p))
    return pd.DataFrame(out)


def signflip_p(x: np.ndarray, n_perm: int = 20000, seed: int = BOOT_SEED) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0 or np.all(x == 0):
        return 1.0
    rng = np.random.default_rng(seed + 1)
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, x.size))
    perm = np.abs((signs * x).mean(axis=1))
    return float((1 + np.sum(perm >= obs - 1e-15)) / (n_perm + 1))


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bootstrap_equal_weight(d: pd.DataFrame, cell_keys=("rho",), n_boot=10000, seed=BOOT_SEED):
    """Equal-weight average over cells of the paired mean difference; resample
    replication IDs jointly (preserving pairing across cells)."""
    piv = d.pivot_table(index="rep", columns=list(cell_keys), values="diff")
    piv = piv.dropna()
    X = piv.values                      # reps x cells
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n_boot, X.shape[0]))
    bm = X[idx].mean(axis=1).mean(axis=1)
    return float(X.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)), X.shape[0]
