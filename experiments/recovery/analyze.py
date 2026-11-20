"""Experiments D and E analysis (held-out recovery study 5809), as frozen in its design record.

Stop rules (any failure aborts before anything is written): a safety violation, an
undrained run, a lock relaxation, a missing (cell, arm, rep), the zero-budget retry
code (k0_log) differing from the one-attempt closed loop (k0), or the correlated-loss
k0 differing from the independent-loss k0.

Outputs: contrasts.csv (pre-specified paired contrasts with Holm adjustment per family),
primary.json, arm_means.csv, strata.csv (deadline strata), and derived.csv (post-hoc
contrasts between cells, same paired stream bootstrap).

Usage: python experiments/recovery/analyze.py [--data results/data/recovery] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lockorder.evaluation.stats import bootstrap_mean_ci, holm, signflip_p  # noqa: E402

C = config("recovery")
B, SEED, BAND = C["bootstrap_draws"], C["bootstrap_seed"], C["band"]
CELL = ["rho", "d", "q", "psi"]
PROTO = [f"cl_{k}" for k in ("transactions", "deferred", "bookings", "grants", "delivered", "switched", "acks",
                              "pinned", "pinned_s", "detours", "parks", "returns", "stranded", "no_gain",
                              "lock_relaxed", "inserted", "gated_bookings")]


def load(data):
    df = pd.read_csv(data / "test.csv.gz", float_precision="round_trip")
    df["psi"] = df["psi"].fillna(0.0)
    return df


def stop_rules(df, split):
    assert (df.n_violations == 0).all(), df[df.n_violations > 0][["block", "rho", "d", "q", "rep", "arm"]].head()
    assert df.drained.astype(bool).all()
    assert (df.cl_lock_relaxed.fillna(0) == 0).all()
    n = C[split]
    for blk in C["block"]:
        g = df[df.block == blk["name"]]
        cnt = g.groupby(["rho", "d", "q", "arm"]).rep.nunique()
        assert len(cnt) == len(blk["rho"]) * len(blk["d"]) * len(blk["q"]) * len(blk["arms"]) and (cnt == n).all(), blk["name"]
    key = ["rho", "d", "q", "psi", "rep"]
    a = df[df.arm == "k0"].set_index(key).sort_index()
    b = df[df.arm == "k0_log"].set_index(key).sort_index()
    a = a.loc[a.index.isin(b.index)]
    assert len(a) == len(b) and (a.mean_delay.values == b.mean_delay.values).all()
    assert (a[PROTO].fillna(-1).values == b[PROTO].fillna(-1).values).all(), "k0_log differs from the baseline"
    ind = df[(df.arm == "k0") & (df.psi == 0)].set_index(["rho", "d", "q", "rep"])
    bur = df[(df.arm == "k0") & (df.psi > 0)].set_index(["rho", "d", "q", "rep"])
    assert (bur.mean_delay == ind.loc[bur.index].mean_delay).all(), "first attempts must not depend on burst"


def cells(df):
    """One frame per cell; rows of blocks sharing a cell (loss + mech) are joined."""
    out = {}
    for k, g in df.groupby(CELL):
        out[k] = g.drop_duplicates(["arm", "rep"])
    return out


def contrast(g, a, b, metric="mean_delay"):
    w = g.pivot(index="rep", columns="arm", values=metric)
    if a not in w or b not in w:
        return None
    x = (w[a] - w[b]).values
    est, lo, hi = bootstrap_mean_ci(x, B, seed=SEED)
    return dict(est=est, lo=lo, hi=hi, pval=signflip_p(x), reps=int(np.isfinite(x).sum()),
                within_band=bool(-BAND <= lo and hi <= BAND))


def recovered_fraction(g, arm, ref="k0", atomic="atomic_rolling"):
    """(ref - arm) / (ref - atomic): share of the one-attempt price removed; stream bootstrap."""
    w = g.pivot(index="rep", columns="arm", values="mean_delay")[[ref, arm, atomic]].dropna()
    X = w.values
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(X), size=(B, len(X)))
    m = X[idx].mean(axis=1)
    r = (m[:, 0] - m[:, 1]) / (m[:, 0] - m[:, 2])
    pt = X.mean(axis=0)
    return dict(est=float((pt[0] - pt[1]) / (pt[0] - pt[2])), lo=float(np.quantile(r, .025)),
                hi=float(np.quantile(r, .975)), reps=len(X))


FAMILIES = {
    # family: list of (cell filter, a, b)
    "F1_one_retry": [("lossy_d01", "k1", "k0")],
    "F2_second_retry": [("lossy_d01", "k2", "k1")],
    "F3_aware_rule": [("lossy_any", "aware2", "k2")],
    "F4_vs_oracle": [("lossy_any", "k2", "oracle"), ("lossy_any", "k1", "oracle")],
    "F5_slot": [("slot", "k1", "k0"), ("slot", "k2", "k0"), ("slot", "aware2", "k2")],
    "F6_mechanisms": [("mech", "handoff_k0", "k0"), ("mech", "handoff_k2", "k2"),
                      ("mech", "handoff_aware2", "handoff_k2"), ("mech", "freeze_k0", "k0"),
                      ("mech", "freeze_k1", "k1")],
    "F7_burst": [("burst", "k1", "k0"), ("burst", "k2", "k0"), ("burst", "aware2", "k2")],
    "R_reference": [("loss", "k0", "atomic_rolling"), ("loss", "k1", "atomic_rolling"),
                    ("loss", "k2", "atomic_rolling"), ("loss", "oracle", "atomic_rolling"),
                    ("loss", "k0", "atomic_fcfs"), ("loss", "k1", "atomic_fcfs"), ("loss", "k2", "atomic_fcfs")],
}


def selected(kind, k):
    rho, d, q, psi = k
    return {"lossy_d01": d == .1 and q > 0 and psi == 0, "lossy_any": q > 0 and psi == 0,
            "slot": d != .1 and psi == 0, "mech": d == .1 and q == .1 and psi == 0,
            "burst": psi > 0, "loss": d == .1 and psi == 0}[kind]


def main(data, dest, split="test"):
    df = load(data)
    stop_rules(df, split)
    dest.mkdir(parents=True, exist_ok=True)
    cs = cells(df)
    rows = []
    for fam, specs in FAMILIES.items():
        for kind, a, b in specs:
            for k, g in cs.items():
                if not selected(kind, k):
                    continue
                for metric in ("mean_delay", "p95_delay", "throughput"):
                    r = contrast(g, a, b, metric)
                    if r is not None:
                        rows.append(dict(family=fam, rho=k[0], d=k[1], q=k[2], psi=k[3], a=a, b=b, metric=metric, **r))
    con = pd.DataFrame(rows)
    con["p_holm"] = np.nan
    for (fam, metric), g in con.groupby(["family", "metric"]):
        con.loc[g.index, "p_holm"] = holm(g.pval.values)
    con.to_csv(dest / "contrasts.csv", index=False)

    bk = (C["base"]["rho"], C["base"]["d"], C["base"]["q"], 0.0)
    base = cs[bk]
    prim = contrast(base, "k1", "k0")
    out = dict(primary=dict(a="k1", b="k0", cell=dict(rho=bk[0], d=bk[1], q=bk[2]), **prim),
               recovered_k1=recovered_fraction(base, "k1"), recovered_k2=recovered_fraction(base, "k2"),
               recovered_oracle=recovered_fraction(base, "oracle"))
    (dest / "primary.json").write_text(json.dumps(out, indent=2) + "\n")

    metrics = ["mean_delay", "p95_delay", "max_delay", "throughput", "wall_s", "solve_ms_mean"] + PROTO + \
              [c for c in df.columns if c.startswith("rt_")]
    means = df.drop_duplicates(CELL + ["arm", "rep"]).groupby(CELL + ["arm"])[metrics].mean().reset_index()
    means.to_csv(dest / "arm_means.csv", index=False)

    # deadline strata: first attempts, first-attempt failures, later resolutions, by spare rounds
    st = df[df.rule.isin(["blind", "aware", "oracle"])].drop_duplicates(CELL + ["arm", "rep"])
    srows = []
    for key, g in st.groupby(CELL + ["arm"]):
        k, arm = key[:4], key[4]
        for c in range(4):
            n, f, r = (g[f"rt_{x}_s{c}"].sum() for x in ("issued", "fail1", "rec"))
            srows.append(dict(zip(CELL, k), arm=arm, spare=c, issued=n, fail1=f, recovered=r,
                              recovered_share=r / f if f else np.nan))
    pd.DataFrame(srows).to_csv(dest / "strata.csv", index=False)
    derived(df, C["test"]).to_csv(dest / "derived.csv", index=False)

    print(json.dumps(out, indent=1))
    show = con[(con.metric == "mean_delay")].copy()
    show["cell"] = show.apply(lambda r: f"{r.est:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] p_h={r.p_holm:.3f}", axis=1)
    print(show[["family", "rho", "d", "q", "psi", "a", "b", "cell"]].to_string(index=False))


def paired(df, rho, d, q, a, b, reps, psi=0., a_cell=None, b_cell=None, metric="mean_delay"):
    """Arm a in cell (rho, a_cell or (d, q)) minus arm b in cell (rho, b_cell or (d, q)), paired by stream."""
    ad, aq = a_cell or (d, q)
    bd, bq = b_cell or (d, q)
    x = df[(df.rho == rho) & (df.d == ad) & (df.q == aq) & (df.psi == psi) & (df.arm == a)].drop_duplicates("rep")
    y = df[(df.rho == rho) & (df.d == bd) & (df.q == bq) & (df.psi == 0) & (df.arm == b)].drop_duplicates("rep")
    z = (x.set_index("rep")[metric] - y.set_index("rep")[metric]).dropna()
    assert len(z) == reps
    est, lo, hi = bootstrap_mean_ci(z.values, B, seed=SEED)
    return dict(rho=rho, a=a, a_cell=f"{ad}/{aq}/{psi}", b=b, b_cell=f"{bd}/{bq}/0", metric=metric,
                est=est, lo=lo, hi=hi, pval=signflip_p(z.values), reps=len(z))


def derived(df, reps):
    """Post-hoc contrasts not in the frozen family list (archived as test_derived.csv)."""
    rows = []
    for rho in (0.5, 0.75):
        for d in (0.05, 0.25, 0.5, 1.0):
            for arm in ("k0", "k1", "k2", "oracle"):
                rows.append(paired(df, rho, d, .1, arm, "atomic_rolling", reps, b_cell=(.1, .1)))
        rows.append(paired(df, rho, .1, .1, "handoff_k2", "atomic_rolling", reps))
        rows.append(paired(df, rho, .1, .1, "k2", "k0", reps, b_cell=(.1, 0.)))
        rows.append(paired(df, rho, .1, .2, "k2", "k0", reps, b_cell=(.1, 0.)))
        rows.append(paired(df, rho, .1, .1, "k1", "k0", reps, b_cell=(.1, 0.)))
        for arm in ("k0", "k2", "oracle"):
            rows.append(paired(df, rho, 1., .1, arm, "atomic_rolling", reps, b_cell=(.1, .1)))
        rows.append(paired(df, rho, .1, .1, "k2", "k0", reps))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "recovery")
    ap.add_argument("--out", type=Path, default=None, help="analysis output directory (default: --data)")
    a = ap.parse_args()
    main(a.data, a.out or a.data)
