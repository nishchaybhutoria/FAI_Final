"""Experiment A analysis: tuning choice, tuning curve, held-out contrasts against rolling.

Statistics: paired by replication (common random numbers). Intervals are percentile
intervals of 10,000 replication resamples (seed 30000); p-values are paired sign-flip
tests, Holm-adjusted within each compared policy across loads.

Usage: python experiments/commitment/analyze.py [--data results/data/commitment] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, config, read_csv, write_json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lockorder.evaluation.stats import bootstrap_mean_ci, holm, signflip_p  # noqa: E402

C = config("commitment")
B, SEED = C["bootstrap_draws"], C["bootstrap_seed"]
FREEZE = C["tune"]["policies"]            # order matters: ties go to the earlier (more flexible) entry
COMPARED = ["fcfs", "freeze_inf", "freeze_4", "adaptive"]


def stop_rules(df: pd.DataFrame, split: str) -> None:
    s = C[split]
    assert (df.error.fillna("") == "").all(), f"{split}: failed runs"
    assert (df.n_violations == 0).all() and df.drained.astype(bool).all(), f"{split}: unsafe or undrained run"
    assert not df.approx.astype(bool).any(), f"{split}: approximate solve"
    n = df.groupby(["rho", "sigma", "policy"]).rep.nunique()
    assert len(n) == len(s["rho"]) * len(s["sigma"]) * len(s["policies"]) and (n == s["reps"]).all(), \
        f"{split}: incomplete design"


def ci(x) -> dict:
    x = np.asarray(x, float)
    est, lo, hi = bootstrap_mean_ci(x, B, seed=SEED)
    return dict(est=est, lo=lo, hi=hi, p=signflip_p(x), reps=len(x))


def tuning(tune: pd.DataFrame):
    means = tune.groupby(["rho", "sigma", "policy"]).mean_delay.mean().unstack()
    choice = []
    for (rho, sigma), r in means.iterrows():
        vals = sorted((r[p], k, p) for k, p in enumerate(FREEZE))
        choice.append(dict(rho=rho, sigma=sigma, choice=vals[0][2], tune_mean=vals[0][0], runner_up=vals[1][2],
                           margin=vals[1][0] - vals[0][0]))
    curve = []
    w = tune.pivot_table(index=["rho", "sigma", "rep"], columns="policy", values="mean_delay")
    for (rho, sigma), g in w.groupby(level=["rho", "sigma"]):
        for p in FREEZE[1:]:
            curve.append(dict(rho=rho, sigma=sigma, policy=p, **ci(g[p] - g["rolling"])))
    return pd.DataFrame(choice), pd.DataFrame(curve)


def held_out(test: pd.DataFrame, choice: pd.DataFrame):
    w = test.pivot_table(index=["rho", "sigma", "rep"], columns="policy", values="mean_delay")
    tuned = choice.set_index(["rho", "sigma"]).choice
    missing = set(tuned.loc[w.index.droplevel("rep").unique()]) - set(w.columns)
    assert not missing, f"tuned choice {missing} was not run on the test split"
    w["tuned_freeze"] = [w.loc[i, tuned.loc[i[:2]]] for i in w.index]
    rows = []
    for (rho, sigma), g in w.groupby(level=["rho", "sigma"]):
        for p in COMPARED + ["tuned_freeze"]:
            rows.append(dict(rho=rho, sigma=sigma, a=p, b="rolling", **ci(g[p] - g["rolling"])))
    con = pd.DataFrame(rows)
    con["p_holm"] = np.nan
    for _, g in con.groupby("a"):
        con.loc[g.index, "p_holm"] = holm(g.p.values)
    # primary contrasts: equal weight over loads (per replication, mean over the loads, then resample)
    prim = {}
    for a, b in (("adaptive", "tuned_freeze"), ("fcfs", "rolling"), ("freeze_4", "rolling"),
                 ("freeze_inf", "rolling")):
        prim[f"{a}-{b}"] = ci((w[a] - w[b]).groupby(level="rep").mean())
    return con, prim


def arm_means(test: pd.DataFrame) -> pd.DataFrame:
    cols = ["mean_delay", "p95_delay", "max_delay", "throughput", "inv_per_veh", "tchg_per_veh", "locks"]
    return test.groupby(["rho", "sigma", "policy"])[cols].mean().reset_index()


def main(data: Path, out: Path) -> dict:
    # parser of the archived analysis (pandas default), so outputs match it bit for bit
    tune, test = read_csv(data / "tune.csv.gz", exact=False), read_csv(data / "test.csv.gz", exact=False)
    stop_rules(tune, "tune")
    stop_rules(test, "test")
    choice, curve = tuning(tune)
    con, prim = held_out(test, choice)
    out.mkdir(parents=True, exist_ok=True)
    choice.to_csv(out / "tuning_choice.csv", index=False)
    curve.to_csv(out / "tuning_curve.csv", index=False)
    con.to_csv(out / "contrasts.csv", index=False)
    arm_means(test).to_csv(out / "arm_means.csv", index=False)
    summary = dict(primary=prim, cells_tuned=len(choice), cells_choosing_rolling=int((choice.choice == "rolling").sum()),
                   runs=dict(tune=len(tune), test=len(test)))
    write_json(out / "primary.json", summary)
    print(f"[commitment] tuning chose 'never freeze' in {summary['cells_choosing_rolling']} of {len(choice)} cells")
    for k, v in prim.items():
        print(f"[commitment] {k}: {v['est']:+.5f} s/veh [{v['lo']:+.5f}, {v['hi']:+.5f}]")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "commitment")
    ap.add_argument("--out", type=Path, default=None, help="analysis output directory (default: --data)")
    a = ap.parse_args()
    main(a.data, a.out or a.data)
