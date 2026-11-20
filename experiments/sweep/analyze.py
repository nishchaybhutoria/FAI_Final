"""Experiments B and C analysis (regime sweep, study 5807).

Per cell (never pooled): nominal saving of atomic installation, the shortfall of
release-gated direct installation, and what a temporary booking (detour) or a
loss-aware choice adds, on each link. Quantities are per live vehicle per fork,
averaged over forks within a stream and then equally over streams; intervals are
percentile intervals of 10,000 stream resamples (seed 30000).

The pre-registered primary and Holm family (loss-aware choice minus best direct target
at 10% loss) and the same-lane-slack hypothesis H1 are computed exactly as frozen
before the held-out streams were opened.

Usage: python experiments/sweep/analyze.py [--data results/data/sweep] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, config, read_csv, write_json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lockorder.evaluation.stats import bootstrap_equal_weight, bootstrap_mean_ci, holm, signflip_p  # noqa: E402

C = config("sweep")
CELL = ["cell_rho", "cell_sigma", "cell_hf"]
B, SEED = C["bootstrap_draws"], C["bootstrap_seed"]
EPS = 1e-9


def ms(x):
    return 1000 * x


def per_stream(g, a, b, metric="saved"):
    wide = g.pivot_table(index=["rep", "epoch"], columns="arm", values=metric)
    return (wide[a] - wide[b]).groupby(level="rep").mean()


def ci(x):
    est, lo, hi = bootstrap_mean_ci(np.asarray(x), B, seed=SEED)
    return dict(est=ms(est), lo=ms(lo), hi=ms(hi))


def link(df, d, p):
    return df[(df.d == d) & (df.u == p)]


def structure(df) -> pd.DataFrame:
    """Which forks fall short of the atomic saving on a perfect link, and whether one detour rescues them."""
    p = link(df, 0.0, 1.0)
    idx = CELL + ["rep", "epoch"]
    w = p.pivot_table(index=idx, columns="arm", values="saved")
    meta = p[p.arm == "keep"].set_index(idx)[["reorder", "rescuable", "ideal"]].loc[w.index]
    short = (w["atomic"] - w["nominal"]) > EPS
    still = (w["atomic"] - w["oblivious"]) > EPS
    rows = []
    for k, g in meta.groupby(level=CELL):
        s, st = short.loc[g.index], still.loc[g.index]
        rows.append(dict(zip(CELL, k), forks=len(g), improving=int((g.ideal > EPS).sum()),
                         reorders=int(g.reorder.sum()), short_direct=int(s.sum()),
                         short_equals_reorder=bool((s == g.reorder.astype(bool)).all()),
                         rescuable=int((g.reorder & g.rescuable).sum()), short_after_detour=int(st.sum())))
    return pd.DataFrame(rows)


def main(data: Path, out: Path) -> dict:
    df = read_csv(data / "test.csv.gz", exact=False)  # parser of the frozen analysis
    assert df.groupby(CELL).rep.nunique().eq(C["test"]).all(), "incomplete design"
    d0, p0 = C["primary_link"]
    prim = link(df, d0, p0)
    base = C["base"]
    res = {}

    g = prim[(prim.cell_rho == base["rho"]) & (prim.cell_sigma == base["sigma"]) & (prim.cell_hf == base["hf"])]
    res["H3_primary"] = dict(realized=ci(per_stream(g, "aware", "menu")),
                             expected=ci(per_stream(g, "aware", "menu", "expected")),
                             forks=int(g[["rep", "epoch"]].drop_duplicates().shape[0]), streams=int(g.rep.nunique()))
    fam = []
    for k, g in prim.groupby(CELL):
        x = per_stream(g, "aware", "menu")
        fam.append(dict(zip(CELL, k), **ci(x), p=signflip_p(x.values)))
    fam = pd.DataFrame(fam)
    fam["p_holm"] = holm(fam.p.values)
    res["H3_family"] = fam.to_dict(orient="records")

    rows = []
    for (rho, sigma), g in prim.groupby(["cell_rho", "cell_sigma"]):
        a = per_stream(g[g.cell_hf == 1.5], "menu", "nominal")
        b = per_stream(g[g.cell_hf == 3.0], "menu", "nominal")
        for rep in a.index.intersection(b.index):
            rows.append(dict(rep=rep, rho=rho, sigma=sigma, diff=a[rep] - b[rep]))
    est, lo, hi, n = bootstrap_equal_weight(pd.DataFrame(rows), cell_keys=("rho", "sigma"), n_boot=B, seed=SEED)
    res["H1"] = dict(est=ms(est), lo=ms(lo), hi=ms(hi), streams=n, supported=bool(lo > 0))

    cells = []
    for k, g in df.groupby(CELL):
        forks = g[(g.arm == "keep") & (g.d == 0)]
        row = dict(zip(CELL, k), forks=len(forks), improving=int((forks.ideal > EPS).sum()),
                   reorders=int(forks.reorder.sum()), rescuable=int(forks.rescuable.sum()),
                   atomic_gain=ci(forks.groupby("rep").ideal.mean())["est"])
        for d, p in C["links"]:
            z = link(g, d, p)
            tag = f"{d:g}/{p:g}"
            for a, b in (("aware", "menu"), ("menu", "nominal"), ("aware", "nominal"), ("oblivious", "nominal"),
                         ("nominal", "atomic")):
                c = ci(per_stream(z, a, b))
                row[f"{a}-{b}@{tag}"] = c["est"]
                row[f"{a}-{b}@{tag}:lo"] = c["lo"]
                row[f"{a}-{b}@{tag}:hi"] = c["hi"]
            for arm in ("aware", "oblivious"):
                y = z[z.arm == arm]
                row[f"{arm}_detours@{tag}"] = int(y.temp_shift.notna().sum())
                row[f"{arm}_stranded@{tag}"] = int(y.stranded.sum())
        cells.append(row)
    cells = pd.DataFrame(cells)
    st = structure(df)
    assert st.short_equals_reorder.all(), "a perfect-link shortfall occurred outside a reordering fork"

    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.csv", index=False)
    st.to_csv(out / "structure.csv", index=False)
    write_json(out / "primary.json", res)
    print(f"[sweep] {int(st.forks.sum())} forks; perfect-link shortfall in exactly the {int(st.reorders.sum())} "
          f"reordering forks; one detour rescues {int(st.rescuable.sum())}")
    p = res["H3_primary"]["realized"]
    print(f"[sweep] primary (loss-aware minus best direct, base cell, 10% loss): {p['est']:+.1f} ms "
          f"[{p['lo']:+.1f}, {p['hi']:+.1f}]")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "sweep")
    ap.add_argument("--out", type=Path, default=None, help="analysis output directory (default: --data)")
    a = ap.parse_args()
    main(a.data, a.out or a.data)
