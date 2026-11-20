"""Generate every report figure from the study data (no simulation is run here).

Inputs are the analysis outputs written by reproduce_report.py (or the individual
analyze.py scripts) under --data (default results/data). Outputs: PDF and PNG files
in results/figures/, and the report figures' PDFs copied to report/figures/ for LaTeX.

Report figures
  fig1_handoff             worked two-vehicle example (schematic of a fixed example; not a result)
  fig2_commitment          Experiment A: cost of locking crossing orders (tuning and test streams)
  fig3_detours             Experiment C: value of a temporary booking by link quality
  fig4_recovery_deadlines  Experiments D/E: price of safe installation by loss, retries and slot length
Supplementary figures (README)
  figS1_recovery_by_load   Experiment D at both loads
  figS2_deadlines          Experiment E in detail
  figS3_mechanisms         temporary bookings and order locks under recovery

Usage: python experiments/generate_figures.py [--data results/data]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FIGURES, REPORT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ink, teal, rust, gold, grid, grey, plum = "#1f384a", "#13766e", "#ad5823", "#8a6d1d", "#dbe3e7", "#7e939c", "#6b4c8a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 9.5,
                     "axes.labelsize": 9.0, "legend.fontsize": 8.0, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.hashsalt": "fai"})
ARMS = [("k0", "One attempt", ink, "o"), ("k1", "One retry", teal, "s"),
        ("k2", "Two retries", rust, "D"), ("oracle", "Reliable feedback (oracle)", gold, "^")]


def done(f, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    f.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=.04,
              metadata={"CreationDate": None, "ModDate": None})
    f.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight", pad_inches=.04)
    plt.close(f)
    if not name.startswith("figS"):          # supplementary figures are for the README only
        (REPORT / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIGURES / f"{name}.pdf", REPORT / "figures" / f"{name}.pdf")
    print(f"  {name}.pdf / .png")


def title(a, text):
    a.set_title(text, loc="left", weight="bold", color=ink, pad=5)


def errbars(a, x, z, **kw):
    z = np.asarray(z, float)
    a.errorbar(x, z[:, 0], yerr=[z[:, 0] - z[:, 1], z[:, 2] - z[:, 0]], capsize=2.2, ms=4.3, lw=1.3, **kw)


# --------------------------------------------------------------------------- Fig. 1
def handoff(_):
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"wspace": .5})
    panels = [(ax[0], "(a) A direct swap has no safe first step", [(10, 14), (12, 14), (10, 9), (12, 9)],
               ["Issued", "Move A first", "Move B first", "Desired"]),
              (ax[1], "(b) One temporary booking makes it safe", [(10, 14), (10, 15), (12, 15), (12, 9)],
               ["Issued", "Park B at 15", "Move A to 12", "Return B to 9"])]
    for a, ttl, rows, labels in panels:
        title(a, ttl)
        for n, (A, B) in enumerate(rows):
            y = 3 - n
            for x, off, col, car in [(A, .015, teal, "A"), (B, -.33, rust, "B")]:
                a.add_patch(Rectangle((x, y + off), 3, .32, color=col))
                a.text(x + 1.5, y + off + .16, car, ha="center", va="center", color="white", weight="bold", fontsize=8)
            if a is ax[0] and n in (1, 2):
                x = max(A, B)
                a.add_patch(Rectangle((x, y - .36), min(A, B) + 3 - x, .75, fill=False, ec="#912f2f", lw=1.2, ls="--"))
        a.set_xlim(8.6, 18.4)
        a.set_ylim(-.5, 3.56)
        a.set_yticks([3, 2, 1, 0], labels)
        a.set_xticks([9, 12, 15, 18])
        a.set_xlabel("Reserved entry time (s)")
        a.grid(axis="x", color=grid)
        a.set_axisbelow(True)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
    done(f, "fig1_handoff")


# --------------------------------------------------------------------------- Fig. 2
def commitment(data):
    curve = pd.read_csv(data / "commitment" / "tuning_curve.csv")
    con = pd.read_csv(data / "commitment" / "contrasts.csv")
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": .3, "width_ratios": [1, 1.15]})
    a = ax[0]
    hs = ["freeze_0", "freeze_1", "freeze_2", "freeze_4", "freeze_6", "freeze_8", "freeze_inf"]
    xs = np.arange(len(hs))
    cols = {0.3: "#9cc5bf", 0.5: teal, 0.7: "#6598ae", 0.9: ink}
    for rho, col in cols.items():
        g = curve[(curve.sigma == 1.0) & (curve.rho == rho)].set_index("policy").loc[hs]
        errbars(a, xs + (rho - .6) * .25, g[["est", "lo", "hi"]].values, fmt="o-", color=col, label=f"ρ = {rho:g}")
    a.axhline(0, color=grey, lw=.9)
    a.set_xticks(xs, ["0", "1", "2", "4", "6", "8", "∞"])
    a.set_xlabel("Freeze horizon H (s): lock a vehicle's order\nonce it is within H of its estimated readiness")
    a.set_ylabel("Extra delay vs never locking (s/veh)")
    title(a, "(a) Tuning streams, σ = 1")
    a.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4, columnspacing=.9)
    a.grid(axis="y", color=grid)
    a.set_axisbelow(True)

    b = ax[1]
    rhos = sorted(con.rho.unique())
    xs = np.arange(len(rhos))
    for k, (pol, name, col, mk) in enumerate([("fcfs", "First-come first-served", plum, "v"),
                                              ("freeze_inf", "Lock at request (H = ∞)", rust, "s"),
                                              ("freeze_4", "Fixed freeze, H = 4 s", gold, "D"),
                                              ("adaptive", "Adaptive stopping rule", teal, "o")]):
        g = con[con.a == pol].set_index("rho").loc[rhos]
        errbars(b, xs + (k - 1.5) * .1, g[["est", "lo", "hi"]].values, fmt=mk + "-", color=col, label=name)
    b.axhline(0, color=grey, lw=.9)
    b.set_xticks(xs, [f"{r:g}" for r in rhos])
    b.set_xlabel("Load ρ")
    b.set_ylabel("Extra delay vs rolling (s/veh)")
    title(b, "(b) Test streams, σ = 1 (60 per load)")
    b.set_ylim(-.17, 1.02)
    b.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.5, columnspacing=.8, fontsize=7.5)
    b.grid(axis="y", color=grid)
    b.set_axisbelow(True)
    done(f, "fig2_commitment")


# --------------------------------------------------------------------------- Fig. 3
def pair_value(p, h=3., dl=1., s=1.):
    """Expected saving of the one-detour handoff of Fig. 1 with one attempt per message (u = v = p)."""
    return ((s + 2 * dl) * p ** 5 - (h - dl - s) * (p * (1 - p) + p * p * (1 - p))
            - (2 * h - 2 * dl - s) * (p ** 3 * (1 - p) + p ** 4 * (1 - p)))


def detours(data):
    cells = pd.read_csv(data / "sweep" / "cells.csv")
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"wspace": .32, "width_ratios": [1, 1.3]})
    a = ax[0]
    e = np.linspace(0, .3, 301)
    a.plot(100 * e, [pair_value(1 - x) for x in e], color=rust, lw=1.8, label="Detour (park B at 15)")
    a.plot(100 * e, [(1 - x) * 1. for x in e], color=teal, lw=1.8, label="Best direct target")
    a.axhline(0, color=grey, lw=.8)
    a.set_xlabel("Loss per message leg (%)")
    a.set_ylabel("Expected saving (vehicle-s)")
    title(a, "(a) Example of Fig. 1, one attempt")
    a.legend(frameon=False, loc="upper right")
    a.grid(color=grid)
    b = ax[1]
    base = cells[(cells.cell_rho == .75) & (cells.cell_sigma == 1.) & (cells.cell_hf == 3.)].iloc[0]
    links = ["0/1", "0.1/0.99", "0.1/0.95", "0.1/0.9"]
    xs = np.arange(len(links))
    for arm, col, mk, off, lab in [("oblivious", rust, "s", -.15, "Detour chosen for a perfect link"),
                                   ("aware", ink, "o", 0., "Loss-aware choice (direct or detour)"),
                                   ("menu", teal, "^", .15, "Best direct target")]:
        z = [[base[f"{arm}-nominal@{l}{s}"] for s in ("", ":lo", ":hi")] for l in links]
        errbars(b, xs + off, z, fmt=mk, color=col, label=lab)
    b.axhline(0, color=grey, lw=.9)
    b.set_xticks(xs, ["perfect", "1%", "5%", "10%"])
    b.set_xlabel("Loss per leg (100 ms slots on lossy links)")
    b.set_ylabel("Gain vs direct installation\n(ms per vehicle per fork)")
    title(b, "(b) Single decisions, ρ = 0.75, σ = 1")
    b.set_ylim(-17, 46)
    b.legend(frameon=False, loc="upper right", fontsize=7.5)
    b.grid(axis="y", color=grid)
    b.set_axisbelow(True)
    done(f, "fig3_detours")


# --------------------------------------------------------------------------- Figs. 4-6
def recovery_inputs(data):
    con = pd.read_csv(data / "recovery" / "contrasts.csv")
    con = con[con.metric == "mean_delay"]
    der = pd.read_csv(data / "recovery" / "derived.csv")

    def ref(rho, d, q, a, b):
        z = con[(con.family == "R_reference") & (con.rho == rho) & (con.d == d) & (con.q == q) & (con.psi == 0)
                & (con.a == a) & (con.b == b)]
        return z.iloc[0]
    return con, der, ref


def recovery(data):
    con, der, ref = recovery_inputs(data)
    qs = [0., .01, .05, .1, .2]
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"wspace": .08})
    for a, rho in zip(ax, (.5, .75)):
        xs = np.arange(len(qs))
        fcfs = ref(rho, .1, .1, "k0", "atomic_rolling").est - ref(rho, .1, .1, "k0", "atomic_fcfs").est
        a.axhline(fcfs, color=grey, ls="--", lw=1.0, label="Instantaneous FCFS" if rho == .5 else None)
        for k, (arm, name, col, mk) in enumerate(ARMS):
            ys = [ref(rho, .1, q, arm, "atomic_rolling") for q in qs]
            errbars(a, xs + (k - 1.5) * .08, [[z.est, z.lo, z.hi] for z in ys], fmt=mk + "-", color=col, label=name)
        a.set_xticks(xs, ["0", "1", "5", "10", "20"])
        a.set_xlabel("Loss per message leg (%)")
        title(a, f"Load ρ = {rho:g}")
        a.grid(axis="y", color=grid)
        a.set_axisbelow(True)
    ax[0].set_ylabel("Extra delay vs instantaneous\nrolling installation (s/veh)")
    h, l = ax[0].get_legend_handles_labels()
    f.legend(h[1:] + h[:1], l[1:] + l[:1], frameon=False, loc="lower center", bbox_to_anchor=(.5, .96), ncol=5,
             handlelength=1.6, columnspacing=1.1)
    done(f, "figS1_recovery_by_load")


def deadlines(data):
    con, der, ref = recovery_inputs(data)
    ds = [.05, .1, .25, .5, 1.]
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"wspace": .36, "width_ratios": [1.15, 1]})
    a = ax[0]
    xs = np.arange(len(ds))
    for k, (arm, name, col, mk) in enumerate(ARMS):
        ys = []
        for d in ds:
            if d == .1:
                z = ref(.75, .1, .1, arm, "atomic_rolling")
            else:
                z = der[(der.rho == .75) & (der.a == arm) & (der.b == "atomic_rolling") &
                        (der.a_cell == f"{d}/0.1/0.0")].iloc[0]
            ys.append((z.est, z.lo, z.hi))
        errbars(a, xs + (k - 1.5) * .08, ys, fmt=mk + "-", color=col, label=name)
    a.set_xticks(xs, ["50", "100", "250", "500", "1000"])
    a.set_xlabel("One-way slot d (ms); 10% loss, ρ = 0.75")
    a.set_ylabel("Extra delay vs instantaneous\nrolling installation (s/veh)")
    title(a, "(a) Recovery weakens as slots lengthen")
    a.grid(axis="y", color=grid)
    a.set_axisbelow(True)
    a.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.08), ncol=2, fontsize=7.4,
             handlelength=1.5, columnspacing=.9)
    st = pd.read_csv(data / "recovery" / "strata.csv")
    st = st[(st.rho == .75) & (st.q == .1) & (st.psi == 0) & (st.arm == "k2")]
    b = ax[1]
    bottom = np.zeros(len(ds))
    for c, col in zip(range(4), ["#ad5823", "#d39a6e", "#9cc5bf", "#13766e"]):
        sh = np.array([st[(st.d == d) & (st.spare == c)].issued.sum() / st[st.d == d].issued.sum() for d in ds])
        b.bar(xs, 100 * sh, bottom=100 * bottom, color=col, width=.66, label=["none", "1", "2", "3 or more"][c])
        bottom += sh
    b.set_xticks(xs, ["50", "100", "250", "500", "1000"])
    b.set_xlabel("One-way slot d (ms)")
    b.set_ylabel("First attempts (%)")
    b.set_ylim(0, 100)
    title(b, "(b) Rounds still usable for a retry")
    b.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, .5), title="spare rounds", title_fontsize=7.6,
             fontsize=7.6)
    done(f, "figS2_deadlines")


def recovery_deadlines(data):
    """Report Fig. 4: (a) price by loss, (b) price by slot length, (c) spare rounds; rho = 0.75."""
    con, der, ref = recovery_inputs(data)
    f, ax = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"wspace": .4, "width_ratios": [1, 1, .78]})
    a = ax[0]
    qs = [0., .01, .05, .1, .2]
    xs = np.arange(len(qs))
    fcfs = ref(.75, .1, .1, "k0", "atomic_rolling").est - ref(.75, .1, .1, "k0", "atomic_fcfs").est
    a.axhline(fcfs, color=grey, ls="--", lw=1.0, label="Instantaneous FCFS")
    for k, (arm, name, col, mk) in enumerate(ARMS):
        ys = [ref(.75, .1, q, arm, "atomic_rolling") for q in qs]
        errbars(a, xs + (k - 1.5) * .08, [[z.est, z.lo, z.hi] for z in ys], fmt=mk + "-", color=col, label=name)
    a.set_xticks(xs, ["0", "1", "5", "10", "20"])
    a.set_xlabel("Loss per message leg (%)\n(d = 100 ms)")
    a.set_ylabel("Extra delay vs instantaneous\nrolling installation (s/veh)")
    title(a, "(a) Loss and retries")
    b = ax[1]
    ds = [.05, .1, .25, .5, 1.]
    xs = np.arange(len(ds))
    for k, (arm, name, col, mk) in enumerate(ARMS):
        ys = []
        for d in ds:
            z = ref(.75, .1, .1, arm, "atomic_rolling") if d == .1 else \
                der[(der.rho == .75) & (der.a == arm) & (der.b == "atomic_rolling") & (der.a_cell == f"{d}/0.1/0.0")].iloc[0]
            ys.append((z.est, z.lo, z.hi))
        errbars(b, xs + (k - 1.5) * .08, ys, fmt=mk + "-", color=col)
    b.set_xticks(xs, ["50", "100", "250", "500", "1000"])
    b.set_xlabel("One-way slot d (ms)\n(10% loss)")
    title(b, "(b) Slot length")
    for z in (a, b):
        z.set_ylim(0, 2.6)
        z.grid(axis="y", color=grid)
        z.set_axisbelow(True)
    st = pd.read_csv(data / "recovery" / "strata.csv")
    st = st[(st.rho == .75) & (st.q == .1) & (st.psi == 0) & (st.arm == "k2")]
    c = ax[2]
    bottom = np.zeros(len(ds))
    for k, col in zip(range(4), ["#ad5823", "#d39a6e", "#9cc5bf", "#13766e"]):
        sh = np.array([st[(st.d == d) & (st.spare == k)].issued.sum() / st[st.d == d].issued.sum() for d in ds])
        c.bar(xs, 100 * sh, bottom=100 * bottom, color=col, width=.7, label=["0", "1", "2", "3+"][k])
        bottom += sh
    c.set_xticks(xs, ["50", "100", "250", "500", "1000"], rotation=40, fontsize=7.6)
    c.set_xlabel("One-way slot d (ms)\n(two-retry arm)")
    c.set_ylabel("First attempts (%)")
    c.set_ylim(0, 100)
    title(c, "(c) Rounds left to retry")
    c.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, .5), fontsize=7.2, title="spare\nrounds",
             title_fontsize=7.2, handlelength=1.0, labelspacing=.3)
    h, l = a.get_legend_handles_labels()
    f.legend(h[1:] + h[:1], l[1:] + l[:1], frameon=False, loc="lower center", bbox_to_anchor=(.5, .97), ncol=5,
             handlelength=1.6, columnspacing=1.1)
    done(f, "fig4_recovery_deadlines")


def mechanisms(data):
    con, _, _ = recovery_inputs(data)
    rows = [("handoff_k0", "k0", "Temporary bookings, one attempt"), ("handoff_k2", "k2", "Temporary bookings, two retries"),
            ("freeze_k0", "k0", "Lock every order, one attempt"), ("freeze_k1", "k1", "Lock every order, one retry")]
    f, a = plt.subplots(figsize=(7.0, 2.2))
    ys = np.arange(len(rows))[::-1]
    for rho, col, mk, off in ((.5, ink, "o", .12), (.75, teal, "s", -.12)):
        for y, (x_, y_, _) in zip(ys, rows):
            z = con[(con.family == "F6_mechanisms") & (con.rho == rho) & (con.a == x_) & (con.b == y_)].iloc[0]
            a.errorbar(z.est, y + off, xerr=[[z.est - z.lo], [z.hi - z.est]], fmt=mk, color=col, capsize=3, ms=4.5, lw=1.3)
    a.axvline(0, color=grey, lw=.9)
    a.set(yticks=ys, yticklabels=[r[2] for r in rows],
          xlabel="Δ mean delay vs direct installation with the same recovery rule (s/veh); 10% loss, 100 ms")
    a.grid(axis="x", color=grid)
    a.set_axisbelow(True)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.legend(handles=[Line2D([], [], marker=m, color=c_, ls="", label=l) for m, c_, l in
                      [("o", ink, "ρ = 0.5"), ("s", teal, "ρ = 0.75")]], loc="upper left", frameon=False, ncol=2)
    done(f, "figS3_mechanisms")


def main(data: Path) -> None:
    print("Figures from", data)
    for fn in (handoff, commitment, detours, recovery_deadlines, recovery, deadlines, mechanisms):
        fn(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    main(ap.parse_args().data)
