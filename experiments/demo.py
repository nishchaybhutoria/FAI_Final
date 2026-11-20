"""Quick demonstration (well under a minute on a laptop).

1. Simulates one held-out traffic stream (study 5809, rep 0; load 0.75, 10% loss per
   message leg, 100 ms slots) under eight algorithm / installation combinations, prints a
   comparison table, and checks every row against the recorded study data.
2. Draws the intersection reservation timeline of the same stream for instantaneous
   installation and for safe installation with one attempt and with two retries.

Outputs: results/screenshots/demo_output.txt and results/screenshots/reservation_timeline.png
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, read_csv  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from simulate import describe, simulate  # noqa: E402

OUT = ROOT / "results" / "screenshots"
CELL = dict(rho=0.75, loss=0.1, slot=0.1, rep=0)
# (study arm, label, simulate() arguments)
ARMS = [("atomic_rolling", "Rolling, instantaneous install (idealized)", dict(policy="rolling", install="atomic")),
        ("atomic_fcfs", "FCFS, instantaneous install (idealized)", dict(policy="fcfs", install="atomic")),
        ("k0", "Rolling, safe install, one attempt", dict(policy="rolling", install="direct", retries=0)),
        ("k1", "Rolling, safe install, one retry", dict(policy="rolling", install="direct", retries=1)),
        ("k2", "Rolling, safe install, two retries", dict(policy="rolling", install="direct", retries=2)),
        ("oracle", "Rolling, reliable feedback (oracle)", dict(policy="rolling", install="direct", rule="oracle")),
        ("handoff_k2", "Rolling + temporary bookings, two retries", dict(policy="rolling", install="detour", retries=2)),
        ("freeze_k1", "Lock every order, safe install, one retry", dict(policy="freeze_inf", install="direct", retries=1))]


def one(arm):
    _, _, kw = arm
    rec = simulate(**kw, **CELL)
    return describe(rec), rec.vehicles


def timeline(recs, path, t0=420.0, t1=470.0):
    """Lane rows over time: readiness (tick), first reservation (outline), actual crossing (filled)."""
    f, ax = plt.subplots(len(recs), 1, figsize=(7.2, 1.25 * len(recs) + .5), sharex=True)
    for a, (label, v) in zip(ax, recs):
        sel = np.where((v["S"] >= t0 - 3) & (v["S"] <= t1))[0]
        for i in sel:
            y = 1 - int(v["lane"][i])
            col = "#13766e" if y == 1 else "#ad5823"
            a.add_patch(Rectangle((v["first_slot"][i], y - .32), 3, .64, fill=False, ec=col, lw=.9, ls="--"))
            a.add_patch(Rectangle((v["S"][i], y - .22), 3, .44, fc=col, ec="white", lw=.7, alpha=.9))
            a.plot([v["r"][i]] * 2, [y - .42, y + .42], color="#1f384a", lw=1.1)
        a.set_ylim(-.6, 1.6)
        a.set_yticks([1, 0], ["lane 1", "lane 2"])
        a.set_title(label, loc="left", fontsize=8.6, weight="bold", color="#1f384a", pad=3)
        a.grid(axis="x", color="#dbe3e7")
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.tick_params(axis="y", length=0)
    ax[-1].set_xlim(t0, t1)
    ax[-1].set_xlabel("Time (s).  Box = 3 s occupancy of the conflict zone; dashed = first reservation; "
                      "solid = actual crossing; tick = readiness time", fontsize=7.6)
    f.tight_layout()
    f.savefig(path, dpi=170)
    plt.close(f)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, ARMS))
    rec = read_csv(DATA / "recovery" / "test.csv.gz")
    rec = rec[(rec.rho == CELL["rho"]) & (rec.d == CELL["slot"]) & (rec.q == CELL["loss"]) & (rec.rep == CELL["rep"])
              & rec.psi.isna()]
    lines = ["Demo: one held-out stream (study 5809, rep 0), load 0.75, sigma 1, 10% loss per leg, 100 ms slots",
             f"{'algorithm / installation':44s} {'mean delay':>10s} {'p95':>6s} {'pinned':>6s} {'retx':>5s} "
             f"{'unsafe':>6s}  matches recorded run",
             "-" * 98]
    for (arm, label, _), (d, _) in zip(ARMS, res):
        saved = rec[rec.arm == arm].iloc[0]
        same = saved.mean_delay == d["mean_delay"] and saved.p95_delay == d["p95_delay"]
        pin = "--" if d["pinned"] is None else f"{d['pinned']:.0f}"
        rt = "--" if d["pinned"] is None else f"{d['retransmissions'] or 0:.0f}"   # one attempt: none
        lines.append(f"{label:44s} {d['mean_delay']:10.3f} {d['p95_delay']:6.2f} {pin:>6s} {rt:>5s} "
                     f"{d['violations']:6d}  {'yes' if same else 'NO'}")
    lines += ["", "mean delay / p95: cohort delay in s per vehicle (entry time minus readiness time);",
              "pinned: vehicles left with two protected bookings after a lost message; retx: retransmissions;",
              "unsafe: safety-audit violations (clearance, FIFO, physical feasibility, ledger)."]
    text = "\n".join(lines)
    print(text)
    (OUT / "demo_output.txt").write_text(text + "\n")
    pick = {"atomic_rolling": 0, "k0": 2, "k2": 4}
    timeline([(ARMS[k][1], res[k][1]) for k in pick.values()], OUT / "reservation_timeline.png")
    print(f"\nWrote {OUT / 'demo_output.txt'} and {OUT / 'reservation_timeline.png'}")
    assert all("NO" not in ln for ln in lines), "a demo run differs from the recorded study data"
