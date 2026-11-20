"""Golden regression for the one-attempt closed loop (``closed_loop.ClosedLoop``).

The payload was generated from the code at commit 08bc331, *before* any
retransmission work. The one-attempt protocol is the unchanged baseline of the
retransmission study, so it must keep reproducing these outputs exactly (floats
compared bit for bit after a JSON round trip, entry times by SHA-256 of their
``repr``). Regenerate only deliberately:

    PYTHONPATH=src python tests/test_closed_loop_golden.py --write
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from lockorder.evaluation.runner import make_policy
from lockorder.replacement import Channel
from lockorder.replacement.closed_loop import ClosedLoop
from lockorder.simulation.scenario import SimConfig, make_scenario

GOLDEN = Path(__file__).parent / "data" / "closed_loop_golden.json"
CFG = SimConfig(rho=0.75, sigma=1.0, t_arr_end=240.0, warmup=40.0, win_end=200.0, audit=True)
ARMS = [("rolling", "direct"), ("rolling", "handoff"), ("freeze_inf", "direct"), ("adaptive", "direct")]
LINKS = [(0.1, 1.0), (0.1, 0.9), (0.1, 0.8), (0.25, 0.9)]
KEYS = ("mean_delay", "p95_delay", "max_delay", "throughput", "locks", "drained")


def entries_hash(S):
    return hashlib.sha256(repr([float(x) for x in S]).encode()).hexdigest()


def payload():
    out = {}
    for rep in (0, 1):
        sc = make_scenario(CFG, "pilot", rep, study=1111)
        for pol, mode in ARMS:
            for d, p in LINKS:
                r = ClosedLoop(sc, make_policy(pol, CFG), Channel(d, p, p), mode).run()
                s = r.summary
                out[f"{rep}|{pol}|{mode}|{d}|{p}"] = dict(
                    {k: s[k] for k in KEYS}, **{k: v for k, v in s.items() if k.startswith("cl_") and k != "cl_mode"},
                    S=entries_hash(r.vehicles["S"]), violations=len(r.violations))
    return json.loads(json.dumps(out))


def test_one_attempt_closed_loop_matches_golden():
    want = json.loads(GOLDEN.read_text())
    got = payload()
    assert got.keys() == want.keys()
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    assert not bad, list(bad)[:3]


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN.write_text(json.dumps(payload(), indent=1, sort_keys=True) + "\n")
    print("wrote", GOLDEN)
