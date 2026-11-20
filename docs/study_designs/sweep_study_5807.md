> **Archived design record.** Copied unchanged from the research archive (study 5807, commit 74662a0). It was written and committed before the held-out streams of that study were run. File and command paths in it refer to the archive layout; in this repository the corresponding files are `configs/sweep.toml`, `experiments/sweep/run.py`, `experiments/sweep/forks.py` and `experiments/sweep/analyze.py`. Only Phase B (the regime sweep) is used by the final report; Phases A and C are archived studies that this repository does not reproduce.

# Revision design record

Internal design record, committed before the data it governs are generated. It is
not an external registration. The question, arms and contrasts below are fixed by
this commit; later commits may add figures, tables and verifiers, but not change
them. Archived studies (keys 329, 7319, 9431) are unchanged and never pooled with
these data.

## Question

When a better schedule must be installed by acknowledged single-vehicle
substitutions, which improvements can actually be realized, when does one
temporary booking (a detour) recover more than a choice of direct target, and
does order commitment become valuable once revision is costly?

## Phase A (completed, secondary): detours on lossy links, seen states

Commit `b0da91c`. A new estimand on the two archived fork samples (9431 test, 7319
test), whose states were analyzed before. It is a reanalysis, never a held-out
confirmation. Move-graph value: `src/lockorder/replacement/handoff.py`. Arms are
defined in `experiments/revision/forks.py`. Findings carried forward:

- The perfect-link witnesses reproduce the archived recovery (+9.37 ms fresh, +15.62 ms reanalysis).
- When the witnesses are executed without regard to loss, the expected value turns negative near 10% erasure. At d = 0, u = v = 0.9: −1.29 ms vs nominal direct (fresh), −2.07 ms (reanalysis). The realized value at 20% erasure is −10.2 ms [−22.1, −2.2] (fresh, d = 0.1).
- The loss-aware choice cannot lose in expectation. Its realized gain is material only at ≥ 99% delivery.

## Phase B: regime sweep (study 5807; this freeze governs the test split)

- **Cells (12):** ρ ∈ {0.5, 0.75, 0.9} × σ ∈ {0.5, 1.0} × h_f ∈ {3.0, 1.5}. Cross-lane clearance is 3 s throughout. With the default kinematics, T_L = 2.333 s, so h_f = 1.5 yields a 2.333 s same-lane gap. Other settings: default motion; 90 s rolling reference; forks at isolated public signals with unchanged live membership in [20, 80) s, exactly as in the archived fork studies.
- **Streams:** 32 test streams per cell (`--split test`, reps 0–31). The pilot is one stream per cell (`docs/revision_pilot_gate.md`).
- **Links:** (d, u = v) ∈ {(0, 1), (0.1, 0.99), (0.1, 0.95), (0.1, 0.9)}. The primary link is (0.1, 0.9).
- **Arms** (`forks.py`): keep, atomic, nominal, menu, oblivious, aware. All arms see the same public snapshot and the same packet uniforms. A return leg uses keyed stream 8.
- **Unit and aggregation:** saved vehicle-seconds per live vehicle per fork, against the issued slots. Average over forks within a stream, then with equal stream weights. Null forks count.
- **Intervals:** paired percentile 95% intervals from 10,000 stream resamples (seed 30000).

Hypotheses and tests:

- **H3 (primary).** Base cell (ρ = 0.75, σ = 1, h_f = 3), primary link. Contrast: realized `aware − menu`, two-sided. The model-expected contrast is reported alongside it; it is nonnegative by construction.
- **H3 per cell (confirmatory family).** The same realized contrast in each of the 12 cells, using a sign-flip permutation p-value over streams (`stats.signflip_p`) with Holm correction across the 12.
- **H1.** Target choice matters more when same-lane slack exists. For each (ρ, σ), take the per-stream `menu − nominal` (realized, primary link) at h_f = 1.5 minus the same at h_f = 3. Streams pair by rep, since arrivals and signs are common random numbers across h_f. Weight the six (ρ, σ) pairs equally and resample reps jointly (`stats.bootstrap_equal_weight`). H1 is supported if the 95% interval lies above 0.
- **H2 (descriptive only).** Reorder frequency, and the perfect-link `aware − nominal` recovery, increase with σ and ρ. Reported per cell, with no inferential claim.

Reported per cell: counts (forks, improving, reorders, rescuable, detours used, stranded), effects conditional on reorder forks, the percentage of the atomic gain, and the maximum delay.

**Stop rules.** If any safety, drain, DP-exactness (label cap) or equivalence assertion fails, stop and fix. No run may be removed. If a cell cannot run exactly, report it as missing, with the reason.

## Phase C: closed loop (frozen separately, before its test split)

Study 5808. The arms (rolling, tuned freeze, `freeze_inf`, adaptive and handoff, each installed through the lossy direct contract) and the cells and primaries are listed in the revision plan and will be fixed in `docs/revision_closed_loop.md`. That freeze comes after the implementation and the tune-split pilot, and before any test stream.
