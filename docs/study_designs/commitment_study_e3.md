> **Archived design record.** Copied unchanged from the research archive (study E3 / study key 329, research-archive commit e334acc). It was written and committed before the held-out streams of that study were run. File and command paths in it refer to the archive layout; in this repository the corresponding files are `configs/commitment.toml` (the subset used by the report) and `experiments/commitment/`. The original design file `configs/studies.toml` is reproduced verbatim below.

```toml
# Pre-registered simulation design (fixed before any test-split run).
# Seed namespaces (scenario.py): pilot 0-9, tune 1000+, test 10000+, MC 20000+, bootstrap 30000.
# Window: warm-up 300 s, cohort = requests in [300, 900) s, arrivals until 1020 s, then drain.

[budget]
tune_reps = 16
test_reps = 60          # pilot SD of paired contrasts up to ~1 s at rho >= 0.9 -> 60 reps (plan: blocks of 10 up to 60)
bootstrap = 10000
equivalence_band = 0.1  # s/vehicle

[e3]   # main study, default physics (L=8 m, V=6 m/s, a=b=3 m/s^2, h=hf=3 s)
blocks = [
  { conditions = ["default"], rhos = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95], sigmas = [0.0, 0.25, 0.5, 1.0], extra = false },
]

[e4]   # causal contrasts and ablations A1-A5 at focus cells
blocks = [
  { conditions = ["default"], rhos = [0.5, 0.9], sigmas = [0.5, 1.0], extra = true },
  { conditions = ["zero_memory", "b1", "b6", "a2", "a6", "no_reveal", "late_reveal"], rhos = [0.5, 0.9], sigmas = [0.5, 1.0], extra = true },
]

[e6]   # robustness: nonuniform clearances; information/motion overlap stress regime (ruling R4)
blocks = [
  { conditions = ["hf2"], rhos = [0.5, 0.7, 0.9], sigmas = [0.5, 1.0], extra = false },
  { conditions = ["overlap", "overlap_zm", "overlap_b1"], rhos = [0.5, 0.7], sigmas = [0.5, 1.0], extra = true },
]
```

## Methodology notes of the original study


## Executed model P (simulator)

* **Geometry and traffic.** Two FIFO approaches share one conflict resource. Requests arrive as
  independent Poisson streams, `λ_ℓ = ρ/(2h)` per lane (offered load `ρ = λ h`, `h = 3 s`). Each lane
  has an upstream holding buffer and a controlled throat of length `L`. Only the lane head is on the
  throat; when it enters the conflict zone, its successor launches from rest at the start of the throat.
  The point-queue ablation (A1, `zero_memory`) removes motion entirely.
* **Readiness marks.** Each request carries a not-before mark
  `r_i = a_i + 8.25 s + σ Σ_{k=1}^{4} Z_ik` (Rademacher signs revealed at request ages 1, 2, 3, 4 s).
  The manager sees only revealed signs. Reservations before resolution must satisfy `s ≥ U_i`, the
  known upper support, so every communicated slot is executable for every realisation.
* **Kinematics.** Default `L = 8 m`, `V = 6 m/s`, `a = b = 3 m/s²` (launch `T₋(L,0) = 7/3 s < h`).
  Each head follows the mixture certificate to its current slot; the profile is replaced only when the
  slot changes. Windows `[t + T₋, t + T₊]` are imposed in every snapshot.
* **Clearances.** All-conflict `h = h_f = 3 s` by default; `hf2` uses `h_f = 2 s` same-lane following.
* **Events.** Entries, births, reveals and (for policies that use them) 0.5 s ticks. In one batch:
  entries, then births and reveals, then one decision. At information/membership events every
  policy re-solves the snapshot with the exact label DP (cap 32 labels per key; the cap never bound in
  any reported run, see `safety.csv`). On pure ticks the incumbent is retained and policies may add
  locks to it; for an exact solver this is equivalent to re-solving with incumbent tie-breaking, since
  the feasible set only shrinks and still contains the incumbent.
* **Objective and window.** Snapshot objective: sum of reserved starts (plus `η × inversions` for the
  penalty family and a `10⁻⁶ s` tie-break toward the incumbent order for all). Evaluation:
  `D_i = S_i − r_i` (actual entry minus true mark) for the cohort of requests in `[300, 900) s`. Arrivals
  continue to 1020 s, then the system drains. Throughput counts entries in `[300, 900) s`. No revision or
  message cost enters the evaluation objective.
* **Safety.** Every entry is checked against its mark, both lanes' clearances, lane FIFO, all locks and
  the launch time. Profiles are checked at construction. In audit runs every plan is also checked by an
  independent pairwise validator.

## Policies

| Policy | Definition |
|---|---|
| FCFS | Lock each request behind every live request of the other lane at birth; times revisable. |
| rolling | Exact snapshot re-optimisation, no voluntary cross-lane locks. |
| freeze `H` | Lock a vehicle's witnessed position relative to all other-lane vehicles once `max(0, r̂ − t) ≤ H`; `H ∈ {0,1,2,4,6,8,∞}` s. |
| penalty `η` | Snapshot cost `Σ s + η · #cross-lane inversions vs the previous plan`, `η ∈ {0.25, 1, 4}` s. |
| slack | FCFS order plus floor `r̂_i(a_i) + β`, `β` = generalised quantile of `σ S_4` at ratio `c_b/(c_b+c_d)`, ratios 1, 3, 19 (β = 0, 2σ, 4σ). |
| adaptive | Pair stopping rule on unlocked adjacent cross-lane witness pairs: gap `m = r̂_y − r̂_x`, epochs = public future reveals before the pair deadline, exact sign kernel, hinge cost plus `κ = 2g`; lock only advice that agrees with the witness. Advice to swap is recorded as deferred. |
| adaptive-Q | As adaptive, but the pair cost is the expected total pair delay with the resource first free at `T₀` (the witness predecessor's start plus clearance). |
| adaptive (whole) | A5: on a commit signal, freeze both members against all other-lane vehicles. |
| adaptive σ×0.5 / ×2 | A4: the decision lookup uses a misspecified σ; data, guards and safety unchanged. |
| tuned_* | Per cell, the family member with the lowest mean delay on the 16 tuning replications (ties: earlier in the candidate list). The choice is fixed before the test split and applied to test rows. |
| offline lower bound | Knows the cohort's true marks; removes other traffic, locks, launch and lane order; one server with the smallest separation; earliest-release-first is optimal for the sum of starts. |

## Statistics

Common random numbers: all policies of one `(condition, ρ, σ, split, rep)` share the scenario, and the
sign/arrival streams are keyed so that `ρ`, `σ` and conditions also share draws. Splits: pilot 0–9,
tuning 1000–1015, test 10000–10059. Paired per-replication differences are summarised with 10,000
bootstrap resamples of replication IDs (percentile 95% CI) and a two-sided sign-flip test. The
confirmatory family (E3 adaptive − tuned freeze, 24 cells) is Holm-adjusted. The pre-specified primary
contrast is adaptive − tuned freeze at `σ = 1`, equal weight over six loads, with replication IDs
resampled jointly across cells. "Practically equivalent" means the 95% CI lies inside ±0.1 s/vehicle.

## Rulings (deviations from the proposal or plan, with reasons)

1. **Guaranteed reservations (`s ≥ U`) and bounded signs** instead of Gaussian hard releases (plan C).
   Gaussian increments cannot support a finite guaranteed reservation. Model G keeps the Gaussian law
   for the stopping study.
2. **Ticks retain the incumbent** (above): an equivalence for the exact solver that makes the full
   design affordable. *Cost if wrong:* none for exact solves; the label cap never bound.
3. **Queue-aware adaptive variant (adaptive-Q) added** after pilot seeds (0–9, never used for tests)
   showed that the plan's pair hinge recommends swaps for queued pairs whose order does not matter.
   It is reported as a secondary method; the plan's rule stays primary. *Cost if wrong:* an extra
   comparator; the primary contrast is unchanged.
4. **Overlap stress regime added** (`L = 60 m`, `V = 12 m/s`, lead 12.25 s, reveals at ages 4–7 s),
   with twins `overlap_zm` (no motion) and `overlap_b1` (weak brakes). The pilot showed that in the
   default regime information is resolved before the 2.3 s approach, so irreversibility is rarely
   exercised. Restricted to `ρ ≤ 0.7` because single-head staging with a 7 s launch caps throughput
   near 2/7 veh/s. Exploratory; not part of the confirmatory family.
5. **Slack ratios 1:1, 3:1, 19:1** instead of 1:1, 3:1, 9:1: for the four-sign law, 3:1 and 9:1 give
   the same quantile (2σ); 19:1 gives 4σ.
6. **Transfer check uses the rolling continuation for "wait"** and exact enumeration of all 256 sign
   patterns per state, instead of the full optimal observation tree. It compares root actions only.
7. **Deferred (P1/P2):** constrained position shifting, event-triggered replanning, online σ
   estimation and bandit tuning (σ is identified after a single reveal in the sign model, so this would
   test nothing), four approaches, message loss.
8. **E4 sensitivities** `b ∈ {1, 6}` at `a = 3` and `a ∈ {2, 6}` at `b = 3`, plus `b = 3, a = 3` from the
   default (plan F). The proposal's `|u| = ∞` is represented by the point-queue ablation, not by large
   finite bounds.
