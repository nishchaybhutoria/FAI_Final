# Report-to-code traceability

Every number quoted in `report/CS329_Final_Report.pdf` is a LaTeX macro (`\nName`) defined in
`report/numbers.tex`. That file, and the two data tables in `report/tables/`, are written by
`experiments/reproduce_report.py` from the study data in `results/data/`. Nothing is typed by hand.

```bash
make reproduce        # = python experiments/reproduce_report.py --check && python experiments/generate_figures.py
```

`--check` recomputes all 375 macros and both tables from the raw study data (running the three
analyses first) and fails unless every value equals the committed one, character for character.

## Verification status (performed while preparing the submission)

| Check | Command | Result |
|---|---|---|
| Unit, property and regression tests | `make test` | 135 passed |
| Report numbers and tables from recorded data | `make reproduce` | all 375 numbers and 2 tables reproduced exactly |
| Full regeneration of all raw data | `make reproduce-full` | 4,872 commitment runs, 384 sweep streams (563,328 fork-arm rows) and 8,520 recovery runs regenerated **bit-identical** to `results/data/` (all columns except wall-clock timings). Done twice: in the development environment (4 min 23 s on 30 worker processes, 32-thread machine, 6,940 CPU-seconds) and in a fresh clone with the pinned PyPI wheels (4 min 25 s, 6,924 CPU-seconds) |
| Report numbers from regenerated data | `python experiments/reproduce_report.py --data results/rerun --check` | all 375 numbers and 2 tables reproduced exactly |
| Quick verification | `make verify` | 52 commitment runs, 121 forks and 46 recovery runs regenerated exactly, then all report numbers checked; 18 s on 4 workers |
| Clean environment | fresh clone on a native filesystem + `make setup test demo verify reproduce figures report` | all passed with the pinned PyPI wheels (Python 3.12); no tracked file changed afterwards. Quick verification and the demo also pass under the `spawn` start method used on macOS and Windows |

## Principal results

Data files are relative to `results/data/`. "Script" is the analysis that produces the quantity;
all of them run inside `make reproduce`. "Regenerate raw data" is the command that recreates the
underlying runs from scratch (`--mode full` reruns everything; the figures in brackets give the
study's share of the total cost).

### Experiment A — commitment (Section 6, Fig. 2)

| Claim in the report | Macro | Value | Config | Data | Script |
|---|---|---|---|---|---|
| Tuning chose *never freeze* in all cells | `CommitNeverFreeze`, `CommitCells` | 24 of 24 | `configs/commitment.toml` [tune] | `commitment/tune.csv.gz` → `tuning_choice.csv` | `experiments/commitment/analyze.py` |
| FCFS − rolling, equal weight over loads, σ = 1 | `FcfsMinusRolling` (+`Lo`/`Hi`) | 0.31 s [0.23, 0.39] | [test] | `commitment/test.csv.gz` → `primary.json` | same |
| Lock at request − rolling | `FreezeInfMinusRolling` | 0.38 s [0.33, 0.44] | [test] | `primary.json` | same |
| 4 s freeze − rolling | `FreezeFourMinusRolling` | 0.073 s [0.056, 0.093] | [test] | `primary.json` | same |
| Adaptive stopping rule − tuned freeze | `StopMinusTuned` | +8.1×10⁻⁵ s [0.7, 20.5]×10⁻⁵ | [test] | `primary.json` | same |
| Fig. 2a tuning curve; Fig. 2b per-load contrasts | — | — | both | `tuning_curve.csv`, `contrasts.csv` | `experiments/generate_figures.py::commitment` |

Regenerate raw data: `python experiments/run_experiments.py --mode full --study commitment` (about 4,000 CPU-seconds).

### Experiments B and C — single decisions (Section 6, Table 2, Fig. 3)

| Claim | Macro | Value | Config | Data | Script |
|---|---|---|---|---|---|
| Forks, improving forks, reordering forks | `SweepForks`, `SweepImproving`, `SweepReorders` | 23,472; 5,473; 678 | `configs/sweep.toml` | `sweep/test.csv.gz` → `structure.csv` | `experiments/sweep/analyze.py` |
| Perfect-link shortfall occurs exactly at the reordering forks | `SweepShortDirect` | 678 (asserted equal) | same | `structure.csv` | same (assertion) |
| Reorders completable with one temporary booking | `SweepRescued`, `SweepRescuedShare` | 609 (90%) | same | `structure.csv` | same |
| Direct installation loss, perfect link, 12 regimes | `SweepGapPerfectMin/Max`, `SweepGapPerfectShareMax` | 4.6–31.9 ms, ≤ 15% | same | `cells.csv` (`nominal-atomic@0/1`) | same |
| Direct installation loss at 10% loss | `SweepGapNineMin/Max`, `SweepGapNineShareMax` | 15.5–89.9 ms, ≤ 28% | same | `cells.csv` (`nominal-atomic@0.1/0.9`) | same |
| Best menu target over nominal, any regime | `SweepMenuMax` | 0.80 ms | same | `cells.csv` | same |
| Perfect-link detour gain across regimes | `SweepRecMin/Max` | 4.6–29.9 ms | same | `cells.csv` (`oblivious-nominal@0/1`) | same |
| Perfect-link detours at 10% loss: negative in all regimes, CI < 0 in | `SweepObliviousNegEst`, `SweepObliviousNeg` | 12; 7 | same | `cells.csv` | same |
| Loss-aware choice vs direct at 10% loss: non-negative estimate in | `SweepAwareNonNeg` | 12 of 12 regimes | same | `cells.csv` (`aware-nominal@0.1/0.9`) | same |
| Pre-specified primary (loss-aware − best direct, base cell, 10%) | `SweepPrimary` | +5.6 ms [−0.5, +13.0]; min Holm p 0.46 | same | `primary.json` | same |
| Two-vehicle example: detour value at u=v=0.9; crossover | `PairNine`, `PairCross` | 1.18 vehicle-s; 12.4% | analytic (Eq. 3) | — | `experiments/reproduce_report.py::sweep` |
| Table 2 | — | — | same | `cells.csv`, `structure.csv` | `reproduce_report.py::sweep_table` → `report/tables/sweep_table.tex` |
| Fig. 3 | — | — | same | `cells.csv` | `generate_figures.py::detours` |

Regenerate raw data: `python experiments/run_experiments.py --mode full --study sweep` (about 105 CPU-seconds).

### Experiments D and E — sustained traffic with loss and recovery (Section 6, Table 3, Fig. 4)

| Claim | Macro | Value | Config | Data | Script |
|---|---|---|---|---|---|
| One-attempt price, clean 100 ms link (ρ = 0.5, 0.75) | `RtPriceZeroCleanLow/High` | 0.22, 0.34 s | `configs/recovery.toml` | `recovery/test.csv.gz` → `contrasts.csv` (family `R_reference`) | `experiments/recovery/analyze.py` |
| One-attempt price at 10% loss | `RtPriceZeroTenLow/High` | 1.27, 1.66 s | same | `contrasts.csv` | same |
| Pinned vehicles per stream, ρ = 0.75 (K = 0, 1, 2) | `RtPinnedZero/One/Two` | 38.6, 9.2, 1.6 | same | `arm_means.csv` | same |
| **Primary**: one retry − one attempt (ρ 0.75, d 0.1 s, 10%) | `RtPrimary` | −1.13 s [−1.24, −1.02], p < 10⁻⁴ | same | `primary.json` | same |
| Share of one-attempt price recovered (1 retry, 2 retries, oracle) | `RtRecOne`, `RtRecTwo`, `RtRecOracle` | 68% [64, 72], 78%, 80% | same | `primary.json` | same |
| Two-retry price; above clean link by | `RtPriceTwoTenHigh`, `RtTwoTenMinusCleanHigh` | 0.36 s; 0.03 s [0.01, 0.06] | same | `contrasts.csv`, `derived.csv` | same |
| Retransmissions per stream; share of grants | `RtRetxOne/Two`, `RtRetxShareOne/Two` | 50, 62; 19%, 23% | same | `arm_means.csv` | same |
| p95 delay, one attempt → two retries | `RtPFiveZero`, `RtPFiveTwo` | 17.2 → 15.5 s | same | `arm_means.csv` | same |
| 20% loss: two retries vs oracle | `RtPriceTwoTwentyHigh`, `RtPriceOracleTwentyHigh` | 0.58 vs 0.37 s | same | `contrasts.csv` | same |
| Correlated loss: one retry gain | `RtBurstOne` | −0.49 s | same | `contrasts.csv` (family `F7_burst`) | same |
| Time-aware rule vs two blind retries, max | `RtAwareMax` | 0.015 s | same | `contrasts.csv` (family `F3_aware_rule`) | same |
| Temporary bookings, one attempt / two retries vs direct | `RtDetourZeroLow/High`, `RtDetourTwoLow/High` | +0.08/+0.09; −0.10/−0.15 s | same | `contrasts.csv` (family `F6_mechanisms`) | same |
| Lock every order, one attempt / one retry | `RtFreezeZeroLow/High`, `RtFreezeOneLow/High` | +0.07/+0.06; +0.08/+0.09 s | same | `contrasts.csv` | same |
| Share of first attempts with ≥ 3 spare rounds (100 ms, 0.5 s, 1 s); none at 1 s | `RtSpareThreeTenth/Half/Second`, `RtSpareZeroSecond` | 99%, 84%, 44%; 16% | same | `strata.csv` | same |
| Failures resolved with ample slack (1, 2 retries) vs model | `RtRecAmpleOne/Two`, `RtModelRecOne/Two` | 82%, 97% vs 81%, 96.4% | same | `strata.csv`; analytic | same |
| One-retry benefit at 50 ms and 1 s slots | `RtSlotFiftyOneHigh`, `RtSlotSecondOneHigh` | −1.13, −0.59 s | same | `contrasts.csv` (family `F5_slot`) | same |
| Price at 1 s slots, two retries vs one attempt | `RtSlotSecondPriceTwoHigh`, `RtSlotSecondPriceZeroHigh` | 1.30 vs 1.94 s | same | `derived.csv` | same |
| Table 3 | — | — | same | `arm_means.csv`, `contrasts.csv`, `derived.csv` | `reproduce_report.py::recovery_table` → `report/tables/recovery_table.tex` |
| Fig. 4 | — | — | same | `contrasts.csv`, `derived.csv`, `strata.csv` | `generate_figures.py::recovery_deadlines` |

Regenerate raw data: `python experiments/run_experiments.py --mode full --study recovery` (about 5,300 CPU-seconds).

### Design counts

| Claim | Macro | Value | Source |
|---|---|---|---|
| Commitment runs (tuning + test) | `CommitTuneRuns`, `CommitTestRuns` | 3,072 + 1,800 | row counts of `commitment/*.csv.gz` |
| Recovery runs | `RtRuns` | 8,520 | row count of `recovery/test.csv.gz` |
| Safety audit: zero violations, every run drained, no approximate solve | — | — | stop rules in all three analyses and in `reproduce_report.py` |

### Figure 1

`fig1_handoff` draws the fixed two-vehicle example of the introduction (bookings 10 s / 14 s, target
9 s / 12 s, temporary booking 15 s). It is an illustration, not an experimental result; it is drawn by
`generate_figures.py::handoff` from those constants.

## Statistical conventions

Paired by stream (common random numbers); 95% percentile intervals from 10,000 stream resamples
(seed 30000); sign-flip p-values; Holm correction within pre-specified families. Bootstrap seeds and
counts are in each study's config file.
