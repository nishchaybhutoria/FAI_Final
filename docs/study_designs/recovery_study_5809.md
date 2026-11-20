> **Archived design record.** Copied unchanged from the research archive (study 5809, commit f4dd91f). It was written and committed before the held-out streams of that study were run. File and command paths in it refer to the archive layout; in this repository the corresponding files are `configs/recovery.toml`, `experiments/recovery/run.py` and `experiments/recovery/analyze.py`.

# Retransmission study design (frozen before opening study 5809 test)

**The question.** Is the large price of safe reservation updates under message loss an unavoidable coordination cost, or a consequence of the one-attempt protocol? When retransmissions improve reliability but use up the time in which a vehicle can still execute the revision, how should the manager recover?

**What this record fixes.** This record, `configs/retransmission.toml`, `experiments/retransmission/run.py` and `experiments/retransmission/analyze.py` were committed together, after the pilot (`docs/retransmission_decision.md`) and before any test stream. Later commits may add figures, tables and verifiers only.

## Protocol semantics (`src/lockorder/replacement/retry.py`)

**Rounds.** Rounds of 2d start at t0 + 2rd. A message sent in round r reaches the vehicle, if not erased, exactly at τ_r = t0 + (2r+1)d, which is that round's switch instant. A reply sent at τ_r arrives within d.

**Deadlines, kept distinct.**

| Deadline | Definition |
|---|---|
| Latest send | τ − d |
| Latest reception | τ (an attempt received later is void) |
| Latest switch | The last τ_r < min(a, b) at which the switch a → b is physically certified from every state the vehicle could be in |
| Latest ACK | Before the next round's send, or dependants slip a round |

**Messages.** Every message carries the update id `gid` and its base version.
- `GRANT(gid, i, base, a → b)` is retransmitted with the same identity and transition; only τ changes.
- `QUERY(gid)`
- `ACK(gid, executed round)` certifies that the vehicle holds b at version base + 1.
- `STATUS-VOID(gid)` certifies that the vehicle holds a at version `base` and will never execute gid.

**Vehicle rules.**
- It executes gid once, at the attempt's τ, only if all of the following hold: its version equals base, it holds a, gid is not void, the attempt is timely, and it has not entered.
- It re-acknowledges an executed gid whenever it receives that gid again, as a duplicate grant or as a query.
- On a query, a vehicle still at (base, a) voids gid and reports so.
- Anything else is stale and ignored.

**Manager rules.**
- **Releases.** Only a matching ACK retires a; only a matching STATUS-VOID retires b, and it can follow only a query sent after the last grant attempt; observed entry removes the vehicle. Nothing is released on silence or elapsed time.
- **Physical certification.** A grant attempt is sent only if its switch is physically certified at its own τ_r. The certificate is recomputed for every attempt, using the actual earlier legs as reported by ACKs.
- **No grant after a query.** Once gid has been queried, no further grant attempt for gid is sent.
- **Budget exhaustion.** The vehicle stays pinned exactly as in the one-attempt protocol.

**Transaction length.** A transaction ends after the first round in which nothing more can be sent, and never before the one-attempt end time t0 + 2(D+1)d, where D is the maximum depth. With a zero budget this reproduces `closed_loop.ClosedLoop` event for event.

**Retry rules.**
- **blind:** retransmit while certifiable and budget remains.
- **aware:** retransmit, or first-issue a move whose prerequisites confirmed late, only if its own change plus the payoff it still unlocks is positive. That payoff is computed with each pending descendant re-certified at the round it would now occur, one attempt each (§R.4). Otherwise query.
- **oracle:** one attempt, with reliable and informative feedback. It is a comparator for the cost of ambiguity, not a protocol.

**Unchanged from study 5808.** Initial bookings are reliable, transactions are serialized, and the target is the rolling DP optimum. The scheduler, physics and policies are the same.

## Arms

All arms of one (ρ, rep) share the exogenous scenario. First transmissions of every update use the archived per-vehicle packet stream (`SeedSequence[study, split, rep, 9, vehicle]`); retransmissions and queries use stream 10.

| Arm | Implementation |
|---|---|
| `atomic_rolling`, `atomic_fcfs` | unmodified engine (references) |
| `k0` | unchanged one-attempt closed loop (`ClosedLoop`) |
| `k0_log` | `RetryLoop` with budget 0; must equal `k0` exactly |
| `k1`, `k2` | blind, budget 1 or 2 |
| `aware2` | aware, budget 2 |
| `oracle` | reliable informative feedback, one attempt |
| `handoff_k0`, `handoff_k2`, `handoff_aware2` | one-detour handoffs with the same rules |
| `freeze_k0`, `freeze_k1` | full order commitment (`freeze_inf`), one attempt or one retry |

## Cells (study 5809; 60 test streams per cell; E3 full-stream horizon; σ = 1; u = v = 1 − q)

| Block | ρ | d (s) | q | Arms |
|---|---|---|---|---|
| loss | 0.5, 0.75 | 0.1 | 0, 0.01, 0.05, 0.1, 0.2 | atomic ×2, k0, k0_log, k1, k2, aware2, oracle |
| slot | 0.5, 0.75 | 0.05, 0.25, 0.5\*, 1.0\* | 0.1 | k0, k0_log, k1, k2, aware2, oracle |
| mech | 0.5, 0.75 | 0.1 | 0.1 | handoff ×3, freeze ×2 |
| burst | 0.75 | 0.1 | 0.1, ψ = 0.5 | k0, k1, k2, aware2 |

\* The 0.5 s and 1.0 s slots are a boundary probe of the deadline mechanism, added after the pilot showed that deadlines almost never bind at 0.1 s. They are secondary and are reported as a stress test, not as realistic V2I latencies.

**Correlated loss.** In the burst block, a leg erased in an update's previous attempt is erased again with probability ψ = 0.5 (0.1 when independent). First attempts keep the independent law.

**Total.** 8,520 test runs, all audited.

## Primary comparison

`k1 − k0`: one safe retry minus the existing one-attempt protocol, at the base cell (ρ = 0.75, d = 0.1 s, q = 0.1). Metric: mean cohort delay per vehicle (s), one value per stream.

The 95% interval is a percentile interval from 10,000 stream resamples (seed 30,000), with a two-sided sign-flip permutation p-value. The practical band is ±0.1 s per vehicle. One primary, so no multiplicity correction.

The earlier study's one-attempt price at this cell (1.63 s per vehicle, study 5808) is a reference, not an assumed result.

## Secondary comparisons

Every family is reported for mean delay, p95 delay and throughput. Holm correction is applied within each family and metric.

| Family | Contrasts | Where |
|---|---|---|
| F1 | k1 − k0 | each lossy cell at d = 0.1 |
| F2 | k2 − k1 | same cells |
| F3 | aware2 − k2 | every lossy cell |
| F4 | k2 − oracle, k1 − oracle | every lossy cell (what recovery leaves unrecovered) |
| F5 | k1 − k0, k2 − k0, aware2 − k2 | each slot length |
| F6 | handoff_k0 − k0, handoff_k2 − k2, handoff_aware2 − handoff_k2, freeze_k0 − k0, freeze_k1 − k1 | mech cells |
| F7 | k1 − k0, k2 − k0, aware2 − k2 | correlated loss |
| Reference | k0, k1, k2, oracle against atomic rolling; k0, k1, k2 against atomic FCFS | loss cells |

**Also reported.**
- **Recovered share.** The share of the one-attempt price removed, (k0 − arm)/(k0 − atomic_rolling), at the base cell for k1, k2 and the oracle, with a stream bootstrap.
- **Descriptive measures:** pinned vehicles and pinned seconds; ambiguous vehicle-seconds; grants, retransmissions, queries, replies, duplicate grants and ACKs; move outcomes (confirmed, voided, unresolved by deadline or by budget, not issued); deferrals and extra transaction rounds; and wall time per run.
- **Deadline strata.** For every first attempt, the number of further rounds (0, 1, 2, 3+) in which a retransmission would still be certifiable, with counts of first-attempt failures and later resolutions per stratum. These strata are descriptive. Revision value and traffic state differ across them, so no causal reading is made.

## Stop rules

Any of the following stops the analysis: a violation, an undrained run, a lock relaxation, a missing (cell, arm, rep), `k0 ≠ k0_log` in delay or any protocol counter, or the burst-block `k0` differing from the independent `k0`. No run may be removed.

## What is not claimed

Fork-level effects and sustained-traffic effects are reported separately. Results hold for this simulator: synthetic independent (or, secondarily, persistence-correlated) erasures, bounded delay, synchronized clocks, exact tracking, cooperative vehicles, reliable initial bookings, serialized transactions, and one conflict point.
