# Methodology: model, algorithms and experimental design

This document expands Sections 2–5 of the report (`report/CS329_Final_Report.pdf`) and points
to the code that implements each part. Equation numbers refer to the report.

## 1. Model

| Element | Definition | Code |
|---|---|---|
| Intersection | Two single-lane FIFO approaches sharing one conflict zone | `simulation/engine.py` |
| Arrivals | Poisson requests, split equally between lanes; offered load ρ = λh with h = 3 s | `simulation/scenario.py` |
| Readiness | Private r_i = a_i + L₀ + σ Σₖ Z_ik, four signs Z_ik ∈ {−1, +1}, one revealed per second; L₀ = 8.25 s | `models/belief.py`, `simulation/scenario.py` |
| Public guard | U_i(k) = a_i + L₀ + σ(Σ_{m≤k} Z_im + 4 − k) ≥ r_i (Eq. 1); never increases | `simulation/engine.py` (`U`) |
| Motion | Double integrator: acceleration ≤ α = 3 m/s², braking ≤ b = 3 m/s², speed ≤ V = 6 m/s, approach L = 8 m. Head of lane: reachable entry window W_i(t). Followers wait at rest and launch when the predecessor enters (T_L ≈ 2.33 s) | `models/physics.py` |
| Headways | Cross-lane h = 3 s; same lane max(h_f, T_L) with h_f = 3 s (1.5 s in some sweep cells) | `scheduling/core.py` |
| Delay | S_i − r_i for the cohort requesting in [300, 900) s (full-stream studies) | `simulation/engine.py` (`_record`) |

**Why the guard.** Booking a vehicle at or after U_i is safe for every possible future signal, so
information can only *improve* the feasible schedule (a negative sign advances the guard by 2σ).
This isolates the question of the project: improvements keep arriving, and the coordinator must
decide whether and how to install them.

## 2. Scheduler (rolling optimization)

At every event (request, signal, entry) the coordinator solves the snapshot problem (Eq. 2):
minimize the sum of entry times of live vehicles subject to the guard, motion windows, FIFO,
headways, order locks and (in the closed loop) blocks for bookings that cannot be revised.

`scheduling/core.py::solve_dp` is a label-setting dynamic program over keys (i, j) = number of
vehicles of lane 0 and lane 1 already scheduled. A label is (cost, last start on lane 0, last start
on lane 1); a label is dominated if another label at the same key is no worse in all three, because
every completion is monotone in these quantities. The program is exact unless a key holds more than
32 non-dominated labels (`SimConfig.cap`); every run records the flag `approx`, and it is false in
every recorded run of all three studies (checked by the analysis stop rules).
`solve_bruteforce` enumerates all interleavings; `tests/test_scheduling.py` and `tests/test_blocks.py`
compare the two, and `validate_plan` independently checks every installed plan.

## 3. Commitment policies (Experiment A)

All policies share the scheduler, physics and safety layer; they differ only in which cross-lane
order locks they install (`policies/base.py`).

| Policy | Rule |
|---|---|
| `rolling` | No voluntary locks: orders stay flexible |
| `fcfs` | Each new request is locked behind every live request of the other lane |
| `freeze_H` | A vehicle's position is locked once it is within H s of its estimated readiness |
| `freeze_inf` | Every vehicle is locked as soon as it is first scheduled ("lock at request") |
| `adaptive` | For each adjacent cross-lane pair, a finite optimal-stopping rule compares committing now with waiting for the remaining public signals, using a kinematic surrogate κ for the cost of a late reversal (`stopping/pair.py`); KEEP locks the pair |

Freeze horizons are tuned per (ρ, σ) cell on separate tuning streams; ties go to the more flexible
policy.

## 4. Safe installation of a revised schedule

### 4.1 Authorization ledger and safety

For each booked vehicle the manager keeps a protected set P_i (initially {O_i}).
* **Issue.** A grant O_i → T_i is first issued only if T_i is compatible with every booking in every
  other vehicle's protected set; issuing adds T_i to P_i. Retransmissions add nothing.
* **ACK.** A matching ACK (sent only after the vehicle switched) sets P_i = {T_i}.
* **Void.** A matching void report (sent only by a vehicle still on O_i, in reply to a query sent after
  the last attempt) sets P_i = {O_i}; the vehicle never executes that grant afterwards.
* **Never on silence.** Timeouts and duplicates change nothing.

*Invariant.* (i) Each vehicle's active booking lies in its protected set; (ii) protected bookings of
different vehicles are pairwise compatible. Proof sketch: (i) a switch moves to a booking that was
added when the grant was issued; an ACK or void shrinks P_i only to the booking the vehicle actually
holds. (ii) Issue checks compatibility; ACKs and voids only remove bookings. Hence the executed
bookings are jointly feasible whatever messages are lost. The simulator audits (i), (ii) and the
soundness of the manager's beliefs after every protocol event, booking and entry
(`replacement/closed_loop.py`, `replacement/retry.py`); no recorded run has a violation.

### 4.2 Release graph and reachability

For a direct target (each changed vehicle moves once, O_i → T_i) put j → i if T_i conflicts with O_j.
If a set D of vehicles has switched, the only possibly incompatible pairs are (T_i, O_j) with i ∈ D,
j ∉ D, so the hybrid is feasible iff D contains every blocker of each member; it is *reachable by
single switches* iff, in addition, D induces no cycle (a cycle has no first move). With monotone
guards, the earliest schedule in the issued order is componentwise no later than O, so every edge of
an order-preserving update points forward in the order: such updates are always acyclic. Reversing a
tightly spaced pair creates a 2-cycle (Fig. 1 of the report). Experiment B confirms this on held-out
data: perfect-link shortfalls occur in exactly the reordering forks (`experiments/sweep/analyze.py`
asserts it).

### 4.3 Value of an installation (Eq. 3)

Execute a predecessor-closed set S of certified moves, each issued once, as soon as all its parents
have acknowledged. Move e is issued iff every strict ancestor's grant and ACK arrived (2m_e
independent legs, counting each distinct ancestor once); given issue, it takes effect iff its own
grant arrives. So P(e effective) = u(uv)^{m_e}. A vehicle's final booking telescopes over its
effective moves, so by linearity of expectation (no independence between vehicles needed)

    V(S) = Σ_{e∈S} (a_e − b_e) u (uv)^{m_e}.

If a temporary booking's return fails, the park's (negative) term remains: stranding is charged
exactly. Maximizing V over predecessor-closed sets is a maximum-weight closure (Picard, 1976); the
implementation enumerates the excluded negative-weight moves, which is exact.
`tests/test_handoff.py` checks the formula against exhaustive enumeration of every delivery history.

### 4.4 Temporary reservations (one-detour handoffs)

For a cycle-blocked holder k and temporary time x, split k into park p_k (O_k → x) and return q_k
(x → T_k) with p_k → q_k; original edges j → i become first(j) → last(i); add e_j → p_k if x conflicts
with O_j and q_k → e_j if x conflicts with T_j; x may not conflict with an unchanged vehicle. The
target is completable with this detour iff the split graph is acyclic. Compatibility of x with a
booking y changes only at y ± H, so it suffices to test every boundary, one interior point of each
cell between boundaries, the window ends and the unbounded tail. Every move, including the return
from the parked trajectory, must be physically certified (`replacement/handoff.py`,
`replacement/bridge.py`). Candidates are scored by Eq. 3.

### 4.5 Physical certification

A move switching at τ is certified if τ < min(a, b) and the new booking lies in the reachable entry
window from every state the vehicle could be in at τ: the head's actual profile (replanned at its
earlier switches), or, for a staged follower, every compatible entry time of its predecessor. The test
is recomputed for every transmission at its own switch instant; it is sufficient, not necessary.

### 4.6 Transactions and the one-attempt contract

Installation runs in serialized transactions of rounds of length 2d. A grant sent in round r takes
effect, if delivered, at τ_r = t₀ + (2r + 1)d; its reply arrives before the next round. Decisions
arriving during a transaction are deferred, and one re-solve follows. A newcomer's first booking is
delivered reliably. A vehicle whose grant is unconfirmed keeps two protected bookings and is *pinned*:
planned at the later booking, blocking the earlier one, and not revised until resolved or entered.
With K = 0 (`replacement/closed_loop.py::ClosedLoop`), a pinned vehicle stays pinned until it enters.

### 4.7 Bounded, version-safe retransmission (`replacement/retry.py`)

Every message carries an update identifier and a base version.
* `GRANT(gid, i, base, a → b)` is retransmitted with the same identity; only the switch instant changes.
* The vehicle executes gid at most once, at an attempt's switch instant, only if it holds a at
  version `base`, the attempt is timely and gid is not void; it re-acknowledges duplicates.
* `QUERY(gid)`: a vehicle that executed gid re-acknowledges; a vehicle still on a voids gid and reports.
* The manager sends an attempt only if the switch is certified at that attempt's own instant, sends no
  grant after a query, and releases only on a matching ACK, a void after the last attempt, or entry.

Rules: **blind** (retransmit while certifiable and budget remains), **aware** (retransmit or issue
late only if the move's own change plus the value it still unlocks is positive, else query), and
**oracle** (one attempt with reliable, informative feedback; a comparator, not a protocol).
With budget 0 the retry code reproduces `ClosedLoop` event for event (`tests/test_retry.py`, and the
analysis stop rule `k0 == k0_log` in every cell).

For one update with n usable transmissions (`replacement/retry_value.py`):
P(switch) = 1 − (1−u)^n, P(confirmed) = 1 − (1−uv)^n. With u = v = 0.9 a round fails to confirm with
probability 0.19, so 1, 2 and 3 attempts leave 19%, 3.6% and 0.69% of updates ambiguous; the
recovery study observes pinned shares close to these.

## 5. Experimental design

| Study | Question | Design | Code |
|---|---|---|---|
| Commitment (archived E3, study key 329) | A: does locking orders help? | Instantaneous installation; tuning: 6 loads × 4 noise levels × 8 freeze policies × 16 streams; held-out test: 6 loads, σ = 1, 5 policies, 60 streams | `configs/commitment.toml`, `experiments/commitment/` |
| Regime sweep (study 5807) | B, C: what can be installed at one decision? | 3 loads × 2 noise × 2 headways × 32 held-out streams; every signal-triggered rolling decision in [20, 80) s is a fork; 6 arms × 4 links | `configs/sweep.toml`, `experiments/sweep/` |
| Recovery (study 5809) | D, E: price of safe installation and value of recovery | σ = 1; ρ ∈ {0.5, 0.75}; 60 held-out streams per cell; loss, slot, mechanism and correlated-loss blocks | `configs/recovery.toml`, `experiments/recovery/` |

**Common random numbers.** Every random quantity is drawn from `numpy.random.SeedSequence` streams
keyed by (study, split, replication, stream type, lane or vehicle). All arms of a replication share
arrivals, signs and first-transmission packet draws; retransmissions use a separate stream.

**Held-out data.** Each design was committed before its test streams were run (sweep: research-archive
commit `74662a0`; recovery: `f4dd91f`; commitment: tuning split run and selected before the test split,
design in the archived `configs/studies.toml`). Method code was unchanged between each freeze and its
results. The frozen design records are kept in `docs/study_designs/`.

**Statistics** (`src/lockorder/evaluation/stats.py`). Paired differences per stream; percentile
intervals from 10,000 stream resamples (seed 30000); paired sign-flip permutation p-values; Holm
correction within each pre-specified family; fork effects averaged per live vehicle, then over forks
within a stream, then equally over streams. Cells are never pooled; fork-level and sustained-traffic
quantities are never combined.

**Stop rules** (checked by the analyses and by `experiments/reproduce_report.py`; any failure aborts before results are written): a safety violation, an undrained
run, an approximate solve, a missing (cell, arm, replication), a lock relaxation, or the zero-budget
retry code differing from the one-attempt closed loop.
