> **Archived design record.** Copied unchanged from the research archive (study 5809 protocol audit, commit ee60ef8). It was written and committed before the held-out streams of that study were run. File and command paths in it refer to the archive layout; in this repository the corresponding files are `src/lockorder/replacement/closed_loop.py` and `src/lockorder/replacement/retry.py`.

# Protocol audit before retransmission (one-attempt contract at `08bc331`)

Scope: the reservation-update protocol as implemented in
`src/lockorder/replacement/closed_loop.py` (sustained traffic), `protocol.py` and
`handoff.py` (isolated forks), and the scheduler hooks it uses
(`Snapshot.blocks` in `scheduling/core.py`). The audited code was not edited. Its
outputs are pinned by `tests/test_closed_loop_golden.py` (commit `f3578cd`) and by
`tests/test_legacy_golden.py`. The quantitative probes below come from
`experiments/retransmission/audit_probe.py`. They use study key 1111, the test-fixture key, and
never touch a held-out key.

## 1. What the one-attempt protocol is

| Object | Closed-loop implementation |
|---|---|
| Manager evidence | `P[i]`, the set of bookings still protected for live vehicle *i* |
| Vehicle state | Implicit. `slot[i]` is the physical booking; there is no version counter or inbox |
| Grant | Issued in round *k* at `t0 + 2kd` if the target is compatible with every other protected slot; adds the target to `P[i]` |
| Delivery | With probability *u*, exactly at `t0 + (2k+1)d`, which is also the switch time |
| ACK | Sent after a switch; with probability *v*, it arrives at `switch + d·U`, `U ∈ [0,1)`. Sets `P[i] = {target}` |
| End of transaction | `t0 + 2(K+1)d`. Any vehicle with two protected slots is then **pinned** until its entry is observed |
| Pinned vehicle | Planned at `max P[i]`; `min P[i]` becomes a block for every other vehicle; never revised again |
| Release by entry | `_enter` deletes `P[i]`. Observed entry is the only release besides an ACK |

## 2. Safety invariant (task §2.1)

Write *Act_i* for vehicle *i*'s active booking. There are two invariants:

- **(I1)** *Act_i* ∈ *P_i*.
- **(I2)** For *i* ≠ *j*, *C_ij(x, y)* holds for all *x* ∈ *P_i*, *y* ∈ *P_j*.

**Where they are checked.** In audit mode, `_check_ledger` checks both after every timed protocol event: send, switch, ACK, gated booking.

**Bookings and entries.** Neither runs `_check_ledger`.
- Newcomer bookings made inside `decide()` assert (I2) inline (`_book`). (I1) holds trivially for them, since `P = {s}` and the slot is `s`.
- Entries only delete a protected set, so they cannot break either invariant.

So the invariants hold after every event. The earlier wording, "checked after every protocol event", is slightly broader than what the code literally does; the extension checks after bookings and entries too.

**Results.** All 3,288 archived closed-loop runs and 32 golden runs show zero violations. An independent check, the engine's entry checks for marks, clearances, FIFO, launch and locks, also found none.

**Verdict.** No safety bug found under the model's assumptions (§4).

## 3. Message cases (task §2.2)

| # | Case | One-attempt closed loop | Fork executor (`protocol.Drivers`) |
|---|---|---|---|
| 1 | Grant never reaches the vehicle | Correct. The vehicle keeps its old slot, the manager keeps both slots, and the vehicle is pinned | Correct |
| 2 | Grant delivered, ACK lost | Correct. The vehicle switched; both slots stay protected; pinned | Correct |
| 3 | Delayed grant arriving after the switch is infeasible | Cannot occur, because delivery is exactly at the switch time. There is no rejection path: a late grant would raise `AssertionError` (fail-stop, not silent) | Rejected (`now > g.at`) |
| 4 | Duplicate grant | Cannot occur (one attempt). A duplicate would fail the base-slot check and raise (fail-stop) | Buffered idempotently |
| 5 | Stale grant from an earlier version | Cannot occur (serialized transactions, bounded delay). There is no version field to reject it | Rejected by epoch/version |
| 6 | ACK delivered after a retry | Not applicable | Not applicable |
| 7 | ACK delivered after the vehicle executed a later version | Cannot occur under bounded delay. **But `_ack` has no version check**: a stale ACK arriving after a later grant would shrink `P[i]` to an outdated slot, which would be unsafe | Matched against the issued grant |
| 8 | Vehicle enters before its transaction resolves | Correct. Entry deletes `P[i]`, and a late ACK for an entered vehicle is ignored | Correct |

**Finding F1 (latent, not triggered).** The closed loop's ACK handling is safe only because two things together exclude every reordered or stale message:
- one-way delay is bounded by *d*, and
- transactions are serialized, with each vehicle granted at most once per round.

Any retransmission scheme breaks this, because it creates several in-flight messages for one grant. So every grant, ACK and status message in the extension carries a grant identifier and a base version, and the manager ignores any reply that does not match the outstanding grant.

**Finding F2 (latent).** The closed loop has no explicit vehicle-side state machine. Its safety rests on the manager's own scheduling: one attempt, delivery exactly at the switch time. A duplicate or late message would fail-stop rather than be handled. The extension adds an explicit vehicle agent with:
- a version counter,
- an execution log (grant → slot and execution time),
- a void set,
- duplicate, stale and late rejection.

**Finding F3 (one float edge, checked).** An ACK sent near the end of its window arrives at `t0 + (2k+1)d + d·U`. Floating-point rounding could in principle put it after the next round's send, which would leave a dependent unreleased. That outcome is conservative: the dependent is simply not issued. A probe of 809 grants found no case (`late_ack = 0`).

**Minor.**
- The `book` handler asserts `self.compatible is not None`, which is vacuous.
- Packet uniforms wrap after 256 grants per vehicle; the observed maximum is 5.

Neither affects any result.

## 4. Assumption audit (task §2.3)

"Safe" below means the model's invariants (I1, I2) plus the engine's entry checks. It is a property of the simulated contract, **not a real-world road-safety claim**.

| Assumption | What depends on it |
|---|---|
| Cooperative, non-forging vehicles that execute only timely granted switches and report truthfully | **Safety.** A vehicle using an unprotected slot breaks (I1) |
| Synchronized clocks; switch at the grant's stated time | **Safety.** Certificates are computed for that instant |
| Exact physical tracking (actual head profile, actual launch) | **Safety.** The physical certificate and the follower-launch argument use it |
| Bounded one-way delay (≤ *d*) | **Safety** of the version-free ACK handling (F1); also the round structure |
| Independent per-leg erasures | **Performance only.** The value formula u(uv)^m and every quantitative result |
| Reliable initial bookings for newcomers | **Performance**; also an availability assumption: an unbooked vehicle must wait at the stop line |
| Correct observation of actual entry | **Safety.** Entry releases every protected slot of the vehicle |
| Serialized transactions | **Safety** of the version-free ledger (F1) and simplicity of the argument; **performance** through deferred decisions |

Safety never depends on delivery probabilities. It depends on cooperation, synchronized time, bounded delay, tracking and entry observation.

## 5. Where the one-attempt cost comes from (probe at ρ = 0.75, d = 0.1 s, 10% loss per leg, rolling target, 4 pilot streams of study 1111)

- **Pinned vehicles.** 158 of 809 grants (19.5%) left a vehicle pinned. The theoretical value is 1 − 0.9² = 19%.
  - 94 (59%) were lost grants: the vehicle still held its old slot.
  - 64 (41%) were lost ACKs: the vehicle had switched.
- **Pinned targets are always earlier.** Every pinned grant targeted an *earlier* slot, because direct-mode revisions only advance vehicles; guards never retreat. So a pinned vehicle is planned at its *old, later* slot, while its *earlier* target is blocked for everybody else. Two clearance windows are held for one vehicle.
- **Duration.** A pinned vehicle stays pinned for 9.1 s on average, from transaction end to entry. During that time it receives none of the improvements that later sign disclosures would justify.
- **Lost grants after the earlier slot.** For a lost grant, the earlier slot passes without entry. Under exact tracking this proves the vehicle holds the later slot, yet the manager keeps it pinned for another 1.65 s on average. This is an inefficiency, not a safety issue. Silence in the *protocol* is not evidence; a *physical* non-entry at a committed time is. The extension does not use this inference, so that the comparison isolates communication recovery, but it is recorded as a further lever.

## 6. Proposed corrections and requirements for the extension

None of the findings is a safety bug under the stated assumptions, so the one-attempt code stays unchanged as Baseline A. Its outputs are pinned by the closed-loop goldens. The retransmission extension must:

1. **Tag every message.** Each grant, ACK and status message carries a grant id and base version (F1). The manager acts only on a reply that matches the outstanding grant, and duplicate ACKs are idempotent.
2. **Model the vehicle explicitly (F2).** The vehicle executes grant *n* only if all of these hold:
   - its version equals *n*'s base version and it holds *n*'s base slot;
   - *n* is not void;
   - the attempt is timely;
   - it has not entered.

   It never executes a version twice, and it re-acknowledges an executed version whenever it receives that version again (duplicate grant or query).
3. **Never release on silence.** Shrink `P[i]` only on:
   - a matching ACK (retires the base slot);
   - a matching status reply proving the grant is void (retires the target slot);
   - observed entry.
4. **Re-certify physically** at every attempt's own switch time. A decision-time certificate is never reused for a later instant.
5. **Count every retransmission and query** as a message and give it its own round, and so its own time.
6. **Keep conservative fallback.** On budget exhaustion the vehicle stays pinned exactly as in Baseline A.
7. **Reproduce Baseline A bit for bit when the retry budget is zero.**
8. **Check (I1) and (I2)** after every protocol event, newcomer booking and entry, and check the vehicle agent against the physical booking.
