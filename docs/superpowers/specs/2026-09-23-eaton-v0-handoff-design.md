# EATON v0 — transient operator handoff gate

Date: 2026-09-23

## Working claim

EATON stands for **Event-Addressed Transient Operator Networks**.

The repo will test a narrower claim than its name suggests:

> A useful computation can be an **episode** rather than a persistent setting. A sparse event temporarily activates an operator in resident state, that operator performs one local job, then relinquishes control so a different operator can use the changed state.

The machine is not assumed to run one persistent global algorithm. Its fast computation is the time-ordered composition of operator episodes over resident state.

This is a synthetic computational mechanism test. It is not a claim that biological dendrites, ephaptic fields, neuromodulators, or transformers literally implement the v0 equations.

## Why EATON is a new repo rather than another SimpleNeuron gate

The closest ancestors already cover adjacent pieces:

- `SimpleNeuron`: rich receiver state stays resident while tiny routed events travel; the same later cue can have different consequences because resident history differs.
- `NSSN2`: sparse recurrent nodes own state-conditioned local dynamics and publish tiny events.
- `FrequencyAddressedNonlinearModalCell`: carrier/frequency can address a rich stateful receiver and asks what cheap operator preserves that computation.
- `AnotherOddThing`: an event can be a probe into a resident operator, and active experiment selection can choose the useful poke.
- `EvoX` / GAx: alternative procedures and temporary computational excursions can be selected and reused.
- `AdaptiveObserverCache`: a persistent control state can redirect a local transformer read; the 2026-09-23 first-ask distance result motivates asking whether useful control should be phasic rather than continuously expressed.

EATON v0 does **not** re-test route semantics, state persistence, frequency addressing, operator identification, or evolutionary search. It isolates one missing mechanism:

```text
activate operator A
        ↓
perform A's local job
        ↓
state changes / event publishes
        ↓
release A
        ↓
operator B uses the changed state
```

The destructive alternative is to leave A active while B is trying to do its different job.

## V0 question

> **When two sequential jobs require different local operators, does transient release preserve downstream computation better than keeping the first operator tonically active, while still requiring resident state to survive the handoff?**

The point is not that tonic control is always bad. The point is to construct the smallest matched world in which *operator lifetime itself* is a load-bearing computational variable.

## Minimal machine

V0 uses one receiver because network topology is not the question yet.

Its resident state is

```text
x = [memory, decision_trace, payload_trace, output]
```

and it receives one scalar event channel `u[t]`.

Two event-addressable operators exist:

```text
A = acquire / select
B = use / transform
```

All coordinates start each episode from a matched nuisance state

```text
x0[j] ~ Normal(0, 0.05)
```

using the same sampled `x0` in all three arms.

When a coordinate is not actively written by an operator it undergoes passive retention

```text
z <- 0.98 * z
```

except `decision_trace`, which is frozen after the stage-1 decision so the early readout is not changed by later decay.

All explicit operator writes use `tanh`, making the state bounded.

### Operator A — acquire

A is event-gated. If A is active and a nonzero event `u` arrives:

```text
memory         <- tanh(0.40 * memory + 1.00 * u)
decision_trace <- memory
```

If A is active but no event arrives, it performs no special write; ordinary passive retention applies. Thus merely leaving A switched on is not itself destructive. Tonic interference occurs only when a later event arrives while A is still eligible to interpret it.

The early published decision is

```text
d_early = +1 if decision_trace >= 0 else -1
```

and is scored against context `c`.

All experimental arms must be identical through this boundary.

### Operator B — use

B has a two-tick episode so that simultaneous A/B activation on the payload tick has no ambiguous update ordering.

On B's **payload tick**, B only stores the incoming event:

```text
payload_trace <- tanh(u)
```

It does not read `memory` yet.

On the following **compute tick**, with no new event:

```text
output <- tanh(2.00 * memory * payload_trace)
```

The final published answer is

```text
y = +1 if output >= 0 else -1
```

and the target is

```text
y_target = c * p
```

so stage 2 requires both the earlier resident context and the new payload.

### Exact event schedule

Each episode has three logical ticks:

```text
t0  context tick
    event = c + eta_context
    A active
    B inactive
    record early decision after A writes

--- handoff boundary ---

optional reset happens here in reset arm only

t1  payload tick
    event = p + eta_payload
    B active: stores payload_trace
    transient: A released, memory only passively retained
    tonic:     A still active, so the same payload event also rewrites memory
    reset:     A released, reset memory only passively retained

t2  compute tick
    event = 0
    B active: computes output from current memory and payload_trace
    A causes no event-gated write in any arm
```

On `t1`, A and B write different state coordinates. Therefore the result is independent of Python call order: tonic A writes `memory`, B writes `payload_trace`, and B does not consume `memory` until `t2`.

### Why tonic A should interfere

A is an acquisition operator connected to the shared event channel. It is useful when `c` arrives. If it remains active when `p` arrives, it interprets the payload as new acquisition material:

```text
transient:
    c --A--> memory
    release A
    p --B--> payload_trace
    B(memory, payload_trace) --> answer

 tonic:
    c --A--> memory
    keep A eligible
    p --A+B--> memory rewritten + payload stored
    B(rewritten memory, payload_trace) --> answer
```

This is deliberately synthetic and transparent. V0 is an existence proof for operator lifetime as a computational degree of freedom, not a claim that this exact interference is biologically natural.

## Three matched arms

Every episode uses the same context, payload, nuisance noise tape, operator parameters, event budget, and initial state.

### 1. `transient`

A is eligible only at `t0`. At `t1` it is released; resident memory is retained passively while B stores the payload. B uses that retained memory at `t2`.

This is the EATON hypothesis arm.

### 2. `tonic`

A is eligible at both `t0` and `t1`. B activates on the same schedule as in `transient`. The only causal difference from `transient` is A's release time.

This is the main attacker.

### 3. `reset`

A runs exactly as in `transient`, and the same early decision is recorded. At the handoff boundary, immediately after the early decision and before `t1`, only `memory` is reset to that episode's original nuisance value `x0[memory]`. A is then released. All other state and all later events match `transient`.

This arm tests whether transient success actually depends on resident context surviving the handoff.

## Frozen world and nuisance variation

Use exactly 64 deterministic seeds numbered `0..63`.

For each seed evaluate exactly 256 episodes:

```text
64 repetitions of each balanced pair
(c,p) in {(-1,-1), (-1,+1), (+1,-1), (+1,+1)}
```

The four pair labels are shuffled once per seed and then replayed identically across arms.

For each episode sample, once and replay across arms:

```text
x0[j]       ~ Normal(0, 0.05) for four state coordinates
eta_context ~ Normal(0, 0.08)
eta_payload ~ Normal(0, 0.08)
```

The delivered events are

```text
u_context = c + eta_context
u_payload = p + eta_payload
```

No clipping is applied. Exact sign reversals from this noise scale are allowed to count as errors; they are part of the frozen nuisance distribution.

These coefficients, seeds, episode counts, noise scales, retention, and gains are frozen by this design before the canonical receipt exists. They must not be retuned after seeing v0 results.

## Primary measurements

For each arm and seed record:

1. **early decision accuracy** — `d_early` versus `c` after `t0`;
2. **final answer accuracy** — `y` versus `c*p` after `t2`;
3. **handoff memory retention** — signed correlation between `memory` immediately before B computes and the original `c`;
4. **operator overlap duration** — number of event ticks on which A and B are simultaneously eligible;
5. **event budget** — count and absolute-amplitude sum of externally delivered events;
6. **state boundedness** — maximum absolute resident coordinate.

Also record paired per-episode correctness differences between `transient` and each attacker.

## Frozen v0 gate

Classify `PASS_TRANSIENT_HANDOFF` only if **all** of the following hold on the canonical 64-seed run:

```text
median early decision accuracy, every arm      >= 0.95
stage-1 state/decision transient vs tonic       identical to atol 1e-12
stage-1 state/decision transient vs reset       identical to atol 1e-12
median final accuracy, transient               >= 0.90
median transient - tonic final accuracy        >= 0.20
median transient - reset final accuracy        >= 0.30
transient paired wins vs tonic                  >= 48 / 64 seeds
transient paired wins vs reset                  >= 56 / 64 seeds
event counts and external amplitude sums        identical to atol 1e-12
all state trajectories                          finite
max absolute state coordinate                   <= 1.000000000001
```

A seed counts as a paired win when transient final accuracy is strictly greater than the attacker's final accuracy on that seed.

Classification precedence:

1. If stage-1 equality or matched-event invariants fail: `INVALID_MATCHED_ARMS`.
2. Else if transient does not clear the tonic margin/win thresholds: `NO_HANDOFF_ADVANTAGE`.
3. Else if transient does not clear the reset margin/win thresholds: `HANDOFF_WITHOUT_RESIDENT_MEMORY`.
4. Else if transient absolute accuracy/boundedness thresholds fail: `UNSTABLE_OR_WEAK_TRANSIENT`.
5. Else: `PASS_TRANSIENT_HANDOFF`.

The reset arm is expected to approach chance on the balanced task, but no post-hoc reset ceiling is part of the pass rule.

## Destructive controls and invariants

Mechanism tests must establish before interpreting the scientific receipt:

- `transient` and `tonic` are bitwise/effectively identical through the end of `t0`;
- `reset` is also identical through the end of `t0`;
- all arms receive identical `(c,p)`, `x0`, and nuisance tapes;
- `transient` and `tonic` differ only in A eligibility at `t1`;
- `reset` differs from `transient` only by the explicit handoff memory reset;
- disabling A leaves the early decision dependent only on nuisance state and therefore at chance in aggregate;
- disabling B leaves the final output dependent only on nuisance state and therefore at chance in aggregate;
- clamping `memory` to the correct context sign before B's compute tick lets B solve the second stage;
- external event counts and amplitudes are matched exactly across arms;
- swapping Python execution order of A and B on `t1` leaves results unchanged because they write disjoint coordinates on that tick.

These are engineering invariants, not positive scientific results.

## Receipt

Canonical result:

```text
results/v0_handoff.json
```

It must contain:

```text
config
frozen parameter block
seed-level metrics
aggregate medians
paired-win counts
invariant checks
classification
claim boundary
```

The experiment runner should write a partial receipt before raising on a late failure when practical.

## Repository shape

Target minimal structure:

```text
README.md
pyproject.toml
src/eaton/
    __init__.py
    core.py          # resident state, events, transient operators
    world.py         # frozen balanced two-stage tapes
    experiment.py    # three matched arms + aggregation/classification
experiments/
    run_v0.py
tests/
    test_core.py
    test_v0_gate.py
results/
    v0_handoff.json
.github/workflows/
    ci.yml
docs/superpowers/specs/
    2026-09-23-eaton-v0-handoff-design.md
```

NumPy only for the mechanism. Pytest for tests. Python 3.11 and 3.12 CI.

## What v0 may establish

If the gate passes, the claim is only:

> In a matched synthetic resident-state machine where sequential jobs require different local operators, treating the first operator as a transient episode preserves downstream computation better than leaving it continuously expressed, and the successful handoff depends on resident state surviving between episodes.

That would earn `operator lifetime` as an explicit computational variable in this lineage.

## What v0 does not establish

A pass would **not** establish that:

- biological dendrites switch operators this way;
- neuromodulation or ephaptic coupling implements the release signal;
- transformers should always pulse attention interventions;
- phasic control is generally superior to tonic control;
- asynchronous overlap is useful;
- frequency is the correct address;
- the operator set is learned;
- EATON outperforms RNNs, transformers, SSMs, spiking networks, or conventional gated recurrent cells.

Those require later attackers or different tasks.

## Deliberately deferred gates

### v1 — asynchronous overlap / noncommutativity

Ask whether partially overlapping event-addressed operator episodes produce order-dependent computation that a settle-then-step endpoint model cannot reproduce.

### v2 — active-set address

Ask whether the currently active operator set itself becomes a useful state variable: the same incoming event has different meaning because a different transient operator constellation is active.

### v3 — slow residue

Add a much slower state that changes which transient operators can be recruited or how strongly they are expressed, connecting back to the GAx `anchor + excursion + residue` abstraction.

None of v1-v3 should be implemented until v0 has a frozen result.

## Relation to the 2026-09-23 AOC result

The AOC first-ask distance experiment motivates but does not validate EATON. At +256 masked positions the observer still reversed the matched local `valve`/`sensor` decision token while the complete sensor sentence did not win. One mundane explanation is that continuously steering later tail tokens damages follow-through. EATON v0 abstracts exactly that engineering possibility without Qwen:

```text
steer the useful boundary
        ↓
release the intervention
        ↓
let the changed resident state carry the next computation
```

A future AOC three-arm tail test can then ask whether Qwen exhibits the same phasic-control advantage. That would be a cross-system comparison, not evidence that the mechanisms are identical.
