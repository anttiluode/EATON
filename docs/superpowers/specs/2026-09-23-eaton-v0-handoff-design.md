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

Each node owns a resident state. V0 uses one receiver because network topology is not the question yet.

Use a four-coordinate latent state:

```text
x = [memory, decision_trace, payload_trace, output]
```

and one scalar event channel `u[t]`.

Two event-addressable operators exist:

```text
A = acquire / select
B = use / transform
```

They are ordinary deterministic state-update functions with bounded nonlinearities. The implementation may use `tanh` to keep state bounded, but the pass/fail claim must not depend on a delicate saturation constant.

### Operator A — acquire

During the first stage, a context event `c ∈ {-1,+1}` arrives. While A is active it writes the sign of the event into resident memory and a decision trace.

Conceptually:

```text
memory        <- retain(memory) + gain_A * u
decision_trace <- memory
```

The early published decision is:

```text
d_early = sign(decision_trace)
```

All experimental arms must get this first-stage decision right. A result where the tonic attacker simply fails stage 1 is uninteresting.

### Operator B — use

During the second stage, a payload event `p ∈ {-1,+1}` arrives on the **same event channel**. B combines the resident memory laid down by A with the new payload to compute a final answer:

```text
output <- combine(memory, payload)
```

The canonical v0 target is the parity-like relation

```text
y_target = c * p
```

so the second stage cannot be solved from the payload alone and cannot be solved from the stored context alone.

The exact bounded implementation can be bilinear or an equivalent tiny nonlinear map. It must be identical across all arms.

### Why tonic A should interfere

A is an **acquisition operator** connected to the shared event channel. It is useful when `c` arrives. If A remains active when `p` arrives, it continues treating the new event as material to write into the same resident memory that B is trying to use.

That is the v0 interference mechanism:

```text
phasic:
    c --A--> memory
    release A
    p --B(memory,p)--> answer

 tonic:
    c --A--> memory
    p --A+B--> memory is rewritten while B uses it
```

This is deliberately synthetic and transparent. V0 is an existence proof for operator lifetime as a computational degree of freedom, not a claim that this exact interference is biologically natural.

## Three matched arms

Every episode uses the same context, payload, nuisance noise tape, operator parameters, event budget, and initial state.

### 1. `transient`

A is active only for the acquisition window. It is explicitly released before the payload event. B is then active for the transform window.

This is the EATON hypothesis arm.

### 2. `tonic`

A is activated by the same first event and is **not released** before the payload event. B activates on schedule, so the second stage runs with A and B simultaneously expressed.

No operator weights, gains, event counts, or state dimensions differ from `transient`. Only A's lifetime differs.

This is the main attacker.

### 3. `reset`

A runs exactly as in `transient`, and the same early decision is recorded. Before B receives the payload, the resident memory coordinates written by A are reset to their neutral state.

This arm tests whether transient success actually depends on state surviving the handoff rather than on a hidden shortcut in B or the event tape.

## World and nuisance variation

Use 64 deterministic seeds by default.

For each seed, evaluate a balanced factorial tape over the four `(c,p)` pairs with repeated matched nuisance perturbations. Suggested canonical budget:

```text
256 episodes / seed
64 repetitions of each of the four (c,p) combinations
```

Small zero-mean noise may perturb event amplitude and resident initialization. The same sampled noise is replayed across all three arms. Noise exists to prevent a single exact algebraic equality from being the entire gate, not to make the task difficult.

No parameter tuning is allowed after the canonical receipt is observed. If implementation reveals that a proposed parameter makes the gate mathematically degenerate before the frozen run, change it before freezing and document the reason.

## Primary measurements

For each arm and seed record:

1. **early decision accuracy** — `sign(decision_trace)` versus `c` after stage 1;
2. **final answer accuracy** — `sign(output)` versus `c*p` after stage 2;
3. **handoff memory retention** — norm/correlation of the context-bearing resident component immediately before B acts;
4. **operator overlap duration** — number of ticks for which A and B are simultaneously active;
5. **event budget** — must match across arms;
6. **state norm / boundedness** — detect trivial blow-up or saturation.

Also record paired per-episode disagreements between `transient` and each attacker.

## Frozen v0 gate

Classify `PASS_TRANSIENT_HANDOFF` only if **all** of the following hold on the canonical 64-seed run:

```text
median early decision accuracy, all arms       >= 0.95
median final accuracy, transient               >= 0.90
median transient - tonic final accuracy        >= 0.20
median transient - reset final accuracy        >= 0.30
transient paired wins vs tonic                  >= 48 / 64 seeds
transient paired wins vs reset                  >= 56 / 64 seeds
event budgets                                   exactly matched
all state trajectories                          finite and bounded
```

The reset arm is expected to approach chance (`~0.5`) on the balanced task, but no upper threshold is required for the scientific classification; the required transient-minus-reset margin is the actual test.

If transient does not beat tonic, the result is `NO_HANDOFF_ADVANTAGE`.

If transient beats tonic but not reset, the result is `HANDOFF_WITHOUT_RESIDENT_MEMORY` and the intended EATON interpretation fails.

If the early decision differs materially across arms, classify `INVALID_STAGE1_MATCH` rather than interpreting final accuracy.

## Destructive controls and invariants

The implementation must test these mechanically before the scientific run:

- `transient` and `tonic` are identical through the end of stage 1;
- all arms receive identical `(c,p)` and nuisance tapes;
- `transient` and `tonic` differ only in A's release time;
- `reset` differs from `transient` only by the explicit between-stage memory reset;
- with A disabled entirely, early accuracy falls to chance;
- with B disabled entirely, final parity accuracy falls to chance;
- with memory artificially clamped to the correct context before B, B can solve the second stage;
- event counts and event amplitudes are matched across arms.

These are engineering invariants, not positive scientific results.

## Receipt

Canonical result:

```text
results/v0_handoff.json
```

It should contain:

```text
config
seed-level metrics
aggregate medians
paired-win counts
invariant checks
classification
claim boundary
```

The experiment runner must write a partial receipt before raising on a late failure when practical.

## Repository shape

Target minimal structure:

```text
README.md
pyproject.toml
src/eaton/
    __init__.py
    core.py          # resident state, events, transient operators
    world.py         # balanced two-stage tapes
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
