# EATON

> **Event-Addressed Transient Operator Networks**
>
> The machine does not run one persistent global algorithm. Sparse events temporarily instantiate operators in resident state; computation is the time-ordered handoff between those episodes.

EATON asks whether **operator lifetime** can itself be a useful computational variable.

The motivating abstraction is deliberately simple:

```text
rich resident state
      ↑
sparse event addresses a transient operator
      ↓
operator changes resident state / publishes a small event
      ↓
operator releases
      ↓
a different operator continues from the changed state
```

This repository is a synthetic mechanism laboratory, not a biological neuron model. The first gate does not claim that dendrites, neuromodulators, ephaptic fields, or transformers literally implement these equations.

## v0 — transient operator handoff

The first frozen question is:

> When two sequential jobs require different local operators, does transient release preserve downstream computation better than keeping the first operator tonically active, while still requiring resident state to survive the handoff?

One receiver owns four resident coordinates:

```text
x = [memory, decision_trace, payload_trace, output]
```

A context event first activates acquisition operator **A**:

```text
memory         <- tanh(0.40 * memory + event)
decision_trace <- memory
```

After the early decision, a payload arrives. Transform operator **B** stores the payload and one tick later combines it with resident memory:

```text
payload_trace <- tanh(payload_event)
output        <- tanh(2.00 * memory * payload_trace)
```

The target is the parity-like relation `context * payload`, so neither input alone is sufficient.

Three matched arms receive exactly the same contexts, payloads, nuisance states, event noise, and event budget:

```text
transient   A handles context, releases, then B uses retained memory

tonic       A handles context and stays eligible when payload arrives;
            A therefore rewrites memory while B stores the payload

reset       A handles context and releases, but resident memory is erased
            before B receives the payload
```

The tonic and transient machines are identical through the first decision. They differ only in A's lifetime at the payload boundary. The reset arm asks whether successful handoff actually depends on resident state surviving between operator episodes.

## Frozen v0 result

Canonical receipt: [`results/v0_handoff.json`](results/v0_handoff.json)

Classification:

```text
PASS_TRANSIENT_HANDOFF
```

The receipt was generated after the design, coefficients, nuisance distribution, seeds, episode budget, and pass thresholds were frozen. No result-dependent retuning was performed.

Across **64 deterministic seeds × 256 episodes per seed**:

| metric | transient | tonic | reset |
|---|---:|---:|---:|
| median early decision accuracy | **1.0000** | **1.0000** | **1.0000** |
| median final accuracy | **1.0000** | 0.5000 | 0.5020 |
| median context/memory correlation before B computes | **0.99894** | 0.17570 | 0.00777 |

Paired seed results:

```text
transient > tonic final accuracy: 64 / 64
transient > reset final accuracy: 64 / 64

median transient - tonic final margin: +0.500000
median transient - reset final margin: +0.498047
```

All matched-arm invariants passed. All trajectories were finite and bounded; the maximum absolute state coordinate across the canonical run was `0.9215206003`.

### What the result means

In this deliberately constructed machine, the first operator is useful for acquiring context but harmful if it remains eligible for a later event whose role is different. Releasing A allows B to use the context that A left in resident state. Keeping A tonically active overwrites that context; erasing the context destroys the handoff for a different reason.

So v0 earns one narrow statement:

> **In a matched synthetic resident-state machine, operator lifetime can be a load-bearing computational variable. A transient operator episode can hand state to a different computation more successfully than either tonic persistence or state reset.**

This is an existence proof. The task was designed to contain exactly this interference structure, so the clean result is not evidence that transient control generally beats recurrent networks, transformers, state-space models, or ordinary gating.

## Exact three-tick schedule

```text
t0  context event
    A active
    -> write resident memory
    -> record early decision

    handoff boundary

 t1 payload event
    B stores payload_trace

    transient: A released
    tonic:     A still eligible and rewrites memory
    reset:     A released; memory was reset at handoff

 t2 no external event
    B computes from memory * payload_trace
    -> final decision
```

A and B write disjoint coordinates on `t1`, so the tonic result is invariant to whether the implementation evaluates A or B first.

## Why this is separate from the ancestor repos

EATON is not re-proving their earlier claims:

- [`SimpleNeuron`](https://github.com/anttiluode/SimpleNeuron) — rich state stays resident while tiny routed events travel; resident history changes a later cue.
- [`NSSN2`](https://github.com/anttiluode/NSSN2) — sparse recurrent receivers with state-conditioned local dynamics.
- [`FrequencyAddressedNonlinearModalCell`](https://github.com/anttiluode/FrequencyAddressedNonlinearModalCell) — address a rich receiver and ask what reduced operator preserves its computation.
- [`AnotherOddThing`](https://github.com/anttiluode/AnotherOddThing) — perturb resident operators and choose informative probes.
- [`EvoX`](https://github.com/anttiluode/EvoX) / GAx — retain/select alternative procedures and temporary computational excursions.
- [`AdaptiveObserverCache`](https://github.com/anttiluode/AdaptiveObserverCache) — persistent observer state redirects local reads in a real transformer cache.

EATON v0 isolates what those did not: **when should an invoked operator stop being expressed?**

## Relation to the AOC clue

The 2026-09-23 first-ask AOC distance experiment motivated this gate. At +256 masked cache positions, observer steering still reversed the local `valve`/`sensor` decision-token preference, while the complete steered sentence did not win. One ordinary possibility is that continuing the intervention after the useful boundary damages follow-through.

EATON v0 does not prove that explanation for Qwen. It gives the hypothesis a clean synthetic witness. A separate AOC test can later compare tonic steering against a phasic intervention that switches off after the decision boundary.

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --out results/v0_handoff.json
```

CI runs the test suite on Python 3.11 and 3.12. The committed receipt is regression-tested against a fresh canonical recomputation.

## Repository map

```text
src/eaton/core.py        exact resident state and three-tick operators
src/eaton/world.py       deterministic balanced matched episode tapes
src/eaton/experiment.py  metrics, invariants, aggregation, classification
experiments/run_v0.py    canonical frozen runner
results/v0_handoff.json  first frozen scientific receipt
tests/                   mechanism + destructive + receipt regression tests
docs/superpowers/specs/  approved frozen design
docs/superpowers/plans/  implementation plan
```

## Claim boundary

The result is a **synthetic existence proof only**. It does not establish that:

- biological dendrites switch operators this way;
- neuromodulation or ephaptic coupling is the release mechanism;
- phasic control is generally superior to tonic control;
- transformers should always pulse attention interventions;
- frequency is the correct address;
- asynchronous overlap is useful;
- the transient operator set is learned;
- EATON outperforms conventional architectures.

## Deferred gates

The next questions remain deliberately unimplemented until v0 is frozen and reviewed:

```text
v1  asynchronous overlap / noncommutativity
    can partially overlapping operator episodes compute something
    a settle-then-step endpoint model cannot?

v2  active-set address
    can the currently active operator constellation itself become state,
    changing what the same incoming event means?

v3  slow residue
    can a slower anchor/residue change which transient operators can be
    recruited, connecting EATON back to the GAx anchor + excursion idea?
```
