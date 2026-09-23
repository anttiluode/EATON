# EATON

> **Event-Addressed Transient Operator Networks**
>
> EATON asks whether useful computation is sometimes better described as a succession of temporary local operators over resident state than as one persistent global algorithm.

The project started from a narrow operator-lifetime question and now asks a sharper black-box question: **can learned dynamics themselves be collapsed into reusable computation coordinates and a temporal grammar?**

```text
state coordinate       = where the recurrent system is
computation coordinate = how the local dynamics transform matched perturbations here
temporal grammar       = event-conditioned transition law among recovered coordinates
```

EATON is a synthetic mechanism laboratory. It is not a biological neuron model, and similarities to dendrites, neuromodulation, transformer paths, or sensorimotor loops are hypotheses/analogies unless a gate tests them directly.

## v0 — transient operator handoff

V0 hand-authored two operators in one resident-state receiver and isolated one question: **when should an invoked operator stop being expressed?**

A context event activates acquisition operator `A`, which writes resident memory. A later payload is handled by operator `B`. Three matched arms differ only at the handoff:

```text
transient   A handles context, releases, B consumes retained memory
tonic       A stays eligible and rewrites memory when payload arrives
reset       A releases, but resident memory is erased before B
```

Canonical receipt: [`results/v0_handoff.json`](results/v0_handoff.json)

```text
PASS_TRANSIENT_HANDOFF
```

Across **64 deterministic seeds × 256 episodes per seed**:

| metric | transient | tonic | reset |
|---|---:|---:|---:|
| median early decision accuracy | **1.0000** | **1.0000** | **1.0000** |
| median final accuracy | **1.0000** | 0.5000 | 0.5020 |
| median context/memory correlation before B computes | **0.99894** | 0.17570 | 0.00777 |

```text
transient > tonic final accuracy: 64 / 64
transient > reset final accuracy: 64 / 64
```

### V0 claim boundary

V0 is a **constructed synthetic gating demonstration**. Its tonic failure is analytically implied by the chosen coefficients: leaving the acquisition writer active lets the payload overwrite the context needed downstream. That makes the harness useful, but it does **not** establish that phasic control is generally superior, that biology uses this schedule, or that operator lifetime is a newly discovered principle. Classical input-gating mechanisms already address the overwrite problem.

## v1 — computation coordinates

V1 removes the hand-authored operator labels from the learner.

A small tanh RNN is trained only to predict symbol sequences. In the structured world, one of three unobserved transition rules persists through time with rare switches. In the volatile control, the same rule mixture is resampled independently at every step. The network never receives the hidden rule label.

After training, the recurrent network is frozen and treated as a black box. At each pre-input hidden state `h_t`, the same small bank of perturbations is applied:

```text
R_t[:, j] = (F(h_t + eps * v_j, s_t) - F(h_t, s_t)) / eps
```

`R_t` is a local response signature: an empirical approximation to how the dynamics transform perturbations **here**. V1 clusters these response signatures rather than raw hidden activations.

The recovery path is deliberately restricted:

- hidden world-rule labels are unavailable during discovery;
- analytic Jacobians are unavailable during discovery;
- RNN weights are not clustering features;
- train/validation/test tapes are disjoint;
- `k = 1..8` competes under a frozen validation MDL/BIC-like score;
- test data are assigned only after `k` is selected.

The matched activation attacker performs the same clustering/model-selection procedure on raw `h_t`. It is then judged on the same target as response coordinates: how well its clusters reconstruct **response signatures**. Other attackers use time bins, current input alone, time-shuffled response signatures, and a separately trained RNN on the volatile world.

### Frozen v1 result

Canonical receipt: [`results/v1_coordinates.json`](results/v1_coordinates.json)

```text
PASS_COMPUTATION_COORDINATES
```

All **8 / 8** structured seeds met the preregistered training-validity criterion. MDL-like selection chose `k=8` for response, activation, and shuffled-response coordinates on every valid seed.

| measurement | frozen median/result |
|---|---:|
| response-coordinate response NRMSE | **0.445310** |
| activation-coordinate response NRMSE | 0.675727 |
| response / activation NRMSE ratio | **0.725984** |
| response beats activation NRMSE | **8 / 8 seeds** |
| response transition NLL | **0.311043** |
| shuffled-response transition NLL | 2.007189 |
| shuffled grammar gain | **1.697674 nats/transition** |
| time-bin transition NLL | 0.408855 |
| time-bin grammar gain | **0.098152 nats/transition** |
| response coordinate beats both grammar attackers | **7 / 8 seeds** |
| response-coordinate next-symbol accuracy | **0.900933** |
| activation-coordinate next-symbol accuracy | 0.937826 |
| current-symbol baseline gain of response coordinate | **+0.557726** |
| structured-minus-volatile grammar specificity | **0.726521 nats/transition** |
| finite-difference / analytic-Jacobian relative error | **1.809863e-06** |
| post-hoc rule mutual information | 0.644891 nats |
| post-hoc best rule agreement | 0.803711 |

### What v1 earns

V1 earns the narrow statement frozen in the design:

> **A trained recurrent system can contain recurring, recoverable local response modes whose temporal organization provides a compact predictive description of its computation.**

The strongest distinction is not simply that the hidden state contains task information. Clustering by **what the local dynamics do to matched perturbations** reconstructed held-out response behavior substantially better than clustering by **where the hidden activation is**, while the recovered coordinate sequence also carried temporal and next-symbol information beyond the preregistered attackers.

The volatile control matters because it asks whether the grammar is just a property of the RNN/probing machinery. The structured-vs-volatile specificity remained strongly positive in the frozen panel.

### What v1 does not establish

`PASS_COMPUTATION_COORDINATES` does **not establish** that:

- the recovered `k=8` alphabet is unique or is the network's one true algorithm;
- every neural computation admits a small discrete grammar;
- local Jacobian action fully defines computation;
- the hidden three-rule world and the recovered eight coordinates should coincide one-to-one;
- biological cortex literally uses discrete computation coordinates;
- transformers have the same temporal grammar;
- EATON has solved mechanistic interpretability.

In particular, the activation-coordinate attacker retained slightly higher next-symbol accuracy than the response-coordinate readout. The result is therefore not “response coordinates dominate state representations.” The narrower finding is that response-space gives a substantially better compression of **local operator behavior**, while remaining temporally predictive.

## Relation to Dong, Cordonnier & Loukas

Dong et al. decompose self-attention networks depth-wise into paths through attention heads and show that skip connections create paths of varying effective length; in their tested tasks, short isolated paths carry much of the predictive power.

EATON does not claim that their path decomposition proves a temporal-operator theory. The useful analogy is methodological:

```text
Dong:  decompose depth-wise computation into paths
EATON: decompose time-wise learned dynamics into reusable response coordinates + transitions
```

An identity/skip route is also a reminder that preserving resident state can itself be computationally meaningful; EATON does not hard-code a `NULL` operator in v1.

## Relation to the surrounding repos

- [`SimpleNeuron`](https://github.com/anttiluode/SimpleNeuron) — resident history changes what a later small routed event means.
- [`NSSN2`](https://github.com/anttiluode/NSSN2) — sparse recurrent receivers with state-conditioned local dynamics.
- [`FrequencyAddressedNonlinearModalCell`](https://github.com/anttiluode/FrequencyAddressedNonlinearModalCell) — addresses richer local computation through a small carrier coordinate.
- [`AnotherOddThing`](https://github.com/anttiluode/AnotherOddThing) — asks which intervention best distinguishes hidden candidate operators.
- [`EvoX`](https://github.com/anttiluode/EvoX) / GAx — slow anchor plus temporary computational excursions and residue.
- [`AdaptiveObserverCache`](https://github.com/anttiluode/AdaptiveObserverCache) — tests persistent observer state and causal read control inside a frozen transformer cache.

EATON's current slot is the **temporal-description layer**: discover recurring local transformations, ask how long/when they are expressed, and determine whether their succession is a smaller causal description than the microscopic trajectory.

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --out results/v0_handoff.json
python experiments/run_v1.py --out results/v1_coordinates.json
```

CI runs the test suite on Python 3.11 and 3.12. Both canonical receipts are regression-tested against fresh recomputation.

## Repository map

```text
src/eaton/core.py            v0 resident-state handoff mechanism
src/eaton/world.py           v0 matched episode tapes
src/eaton/experiment.py      v0 metrics/classification

src/eaton/grammar_world.py   v1 structured + volatile symbol worlds
src/eaton/rnn.py             deterministic NumPy tanh RNN
src/eaton/coordinates.py     response probes, clustering, reconstruction
src/eaton/grammar.py         transition/readout/MDL metrics
src/eaton/v1_experiment.py   attackers, aggregation, frozen classification

experiments/run_v0.py        v0 canonical runner
experiments/run_v1.py        v1 canonical runner
results/v0_handoff.json      frozen v0 receipt
results/v1_coordinates.json  frozen v1 receipt
tests/                       mechanism, attacker, and receipt regressions
```

## Next gates

V1 turns “computations are coordinates” into an executable claim. The next gates are deliberately harder:

```text
v2  cross-implementation equivalence
    train independent networks on the same task and ask whether different
    weights collapse to comparable computation grammars

v3  overlap / noncommutativity
    allow partially overlapping computational episodes and ask whether
    timing itself carries irreducible computation

v4  slow anchor / GAx bridge
    add a slower persistent state that changes which transient computation
    coordinates can be recruited by the same event

v5  transformer bridge
    apply response-coordinate recovery to a frozen transformer intervention
    trajectory rather than a toy recurrent network
```
