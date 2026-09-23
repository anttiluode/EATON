# EATON v1 — computation coordinates and temporal grammar recovery

Date: 2026-09-23

## Working question

EATON v1 asks a different question from v0.

V0 hand-authored two operators and tested one release boundary. Its result is useful as a harness/gating demonstration, but the outcome is analytically implied by the chosen coefficients.

V1 therefore removes the hand-authored operator labels from the learner and asks:

> **Can a frozen recurrent system trained only on temporally structured data be read back as a small set of recurring computation-coordinates, where coordinates are defined by local response operators rather than by raw activation similarity, and where the resulting coordinate sequence forms a compact temporal grammar?**

The intended object is not a biological claim and not a claim that every neural network has a unique symbolic program. It is a falsifiable black-box-reading experiment.

## Core idea

A recurrent system has resident state `h_t`, receives an event `x_t`, and advances to

```text
h_{t+1} = F(h_t, x_t)
```

The usual representation analysis clusters `h_t` itself.

V1 instead asks what small perturbations would do *at that state*.

Choose one frozen set of hidden-state probe directions `v_1 ... v_P` and a frozen epsilon `eps`. For every observed state:

```text
R_t[:, j] = (F(h_t + eps * v_j, x_t) - F(h_t, x_t)) / eps
```

Flatten `R_t` into one local response signature.

The signature is an experimentally accessible approximation to a local operator/Jacobian action. Two states may be far apart in activation space but belong to the same **computation coordinate** if the same set of perturbations is transformed similarly there.

V1 clusters these response signatures, not hidden vectors, and asks whether the resulting discrete coordinates both:

1. compress the local operator field, and
2. compose into a predictable temporal grammar over held-out trajectories.

The compact slogan is:

```text
state coordinate:       where the system is
computation coordinate: what the system currently does to perturbations
```

## Why the data must have temporal structure

The motivating hypothesis from the 2026-09-23 discussion is that orderly world-time can induce orderly computational-time.

So v1 trains the same small recurrent architecture on two worlds that share the same observable alphabet and the same one-step rule mixture but differ in whether the hidden rule persists through time.

### Structured world

There are four observable symbols `0..3` and three unobserved transition rules. The active rule is persistent, with rare switches.

For a current symbol `s`, the rules produce distinct next symbols:

```text
rule 0: next = (s + 1) mod 4
rule 1: next = (s + 2) mod 4
rule 2: next = (s + 3) mod 4
```

The active rule is not given to the network. It must infer it from recent transitions and retain that belief while the rule persists.

### Volatile control world

At every step, the rule is resampled independently from the same marginal distribution used in the structured world.

Thus the current-symbol-to-next-symbol mixture is matched in expectation, but temporal persistence is destroyed. History cannot reliably identify which rule will govern the next transition.

This control asks whether a recovered temporal grammar is merely an artifact of the recurrent architecture/probing pipeline or whether it tracks learnable temporal structure in the data.

## Recurrent model

Use one small vanilla tanh RNN implemented only with NumPy:

```text
h_{t+1} = tanh(W_h h_t + W_x onehot(s_t) + b_h)
logits_t = W_y h_{t+1} + b_y
```

Train by next-symbol cross entropy with full-sequence teacher forcing.

The architecture, optimizer, sequence lengths, train/validation/test counts, hidden width, initialization scale, clipping rule, learning rate, epoch budget, and random seeds will be frozen in the implementation plan before canonical results are generated.

The model itself is not the research contribution; it is deliberately small enough that the full local dynamics can be checked.

## Black-box discipline

The recovery algorithm may call only:

```text
F(h, x)
```

for observed `h`, observed `x`, and the frozen perturbation probes.

It must not use:

- hidden world-rule labels,
- the analytic Jacobian,
- RNN weight matrices as clustering features,
- training gradients,
- hand-authored phase labels.

The analytic Jacobian may be computed afterward only as an oracle validation that finite-difference response signatures really approximate local differential behavior.

## Candidate computation coordinates

For each candidate coordinate count

```text
k in {1, 2, 3, 4, 5, 6, 7, 8}
```

fit deterministic NumPy k-means on training response signatures.

Initialization is frozen by seed and uses k-means++ style seeding implemented in-repo. Empty clusters are repaired deterministically from the worst represented sample.

Validation and test points are assigned by nearest response centroid.

No value of `k` is selected because the world happens to contain three latent rules.

## Temporal grammar

Let `z_t` be the recovered computation coordinate.

The simplest v1 grammar is a first-order event-conditioned transition model:

```text
P(z_{t+1} | z_t, s_t)
```

with additive smoothing frozen before the canonical run.

This is intentionally modest. V1 does not yet infer arbitrary context-free productions or asynchronous overlapping operators. It tests whether a small coordinate alphabet plus transition law is already a useful reduction of the learned dynamics.

## Model selection: closest compact grammar

The phrase "closest temporal grammar given the dataset" is made operational by a frozen MDL/BIC-like score.

For each `k`, the training fit defines:

1. a spherical Gaussian response model around each response centroid, and
2. the event-conditioned coordinate transition table.

Validation code length combines:

```text
response negative log likelihood
+ transition negative log likelihood
+ 0.5 * parameter_count * log(N_train)
```

The exact parameter count and numerical floor rules will be frozen in the implementation plan.

The selected grammar is the `k` with minimum validation description length. Test data are untouched until after `k` is selected.

This is called **MDL-like** rather than a claim of a universal minimum-description-length optimum.

## Primary measurements

### 1. Operator reconstruction

For each test response signature `R_t`, reconstruct it by the centroid of its assigned computation coordinate.

Report normalized response reconstruction error:

```text
NRMSE_response
```

This is the most direct measurement of whether the coordinates summarize local computation.

### 2. Temporal grammar predictability

Evaluate held-out event-conditioned transition NLL:

```text
NLL(z_{t+1} | z_t, s_t)
```

and compare it with shuffled-coordinate controls.

### 3. Predictive sufficiency for the world

Fit a tiny table on training data:

```text
P(s_{t+1} | z_t, s_t)
```

and report held-out next-symbol accuracy/NLL.

This does not replace the RNN; it asks how much task-relevant context survives the coordinate collapse.

### 4. Oracle alignment, reported only after discovery

Because the synthetic world has hidden rule labels, measure post-hoc mutual information / best-permutation agreement between recovered coordinates and latent world rules.

This metric is diagnostic only. It does not participate in fitting or model selection.

### 5. Finite-difference validity

After all discovery choices are frozen, compare the finite-difference response signature to the exact analytic Jacobian action on the same probe directions.

This checks the measuring instrument, not the scientific claim.

## Required attackers

V1 is not allowed to pass merely because k-means discovers clusters in anything.

### Attacker A — activation coordinates

Cluster raw hidden states `h_t` with the same candidate `k`, selection protocol, train/validation split, and deterministic k-means implementation.

For each activation cluster, reconstruct local response by the mean training response signature for that activation cluster.

Question:

> Does clustering by what the system *does* summarize operator behavior better than clustering by where the activation happens to be?

### Attacker B — time bins

Replace learned coordinates by `k` equal-progress bins over sequence position.

This attacks the trivial possibility that a temporal task simply has early/middle/late phases.

### Attacker C — current input only

Use the current observable symbol `s_t` as the coordinate.

This attacks the possibility that the recovered grammar is just a recoding of the visible event.

### Attacker D — shuffled response signatures

Permute response signatures across time before clustering while keeping the event/trajectory tapes unchanged.

This attacks pipeline-induced grammar structure.

### Attacker E — volatile world

Train and analyze an otherwise identical RNN on the volatile control world.

This attacks the claim that compact temporal operator structure arises even when the data contain no persistent hidden temporal regularity to exploit.

## Frozen pass classifications

The implementation plan will convert the following into exact numeric thresholds before canonical results are generated.

Conceptually, v1 has three possible outcomes.

### `PASS_COMPUTATION_COORDINATES`

Requires all of:

- the structured-world RNN learns a real history advantage over the volatile-control ceiling/current-symbol baseline;
- response-coordinate clustering reconstructs held-out local response signatures materially better than activation clustering at matched selected complexity;
- recovered coordinate transitions are materially more predictable than shuffled-coordinate and time-bin controls;
- the collapsed coordinate plus current symbol retains substantial held-out next-symbol information;
- the result is stable across a preregistered multi-seed panel rather than one lucky RNN.

This would earn only the narrow claim:

> A trained recurrent system can contain recurring, recoverable local response modes whose temporal organization provides a compact predictive description of its computation.

### `PASS_STATE_NOT_OPERATOR`

The RNN learns the task and hidden activations contain useful temporal state, but response-coordinate clustering does not beat matched activation clustering for operator reconstruction/grammar quality.

Interpretation: the current experiment supports state-space structure but not the stronger "computation coordinates" framing.

### `FAIL_GRAMMAR_RECOVERY`

Training succeeds but none of the recovered low-complexity coordinate systems gives a stable, predictive temporal grammar beyond attackers.

Interpretation: v1 does not justify discretizing the learned dynamics into reusable operator episodes.

Training failure is reported separately and is not counted as evidence against the scientific hypothesis.

## What v1 does not claim

A pass would not establish that:

- biological cortex literally uses discrete operator coordinates;
- transformers share the same recovered grammar;
- the recovered coordinate alphabet is unique;
- computation is always low-dimensional or symbolic;
- first-order Markov grammar is sufficient in general;
- local Jacobian response fully defines computation;
- the hidden world rules are themselves the network's algorithm;
- EATON has solved mechanistic interpretability.

## Relation to Dong et al. (2021/2023 revision)

Dong, Cordonnier, and Loukas decompose self-attention networks into paths through attention heads and show that skip connections create paths of varying effective lengths; their experiments find short isolated paths carry much more predictive power than long ones in the tested tasks.

EATON v1 does not reproduce that result. The useful connection is methodological:

```text
Dong:  decompose depth-wise computation into paths
EATON: attempt to decompose time-wise learned dynamics into reusable local response coordinates and transitions
```

The identity/skip path also motivates treating "preserve state / no special local transformation" as a legitimate eventual coordinate rather than forcing every interval to host an elaborate operator. V1 does not hard-code such a NULL coordinate; if one exists, it must emerge from response geometry.

## Relation to AOC

AdaptiveObserverCache remains the strongest immediate real-model test of phasic versus tonic intervention.

EATON v1 is complementary. It asks whether one can *discover* a compact temporal operator vocabulary from a trained dynamical system without hand-authoring the operator schedule first.

If v1 works, a later EATON/AOC bridge can apply the same response-signature idea to selected Qwen residual/query states and ask whether many microscopic states collapse onto a smaller causal temporal grammar.

## Deferred gates

If v1 passes, the next gates become:

```text
v2  cross-implementation equivalence
    train independent networks on the same task;
    ask whether different weights collapse to comparable computation grammars

v3  overlap / noncommutativity
    allow simultaneous or partially overlapping computation coordinates;
    ask whether timing itself carries irreducible computation

v4  slow anchor / GAx bridge
    add a slower persistent state that changes which transient coordinates
    can be recruited by the same event

v5  transformer bridge
    apply response-coordinate recovery to a frozen transformer intervention
    trajectory rather than to a toy recurrent network
```

## Repository shape

V1 should extend the repo without disturbing the frozen v0 receipt.

Proposed files:

```text
src/eaton/grammar_world.py      structured and volatile symbol worlds
src/eaton/rnn.py                small NumPy tanh RNN + deterministic training
src/eaton/coordinates.py        probe bank, response signatures, k-means, MDL-like selection
src/eaton/grammar.py            transition tables and grammar metrics
src/eaton/v1_experiment.py      multi-seed orchestration and classification
experiments/run_v1.py           canonical runner
results/v1_coordinates.json     frozen canonical receipt after implementation

tests/test_grammar_world.py
tests/test_rnn.py
tests/test_coordinates.py
tests/test_grammar.py
tests/test_v1_experiment.py
```

The existing v0 modules/results/tests remain unchanged except for README/index exports if needed.

## Design principle

The key destructive check is simple:

> **Do not reward EATON for finding clusters. Reward it only if clustering local response behavior yields a smaller, more predictive description of the learned dynamics than matched descriptions based on activations, time, visible input, or shuffled structure.**

That is the first gate that can turn "computations are coordinates" from a metaphor into an executable claim.
