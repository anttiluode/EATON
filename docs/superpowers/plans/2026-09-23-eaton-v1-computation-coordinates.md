# EATON v1 Computation Coordinates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen NumPy experiment that trains a small recurrent network on temporally structured versus temporally volatile symbol streams, recovers discrete computation-coordinates from local finite-difference response operators, and tests whether those coordinates form a compact predictive temporal grammar beyond matched activation/time/input/shuffle attackers.

**Architecture:** Add five focused v1 modules beside the frozen v0 code: a world generator, a small trainable tanh RNN, response-coordinate recovery, grammar/MDL metrics, and a canonical multi-seed experiment orchestrator. The recovery pipeline treats the trained RNN as a black box through `step`/`step_batch`; hidden world-rule labels and analytic Jacobians remain unavailable to model selection and are used only for post-hoc validation.

**Tech Stack:** Python >=3.11, NumPy >=1.26, pytest >=8; no new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-eaton-v1-computation-coordinates-design.md`

## Global Constraints

- Preserve the frozen v0 implementation, tests, and canonical receipt unchanged.
- Runtime dependencies remain exactly NumPy >=1.26; tests use pytest >=8.
- All canonical computations use float64.
- The recovery pipeline may use only observed symbols, observed hidden states, and calls to the frozen recurrent transition function; it must not use hidden world-rule labels, analytic Jacobians, training gradients, or raw weight matrices as clustering features.
- Hidden world-rule labels are diagnostic-only after model selection.
- Analytic Jacobian action is diagnostic-only after model selection.
- Candidate coordinate counts are exactly `k = 1..8`; no hard-coded `k=3` preference is permitted.
- Canonical seed panel is exactly `0..7`.
- No result-dependent retuning: all training values, thresholds, smoothing, probe settings, and classification rules below are frozen before `results/v1_coordinates.json` is generated.

## Frozen v1 numerical configuration

```python
WORLD = {
    "alphabet_size": 4,
    "rule_count": 3,
    "sequence_length": 48,
    "train_sequences": 384,
    "validation_sequences": 96,
    "test_sequences": 96,
    "structured_switch_probability": 0.05,
}

RNN = {
    "hidden_size": 16,
    "init_scale": 0.15,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
}

COORDINATES = {
    "probe_count": 8,
    "probe_epsilon": 1e-5,
    "k_candidates": tuple(range(1, 9)),
    "kmeans_max_iter": 100,
    "kmeans_restarts": 4,
    "feature_std_floor": 1e-8,
    "gaussian_variance_floor": 1e-8,
    "transition_smoothing": 0.5,
}

PASS = {
    "minimum_valid_structured_seeds": 6,
    "structured_accuracy_min": 0.80,
    "structured_accuracy_gain_over_current_symbol_min": 0.25,
    "response_vs_activation_nrmse_ratio_median_max": 0.90,
    "response_vs_activation_seed_win_fraction_min": 0.75,
    "grammar_gain_over_shuffled_nats_median_min": 0.10,
    "grammar_gain_over_timebin_nats_median_min": 0.05,
    "grammar_seed_win_fraction_min": 0.75,
    "collapsed_world_accuracy_median_min": 0.75,
    "collapsed_world_accuracy_gain_over_current_symbol_min": 0.25,
    "structured_minus_volatile_grammar_gain_median_min": 0.05,
    "finite_difference_relative_error_median_max": 1e-3,
}
```

A structured seed is **valid for scientific aggregation** only when its frozen RNN reaches test next-symbol accuracy >=0.80 and exceeds its train-fitted current-symbol baseline by >=0.25. Fewer than 6 valid structured seeds yields `TRAINING_FAILURE`, not evidence against the computation-coordinate hypothesis.

## Classification rules

`PASS_COMPUTATION_COORDINATES` requires at least 6 valid structured seeds and all of these on the valid-seed panel:

```text
median(response_nrmse / activation_nrmse) <= 0.90
response_nrmse < activation_nrmse on >= 75% of valid seeds
median(shuffled_transition_nll - response_transition_nll) >= 0.10 nats/transition
median(timebin_transition_nll - response_transition_nll) >= 0.05 nats/transition
response_transition_nll beats shuffled and time-bin on >= 75% of valid seeds
median(response-coordinate world accuracy) >= 0.75
median(response-coordinate world accuracy - current-symbol baseline) >= 0.25
median(structured grammar gain over shuffled - volatile grammar gain over shuffled) >= 0.05 nats/transition
median finite-difference/Jacobian relative error <= 1e-3
```

`PASS_STATE_NOT_OPERATOR` requires at least 6 valid structured seeds, failure of one or more response-over-activation/operator-grammar criteria above, and median activation-coordinate world accuracy >=0.75 with >=0.25 gain over current-symbol baseline.

`FAIL_GRAMMAR_RECOVERY` applies when training is valid but neither pass classification applies. `TRAINING_FAILURE` applies first when fewer than 6 structured seeds satisfy the frozen training criterion.

## Review Focus

- **Rule-timing leakage:** a generated rule at step `t` must govern `s_t -> s_{t+1}` and must never be inferred from future symbols when training or probing; Task 1 tests exact alignment on hand-computed tapes.
- **Train/validation/test leakage:** split RNG streams and model-selection data must be disjoint and deterministic; Tasks 1 and 5 test split reproducibility and that test data are not consumed by `select_k`.
- **Degenerate clustering:** constant features, `k > distinct_points`, and empty clusters must remain finite and deterministic; Task 3 pins floor and repair behavior.
- **Sequence-boundary contamination:** temporal grammar counts must never create transitions from the end of one sequence to the start of another; Task 4 has an explicit two-sequence destructive test.
- **Coordinate-label permutation:** grammar/world metrics must be invariant to a renaming of cluster IDs; Task 4 tests a full label permutation.

---

### Task 1: Temporally structured and volatile worlds

**Files:**
- Create: `src/eaton/grammar_world.py`
- Create: `tests/test_grammar_world.py`

**Interfaces:**
- Produces `WorldConfig`, `WorldBatch`, `generate_world_split(config, seed, split, volatile)`, and `current_symbol_baseline(train, test)`.
- `WorldBatch.symbols` has shape `(n_sequences, sequence_length + 1)` and integer dtype; `WorldBatch.rules` has shape `(n_sequences, sequence_length)` and is diagnostics-only outside this module.

- [ ] **Step 1: Write failing tests for exact rule semantics, temporal persistence, volatile marginals, and split determinism**

```python
import numpy as np
from eaton.grammar_world import WorldConfig, generate_world_split


def test_rule_offsets_match_definition():
    cfg = WorldConfig(sequence_length=4, train_sequences=1, validation_sequences=1, test_sequences=1)
    # helper accepts explicit symbols/rules only in tests
    from eaton.grammar_world import apply_rules
    symbols = apply_rules(start_symbol=0, rules=np.array([0, 1, 2, 0]), alphabet_size=4)
    assert symbols.tolist() == [0, 1, 3, 2, 3]


def test_structured_rules_persist_more_than_volatile():
    cfg = WorldConfig()
    s = generate_world_split(cfg, seed=3, split="train", volatile=False)
    v = generate_world_split(cfg, seed=3, split="train", volatile=True)
    structured_same = np.mean(s.rules[:, 1:] == s.rules[:, :-1])
    volatile_same = np.mean(v.rules[:, 1:] == v.rules[:, :-1])
    assert structured_same > 0.90
    assert 0.25 < volatile_same < 0.42


def test_split_generation_is_deterministic_and_disjoint():
    cfg = WorldConfig()
    a = generate_world_split(cfg, 5, "validation", False)
    b = generate_world_split(cfg, 5, "validation", False)
    c = generate_world_split(cfg, 5, "test", False)
    assert np.array_equal(a.symbols, b.symbols)
    assert not np.array_equal(a.symbols, c.symbols)
```

- [ ] **Step 2: Run the new world tests and verify they fail because the module does not exist**

Run: `pytest tests/test_grammar_world.py -q`
Expected: import failure for `eaton.grammar_world`.

- [ ] **Step 3: Implement the frozen world generator**

Implement these exact semantics:

```python
@dataclass(frozen=True)
class WorldConfig:
    alphabet_size: int = 4
    rule_count: int = 3
    sequence_length: int = 48
    train_sequences: int = 384
    validation_sequences: int = 96
    test_sequences: int = 96
    structured_switch_probability: float = 0.05

@dataclass(frozen=True)
class WorldBatch:
    symbols: np.ndarray
    rules: np.ndarray

RULE_OFFSETS = np.array([1, 2, 3], dtype=np.int64)
```

For structured data, draw `r_0` uniformly. For every `t`, first apply `r_t` to form `s_{t+1}`; only then draw `r_{t+1}`, keeping `r_t` with probability `0.95` and otherwise choosing uniformly from the other two rules. For volatile data, draw every `r_t` independently and uniformly from `0..2`. Start symbols are uniform over `0..3`.

Use split-specific seed offsets exactly `train:+1000`, `validation:+2000`, `test:+3000`, plus `+100_000` for volatile worlds, so all tapes are deterministic and disjoint.

`current_symbol_baseline(train, test)` fits a 4x4 count table on train transitions with additive smoothing `0.5` and returns test accuracy and mean NLL.

- [ ] **Step 4: Run world tests and the frozen v0 suite**

Run: `pytest tests/test_grammar_world.py tests/test_world.py tests/test_core.py -q`
Expected: all pass.

- [ ] **Step 5: Commit the world layer**

```bash
git add src/eaton/grammar_world.py tests/test_grammar_world.py
git commit -m "feat: add EATON v1 temporal worlds"
```

### Task 2: Small deterministic NumPy RNN

**Files:**
- Create: `src/eaton/rnn.py`
- Create: `tests/test_rnn.py`

**Interfaces:**
- Consumes `WorldBatch`.
- Produces `RNNConfig`, `RNNParams`, `TrainResult`, `step`, `step_batch`, `rollout_states`, `sequence_metrics`, and `train_rnn`.
- `rollout_states(params, symbols)` returns pre-input hidden states with shape `(N, T, H)` so response probing at `(h_t, s_t)` is unambiguous.

- [ ] **Step 1: Write failing tests for state timing, deterministic training, clipping, and learnability**

```python
import numpy as np
from eaton.rnn import RNNConfig, init_params, step, rollout_states, train_rnn
from eaton.grammar_world import WorldConfig, generate_world_split


def test_rollout_returns_pre_input_state():
    p = init_params(RNNConfig(hidden_size=3), seed=0)
    symbols = np.array([[0, 1, 2]], dtype=np.int64)
    states = rollout_states(p, symbols)
    assert states.shape == (1, 2, 3)
    assert np.allclose(states[0, 0], 0.0)
    assert np.allclose(states[0, 1], step(p, states[0, 0], 0)[0])


def test_training_is_bitwise_deterministic_for_same_seed():
    w = WorldConfig(train_sequences=32, validation_sequences=16, test_sequences=16, sequence_length=12)
    train = generate_world_split(w, 2, "train", False)
    a = train_rnn(train, RNNConfig(hidden_size=8, epochs=3, batch_size=8), seed=9)
    b = train_rnn(train, RNNConfig(hidden_size=8, epochs=3, batch_size=8), seed=9)
    assert all(np.array_equal(x, y) for x, y in zip(a.params.arrays(), b.params.arrays()))
    assert a.loss_history == b.loss_history


def test_training_reduces_loss_on_persistent_rule_world():
    w = WorldConfig(train_sequences=64, validation_sequences=16, test_sequences=16, sequence_length=20)
    train = generate_world_split(w, 1, "train", False)
    r = train_rnn(train, RNNConfig(hidden_size=8, epochs=10, batch_size=16), seed=1)
    assert r.loss_history[-1] < 0.8 * r.loss_history[0]
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `pytest tests/test_rnn.py -q`
Expected: import failure for `eaton.rnn`.

- [ ] **Step 3: Implement the recurrent model and full-sequence BPTT**

Use exactly:

```python
h_{t+1} = tanh(h_t @ W_h.T + W_x[:, s_t] + b_h)
logits_t = h_{t+1} @ W_y.T + b_y
```

with `W_h:(H,H)`, `W_x:(H,4)`, `W_y:(4,H)`. Initialize all weight matrices from `Normal(0, init_scale/sqrt(fan_in))`; biases are zeros. Use float64 throughout.

Implement vectorized mini-batch full-sequence cross-entropy and reverse-time BPTT. Global gradient clipping is:

```python
norm = sqrt(sum(sum(g*g) for g in grads))
if norm > clip:
    grads *= clip / norm
```

Use deterministic Adam with the frozen values in the header; never shuffle with global NumPy state. A seeded generator creates an epoch permutation.

`train_rnn` returns all 50 epoch mean losses; no early stopping is allowed in canonical runs.

- [ ] **Step 4: Run focused RNN tests, then all v0 tests**

Run: `pytest tests/test_rnn.py -q && pytest tests/test_core.py tests/test_experiment.py tests/test_v0_gate.py tests/test_world.py -q`
Expected: all pass.

- [ ] **Step 5: Commit the RNN**

```bash
git add src/eaton/rnn.py tests/test_rnn.py
git commit -m "feat: add deterministic recurrent learner"
```

### Task 3: Computation-coordinate response signatures and deterministic clustering

**Files:**
- Create: `src/eaton/coordinates.py`
- Create: `tests/test_coordinates.py`

**Interfaces:**
- Consumes `RNNParams`, pre-input hidden states, and current symbols.
- Produces `CoordinateConfig`, `FeatureScaler`, `KMeansModel`, `make_probe_bank`, `response_signatures`, `analytic_response_signatures`, `fit_kmeans`, `assign_clusters`, `fit_response_reconstruction`, and `response_nrmse`.

- [ ] **Step 1: Write failing tests for finite differences, deterministic k-means, constant features, and empty-cluster repair**

```python
import numpy as np
from eaton.coordinates import (
    CoordinateConfig, make_probe_bank, response_signatures,
    analytic_response_signatures, fit_kmeans, response_nrmse,
)
from eaton.rnn import RNNConfig, init_params


def test_finite_difference_matches_analytic_jacobian_action():
    p = init_params(RNNConfig(hidden_size=6), seed=2)
    states = np.random.default_rng(1).normal(size=(12, 6)) * 0.2
    symbols = np.arange(12) % 4
    probes = make_probe_bank(6, 4, seed=8)
    fd = response_signatures(p, states, symbols, probes, epsilon=1e-5)
    exact = analytic_response_signatures(p, states, symbols, probes)
    rel = np.linalg.norm(fd - exact) / np.linalg.norm(exact)
    assert rel <= 1e-3


def test_kmeans_is_deterministic_and_repairs_empty_clusters():
    x = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
    a = fit_kmeans(x, k=3, seed=4, max_iter=30, restarts=2)
    b = fit_kmeans(x, k=3, seed=4, max_iter=30, restarts=2)
    assert np.array_equal(a.labels, b.labels)
    assert np.all(np.isfinite(a.centroids))
    assert len(np.unique(a.labels)) == 3


def test_constant_dimensions_stay_finite():
    x = np.ones((20, 5))
    m = fit_kmeans(x, k=1, seed=0, max_iter=10, restarts=1)
    assert np.all(np.isfinite(m.scaler.scale))
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest tests/test_coordinates.py -q`
Expected: import failure for `eaton.coordinates`.

- [ ] **Step 3: Implement the response instrument**

`make_probe_bank(H,P,seed)` draws an `H x P` Gaussian matrix, QR-orthonormalizes it, and returns `Q.T` with shape `(P,H)`. Canonical `P=8 <= H=16`.

For batches, compute each probe column with `step_batch`:

```python
base = step_batch(params, states, symbols)[0]
perturbed = step_batch(params, states + eps * probe[j], symbols)[0]
action = (perturbed - base) / eps
```

Concatenate actions in probe-major order to `(N, P*H)`.

`analytic_response_signatures` computes the diagnostic-only action

```python
J_t v = diag(1 - h_next**2) @ W_h @ v
```

for the same probes. Keep this function physically separate from the discovery path and never call it from k selection.

- [ ] **Step 4: Implement deterministic standardized k-means and raw-response reconstruction**

Fit `FeatureScaler(mean, scale)` on training features with `scale=max(std, 1e-8)`. K-means++ uses the supplied seed; 4 canonical restarts choose lowest inertia with lexicographic centroid bytes as deterministic tie-break. If a cluster empties, move the point with largest squared distance to its assigned centroid into the empty cluster; tie-break by lowest sample index.

If `k > number_of_distinct_standardized_rows`, raise `ValueError` rather than silently inventing duplicate clusters. Candidate selection catches this and marks that `k` invalid.

For an assignment, fit one **raw response** mean per cluster and reconstruct held-out response signatures from those means. Define:

```python
NRMSE = sqrt(mean((R - R_hat)**2)) / max(sqrt(mean(R**2)), 1e-12)
```

- [ ] **Step 5: Run coordinate tests and full existing suite**

Run: `pytest tests/test_coordinates.py -q && pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit response-coordinate machinery**

```bash
git add src/eaton/coordinates.py tests/test_coordinates.py
git commit -m "feat: recover local computation coordinates"
```

### Task 4: Temporal grammar, MDL-like selection, and attackers

**Files:**
- Create: `src/eaton/grammar.py`
- Create: `tests/test_grammar.py`

**Interfaces:**
- Consumes coordinate assignments and split-aligned symbol arrays.
- Produces `TransitionModel`, `WorldReadout`, `GaussianClusterModel`, `fit_transition_model`, `transition_nll`, `fit_world_readout`, `world_readout_metrics`, `fit_gaussian_cluster_model`, `gaussian_nll`, `mdl_score`, `time_bin_labels`, and `permute_labels`.

- [ ] **Step 1: Write failing tests for sequence boundaries, label permutation invariance, smoothing, and exact MDL arithmetic**

```python
import numpy as np
from eaton.grammar import fit_transition_model, transition_nll, fit_world_readout, world_readout_metrics


def test_transition_counts_do_not_cross_sequence_boundaries():
    z = np.array([[0, 1], [1, 0]])
    s = np.array([[0, 1], [2, 3]])
    model = fit_transition_model(z, s, k=2, alphabet_size=4, alpha=0.5)
    # only 0->1 under symbol 0 and 1->0 under symbol 2 are observed
    assert model.raw_counts.sum() == 2
    assert model.raw_counts[1, 1, 1] == 0


def test_metrics_are_invariant_to_coordinate_renaming():
    z_train = np.array([[0, 1, 2, 0], [2, 1, 0, 2]])
    z_test = z_train.copy()
    s = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    nxt = np.array([[1, 2, 3, 0], [2, 3, 0, 1]])
    a = fit_world_readout(z_train, s, nxt, k=3, alphabet_size=4, alpha=0.5)
    perm = np.array([2, 0, 1])
    b = fit_world_readout(perm[z_train], s, nxt, k=3, alphabet_size=4, alpha=0.5)
    ma = world_readout_metrics(a, z_test, s, nxt)
    mb = world_readout_metrics(b, perm[z_test], s, nxt)
    assert ma == mb
```

- [ ] **Step 2: Run and verify import failure**

Run: `pytest tests/test_grammar.py -q`
Expected: import failure for `eaton.grammar`.

- [ ] **Step 3: Implement event-conditioned coordinate grammar and world readout**

Transition counts have shape `(k, alphabet_size, k)` and include only `t=0..T-2` within each sequence:

```python
P(z_{t+1} | z_t, s_t)
```

World readout counts have shape `(k, alphabet_size, alphabet_size)` and use all prediction positions:

```python
P(s_{t+1} | z_t, s_t)
```

Both use additive smoothing exactly `alpha=0.5`. Return mean NLL in natural-log nats and accuracy for the world readout.

`time_bin_labels(n_sequences,T,k)` sets label `min(k-1, floor(t*k/T))`, repeated per sequence.

- [ ] **Step 4: Implement the frozen MDL-like score**

For each candidate clustering, standardize its clustering feature using the training scaler and fit a spherical Gaussian around each standardized centroid. Fit one shared variance:

```python
sigma2 = max(mean((X_std - centroid[label])**2), 1e-8)
```

Validation feature NLL is the full summed Gaussian NLL over all feature coordinates. Validation transition NLL is the **summed** negative log likelihood of coordinate transitions. Complexity is:

```python
gaussian_parameters = k * feature_dim + 1
transition_parameters = k * alphabet_size * (k - 1)
parameter_count = gaussian_parameters + transition_parameters
penalty = 0.5 * parameter_count * log(N_train_points)
score = validation_feature_nll_sum + validation_transition_nll_sum + penalty
```

`select_k` lives in Task 5 so that test data cannot enter this score.

- [ ] **Step 5: Run grammar tests and entire suite**

Run: `pytest tests/test_grammar.py -q && pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit grammar primitives**

```bash
git add src/eaton/grammar.py tests/test_grammar.py
git commit -m "feat: add temporal grammar and MDL metrics"
```

### Task 5: Frozen v1 experiment orchestration and preregistered classification

**Files:**
- Create: `src/eaton/v1_experiment.py`
- Create: `tests/test_v1_experiment.py`
- Modify: `src/eaton/__init__.py`

**Interfaces:**
- Consumes all v1 modules.
- Produces `V1Config`, `select_coordinate_model`, `run_v1_seed`, `aggregate_v1`, `classify_v1`, and `run_v1_canonical`.

- [ ] **Step 1: Write failing tests that freeze every canonical constant and classification boundary**

```python
from eaton.v1_experiment import V1Config, classify_v1


def test_v1_config_is_preregistered():
    c = V1Config()
    assert c.seed_count == 8
    assert c.world.sequence_length == 48
    assert c.world.structured_switch_probability == 0.05
    assert c.rnn.hidden_size == 16
    assert c.rnn.epochs == 50
    assert c.coordinates.probe_count == 8
    assert c.coordinates.probe_epsilon == 1e-5
    assert c.coordinates.k_candidates == tuple(range(1, 9))
    assert c.pass_thresholds["response_vs_activation_nrmse_ratio_median_max"] == 0.90


def test_training_failure_precedes_scientific_classification():
    summary = {"valid_structured_seed_count": 5}
    assert classify_v1(summary, V1Config()) == "TRAINING_FAILURE"
```

Add synthetic-summary tests immediately on both sides of every pass threshold, including exactly `0.90` ratio, exactly `0.10` shuffled grammar gain, exactly `0.05` time-bin gain, exactly `0.75` seed-win fraction, and exactly `1e-3` finite-difference error.

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest tests/test_v1_experiment.py -q`
Expected: import failure for `eaton.v1_experiment`.

- [ ] **Step 3: Implement split-safe candidate fitting and k selection**

For one trained network and one feature family (`response` or `activation`):

1. roll out pre-input states separately on train/validation/test;
2. construct features separately for each split;
3. fit scaler and k-means **only on train** for every `k=1..8`;
4. assign validation labels and compute validation feature Gaussian NLL + validation transition NLL + frozen penalty;
5. choose the finite candidate with smallest validation MDL-like score, tie-breaking by smaller `k`;
6. assign test labels only after `k` is chosen.

Add a test-only spy object or explicit counters proving `select_coordinate_model` accepts train+validation but has no test argument.

For activation coordinates, clustering features are raw `h_t`; after labels are chosen, fit mean **response signature** per activation cluster on train and use those means to compute test response NRMSE. This is the matched `what-it-does` versus `where-it-is` comparison.

- [ ] **Step 4: Implement all attackers with frozen semantics**

- `time-bin`: use response-selected `k` and deterministic equal-progress bins.
- `current-input`: coordinate is exactly `s_t` (`k=4`); use only for world-prediction baseline in addition to the separately fitted current-symbol table.
- `shuffled-response`: for each split independently, flatten response signatures, apply a seed-fixed permutation (`seed + 70_000 + split_offset`), reshape, then run the same response clustering/model-selection pipeline; this destroys temporal alignment without changing feature distribution.
- `volatile`: train a separate RNN with the same architecture/training seed on the volatile world and run the same response recovery pipeline.

Do not use hidden rule labels in any attacker or selection operation.

- [ ] **Step 5: Implement one-seed metrics and post-hoc diagnostics**

Each seed result must include at least:

```text
structured_rnn_accuracy
structured_current_symbol_accuracy
volatile_rnn_accuracy
response_selected_k
activation_selected_k
shuffled_selected_k
response_nrmse
activation_nrmse
response_transition_nll
shuffled_transition_nll
timebin_transition_nll
response_world_accuracy
activation_world_accuracy
current_symbol_world_accuracy
volatile_response_transition_nll
volatile_shuffled_transition_nll
finite_difference_relative_error
posthoc_rule_mutual_information
posthoc_best_rule_agreement
```

Post-hoc rule agreement may split one true rule across multiple recovered coordinates. Compute best-rule agreement by mapping each recovered coordinate to its majority hidden rule on **test only for reporting**; do not feed the mapping back into any metric. Mutual information is empirical test-set MI in nats.

- [ ] **Step 6: Implement aggregation and the exact classifications frozen above**

Aggregate with medians across valid structured seeds and report seed-win counts/fractions. `grammar_seed_win_fraction` requires response transition NLL lower than both shuffled and time-bin for that seed.

Volatile specificity is:

```python
structured_gain = shuffled_transition_nll - response_transition_nll
volatile_gain = volatile_shuffled_transition_nll - volatile_response_transition_nll
specificity = structured_gain - volatile_gain
```

Use the exact classification order and thresholds in this plan header.

- [ ] **Step 7: Export only stable v1 entry points without changing v0 names**

Append aliases to `src/eaton/__init__.py`:

```python
from .v1_experiment import V1Config, run_v1_canonical
__all__ += ["V1Config", "run_v1_canonical"]
```

Do not rename or shadow v0 `run_canonical`, `run_seed`, `classify`, or `FrozenConfig`.

- [ ] **Step 8: Run focused orchestration tests and full regression suite**

Run: `pytest tests/test_v1_experiment.py -q && pytest -q`
Expected: all pass before any canonical receipt exists.

- [ ] **Step 9: Commit the frozen experiment harness**

```bash
git add src/eaton/v1_experiment.py src/eaton/__init__.py tests/test_v1_experiment.py
git commit -m "feat: freeze EATON v1 computation-coordinate gate"
```

### Task 6: Canonical runner, receipt, and destructive verification

**Files:**
- Create: `experiments/run_v1.py`
- Create after successful run: `results/v1_coordinates.json`
- Create: `tests/test_v1_receipt.py`

**Interfaces:**
- `python experiments/run_v1.py --out results/v1_coordinates.json` calls `run_v1_canonical()` and serializes config, all seed results, summary, and classification with stable JSON ordering/indentation.

- [ ] **Step 1: Write a failing runner/receipt regression test before generating the receipt**

```python
import json
from pathlib import Path
from eaton.v1_experiment import V1Config, run_v1_canonical


def test_committed_v1_receipt_matches_fresh_canonical_run():
    expected = json.loads(Path("results/v1_coordinates.json").read_text())
    fresh = run_v1_canonical(V1Config())
    assert fresh == expected
```

The initial failure must be `FileNotFoundError` for the receipt, not a skipped test.

- [ ] **Step 2: Implement `experiments/run_v1.py`**

Use `argparse` with one optional `--out` defaulting to `results/v1_coordinates.json`; create parent directories; serialize via `json.dumps(result, indent=2, sort_keys=True) + "\n"`; print the summary classification and principal medians to stdout.

- [ ] **Step 3: Run all tests except the expected missing-receipt test**

Run: `pytest -q --ignore=tests/test_v1_receipt.py`
Expected: all pass.

- [ ] **Step 4: Generate the first canonical receipt exactly once with frozen code/config**

Run: `python experiments/run_v1.py --out results/v1_coordinates.json`
Expected: completes with one of `PASS_COMPUTATION_COORDINATES`, `PASS_STATE_NOT_OPERATOR`, `FAIL_GRAMMAR_RECOVERY`, or `TRAINING_FAILURE`; no threshold/config edits are permitted after seeing this output.

- [ ] **Step 5: Run the committed-receipt regression plus the full suite**

Run: `pytest tests/test_v1_receipt.py -q && pytest -q`
Expected: receipt regression passes exactly and all v0/v1 tests pass.

- [ ] **Step 6: Perform destructive checks without changing canonical configuration**

Run a separate noncanonical diagnostic script or direct Python snippets that confirm:

```text
1. permuting response signatures in time worsens grammar predictability relative to canonical response coordinates when the canonical result claims such a gain;
2. replacing response labels by time bins does not accidentally receive cross-sequence transitions;
3. using analytic Jacobians instead of finite differences gives the same response geometry to <=1e-3 relative error but does not change the committed receipt;
4. hidden rule labels never occur in source lines executed by select_coordinate_model before post-hoc diagnostics.
```

Record these checks in the PR body, not the canonical JSON.

- [ ] **Step 7: Commit runner and frozen evidence**

```bash
git add experiments/run_v1.py results/v1_coordinates.json tests/test_v1_receipt.py
git commit -m "test: freeze EATON v1 canonical receipt"
```

### Task 7: README interpretation, claim boundary, CI verification, and PR

**Files:**
- Modify: `README.md`
- Modify only if needed for existing CI discovery: `.github/workflows/ci.yml`

**Interfaces:**
- Documentation must report the actual frozen classification and numbers from `results/v1_coordinates.json`; it must not upgrade a negative/mixed result into the intended theory.

- [ ] **Step 1: Add a README section describing v1 before inserting results**

Document exactly these distinctions:

```text
state coordinate       = where the recurrent system is
computation coordinate = how the local dynamics transform matched perturbations here
temporal grammar       = event-conditioned transition law among recovered coordinates
```

Explain structured versus volatile worlds, black-box restrictions, response-versus-activation attacker, and the MDL-like `k=1..8` selection. Keep Dong et al. as a methodological analogy only: depth-wise path decomposition there versus time-wise response-coordinate decomposition here.

- [ ] **Step 2: Insert only the measured canonical result and its allowed interpretation**

If classification is `PASS_COMPUTATION_COORDINATES`, state only the narrow claim from the spec. If `PASS_STATE_NOT_OPERATOR`, say activation/state structure survived but operator-coordinate superiority did not. If `FAIL_GRAMMAR_RECOVERY`, say the learned RNN solved the task but the proposed low-complexity operator grammar did not survive attackers. If `TRAINING_FAILURE`, say the experiment is invalid for the scientific question until training is independently repaired under a new preregistration.

Always report selected-k distribution, response/activation NRMSE medians, grammar gains, collapsed world accuracy, volatile specificity, and finite-difference validity.

- [ ] **Step 3: Verify CI already runs all pytest files**

Inspect `.github/workflows/ci.yml`. Change it only if it names individual v0 tests; if it already runs `pytest`/`pytest -q`, leave it untouched.

- [ ] **Step 4: Run final verification on both supported Python versions where available**

Local commands:

```bash
python3.11 -m pytest -q
python3.12 -m pytest -q
python3.11 experiments/run_v1.py --out /tmp/eaton-v1-311.json
python3.12 experiments/run_v1.py --out /tmp/eaton-v1-312.json
cmp /tmp/eaton-v1-311.json /tmp/eaton-v1-312.json
```

If only one interpreter exists locally, require GitHub Actions to supply the second before merge and compare both workflow artifacts or rerun outputs; do not claim cross-version identity without evidence.

- [ ] **Step 5: Commit documentation and open the PR**

```bash
git add README.md .github/workflows/ci.yml
git commit -m "docs: record EATON v1 computation-coordinate result"
```

Open a PR from `eaton-v1-computation-coordinates` to `main`. The body must list: frozen design/spec commit, plan commit, classification, canonical metrics, all attacker outcomes, receipt reproducibility, v0 regression status, and explicit claim boundary.

## Self-review performed before execution

- **Spec coverage:** every spec section maps to a task: structured/volatile worlds (Task 1), trainable black-box RNN (Task 2), response probes and analytic validation (Task 3), grammar/MDL (Task 4), attackers/classification/post-hoc rule alignment (Task 5), canonical frozen evidence (Task 6), and interpretation/claim boundary (Task 7).
- **Placeholder scan:** no `TBD`, `TODO`, "similar to", or unspecified implementation/error-handling steps remain.
- **Type consistency:** `WorldBatch -> RNNParams/states -> response signatures/labels -> grammar metrics -> V1 seed result` is consistent across task interfaces; v0 names remain untouched.
- **Review Focus:** all five identified failure modes have explicit tests in Tasks 1, 3, 4, or 5.
- **Preregistration check:** all scientific thresholds and canonical hyperparameters are fixed in this plan before implementation/canonical results.
