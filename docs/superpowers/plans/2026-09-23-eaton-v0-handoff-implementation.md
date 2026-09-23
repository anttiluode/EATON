# EATON v0 Transient Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze EATON's first falsifiable mechanism: a resident-state machine where releasing an acquisition operator after its useful boundary preserves a later computation better than tonic persistence, while a reset attacker proves the handoff depends on resident state.

**Architecture:** `core.py` implements the exact three-tick resident dynamics and operator lifetimes from the frozen spec. `world.py` samples each episode once and replays that identical tape across arms. `experiment.py` owns matched-arm invariants, seed metrics, aggregation, and classification. A single runner writes the canonical receipt incrementally; tests protect both mechanism semantics and the frozen scientific result.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-23-eaton-v0-handoff-design.md`

## Global Constraints

- NumPy is the only mechanism/runtime dependency; pytest is the test dependency.
- Python versions are exactly 3.11 and 3.12 in CI.
- Canonical seeds are integers `0..63`.
- Each seed has exactly 256 episodes: 64 repetitions of each pair in `{(-1,-1), (-1,+1), (+1,-1), (+1,+1)}`.
- Each episode samples `x0[j] ~ Normal(0, 0.05)`, `eta_context ~ Normal(0, 0.08)`, and `eta_payload ~ Normal(0, 0.08)` exactly once; all arms replay those same values.
- Delivered events are `c + eta_context` and `p + eta_payload`; do not clip them.
- Passive retention is `0.98`; `decision_trace` freezes after stage 1.
- A writes `memory <- tanh(0.40*memory + 1.00*u)` and `decision_trace <- memory`.
- B payload step writes `payload_trace <- tanh(u)`; B compute step writes `output <- tanh(2.00*memory*payload_trace)`.
- Schedule is exactly `t0=context/A`, handoff, `t1=payload/B (+A only tonic)`, `t2=no-event/B compute`.
- `transient` and `tonic` differ only in A eligibility at `t1`.
- `reset` differs from `transient` only by restoring `memory` to the episode's original `x0[memory]` at the handoff boundary.
- Canonical receipt is `results/v0_handoff.json`.
- No v1-v3 implementation belongs in this plan.
- Frozen coefficients, noise, seeds, episode counts, gains, retention, and pass thresholds must not change after the canonical result is observed.

## Review Focus

1. **Noise sign reversals:** a nuisance draw can reverse an event sign and must count naturally as an error; no clipping or hidden repair.
2. **Payload-tick operator order:** tonic A and B write disjoint coordinates, so `(A,B)` and `(B,A)` execution orders must be identical.
3. **Reset isolation:** reset changes only `memory` at the handoff and restores exactly `x0[memory]`.
4. **Matched tape replay:** no arm-specific RNG calls; labels, initial state, and nuisance values are identical across arms.
5. **Classification precedence:** matched-arm failures produce `INVALID_MATCHED_ARMS`; numerical instability/weak transient is evaluated later and produces `UNSTABLE_OR_WEAK_TRANSIENT` exactly as the spec says.

---

### Task 1: Package scaffold and exact three-tick core

**Files:**
- Create: `pyproject.toml`
- Create: `src/eaton/__init__.py`
- Create: `src/eaton/core.py`
- Create: `tests/test_core.py`

**Interfaces:**
- Produces `FrozenConfig`, `EpisodeInput`, `EpisodeResult`.
- Produces `run_episode(episode, arm, config, *, disable_a=False, disable_b=False, clamp_memory_to_context=False, operator_order=("A","B")) -> EpisodeResult`.

- [ ] **Step 1: Write the package scaffold and first failing config test**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eaton"
version = "0.0.1"
description = "Event-Addressed Transient Operator Networks"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
test = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`src/eaton/__init__.py` initially:

```python
from .core import EpisodeInput, EpisodeResult, FrozenConfig, run_episode

__all__ = ["EpisodeInput", "EpisodeResult", "FrozenConfig", "run_episode"]
```

`tests/test_core.py` begins with:

```python
from eaton.core import FrozenConfig


def test_frozen_config_matches_spec():
    cfg = FrozenConfig()
    assert cfg.retention == 0.98
    assert cfg.memory_self == 0.40
    assert cfg.acquire_gain == 1.00
    assert cfg.output_gain == 2.00
    assert cfg.init_sigma == 0.05
    assert cfg.event_noise_sigma == 0.08
    assert cfg.seed_count == 64
    assert cfg.episodes_per_seed == 256
    assert cfg.repetitions_per_pair == 64
```

- [ ] **Step 2: Run the test and verify RED**

```bash
python -m pip install -e '.[test]'
pytest tests/test_core.py::test_frozen_config_matches_spec -q
```

Expected: FAIL because `eaton.core` does not exist.

- [ ] **Step 3: Implement the frozen types**

`src/eaton/core.py` starts with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Arm = Literal["transient", "tonic", "reset"]
MEMORY, DECISION, PAYLOAD, OUTPUT = range(4)
STATE_DIM = 4


@dataclass(frozen=True)
class FrozenConfig:
    retention: float = 0.98
    memory_self: float = 0.40
    acquire_gain: float = 1.00
    output_gain: float = 2.00
    init_sigma: float = 0.05
    event_noise_sigma: float = 0.08
    seed_count: int = 64
    episodes_per_seed: int = 256
    repetitions_per_pair: int = 64
    state_bound: float = 1.000000000001


@dataclass(frozen=True)
class EpisodeInput:
    context: int
    payload: int
    x0: tuple[float, float, float, float]
    eta_context: float
    eta_payload: float

    @property
    def context_event(self) -> float:
        return float(self.context + self.eta_context)

    @property
    def payload_event(self) -> float:
        return float(self.payload + self.eta_payload)


@dataclass(frozen=True)
class EpisodeResult:
    arm: str
    early_decision: int
    final_answer: int
    early_state: tuple[float, float, float, float]
    handoff_state: tuple[float, float, float, float]
    precompute_state: tuple[float, float, float, float]
    final_state: tuple[float, float, float, float]
    memory_before_compute: float
    overlap_event_ticks: int
    event_count: int
    event_abs_sum: float
    max_abs_state: float
    early_correct: bool
    final_correct: bool


def _sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def _state_tuple(x: np.ndarray) -> tuple[float, float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
```

- [ ] **Step 4: Run the config test and verify GREEN**

```bash
pytest tests/test_core.py::test_frozen_config_matches_spec -q
```

Expected: PASS.

- [ ] **Step 5: Add failing tests for stage matching, reset isolation, order independence, noise reversal, and event budgets**

Append to `tests/test_core.py`:

```python
import math
import numpy as np

from eaton.core import EpisodeInput, run_episode


def episode(context=1, payload=-1, eta_context=0.0, eta_payload=0.0):
    return EpisodeInput(
        context=context,
        payload=payload,
        x0=(0.01, -0.02, 0.03, -0.04),
        eta_context=eta_context,
        eta_payload=eta_payload,
    )


def test_all_arms_are_identical_through_stage1():
    ep = episode()
    rs = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    assert rs[0].early_state == rs[1].early_state == rs[2].early_state
    assert rs[0].early_decision == rs[1].early_decision == rs[2].early_decision == 1


def test_reset_changes_only_memory_at_handoff():
    ep = episode()
    t = run_episode(ep, "transient", FrozenConfig())
    r = run_episode(ep, "reset", FrozenConfig())
    diffs = [i for i, (a, b) in enumerate(zip(t.handoff_state, r.handoff_state)) if a != b]
    assert diffs == [0]
    assert r.handoff_state[0] == ep.x0[0]


def test_tonic_payload_tick_is_order_independent():
    ep = episode(context=-1, payload=1)
    ab = run_episode(ep, "tonic", FrozenConfig(), operator_order=("A", "B"))
    ba = run_episode(ep, "tonic", FrozenConfig(), operator_order=("B", "A"))
    assert ab == ba


def test_external_event_budget_matches_all_arms():
    ep = episode(eta_context=0.1, eta_payload=-0.2)
    rs = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    assert {r.event_count for r in rs} == {2}
    expected = abs(1.1) + abs(-1.2)
    assert all(math.isclose(r.event_abs_sum, expected, abs_tol=1e-15) for r in rs)


def test_noise_can_reverse_context_without_clipping():
    ep = episode(context=1, payload=1, eta_context=-1.5)
    r = run_episode(ep, "transient", FrozenConfig())
    assert ep.context_event == -0.5
    assert r.early_decision == -1
    assert r.early_correct is False


def test_large_events_still_produce_finite_bounded_written_state():
    ep = EpisodeInput(1, -1, (0.04, -0.04, 0.04, -0.04), 1000.0, -1000.0)
    r = run_episode(ep, "tonic", FrozenConfig())
    assert np.isfinite(r.max_abs_state)
    assert r.max_abs_state <= FrozenConfig().state_bound
```

- [ ] **Step 6: Run core tests and verify RED**

```bash
pytest tests/test_core.py -q
```

Expected: FAIL because `run_episode` is missing.

- [ ] **Step 7: Implement passive retention, disjoint payload writes, and `run_episode`**

Add to `src/eaton/core.py`:

```python
def _passive(x: np.ndarray, written: set[int], cfg: FrozenConfig, *, freeze_decision: bool) -> None:
    for i in range(STATE_DIM):
        if i in written or (freeze_decision and i == DECISION):
            continue
        x[i] *= cfg.retention


def _payload_tick(
    x: np.ndarray,
    event: float,
    *,
    a_active: bool,
    b_active: bool,
    cfg: FrozenConfig,
    operator_order: tuple[str, str],
) -> None:
    if sorted(operator_order) != ["A", "B"]:
        raise ValueError("operator_order must contain A and B exactly once")
    pre = x.copy()
    a_memory = None
    b_payload = None
    for op in operator_order:
        if op == "A" and a_active:
            a_memory = float(np.tanh(cfg.memory_self * pre[MEMORY] + cfg.acquire_gain * event))
        elif op == "B" and b_active:
            b_payload = float(np.tanh(event))
    written = set()
    if a_memory is not None:
        written.add(MEMORY)
    if b_payload is not None:
        written.add(PAYLOAD)
    _passive(x, written, cfg, freeze_decision=True)
    if a_memory is not None:
        x[MEMORY] = a_memory
    if b_payload is not None:
        x[PAYLOAD] = b_payload


def run_episode(
    episode: EpisodeInput,
    arm: str,
    config: FrozenConfig,
    *,
    disable_a: bool = False,
    disable_b: bool = False,
    clamp_memory_to_context: bool = False,
    operator_order: tuple[str, str] = ("A", "B"),
) -> EpisodeResult:
    if arm not in {"transient", "tonic", "reset"}:
        raise ValueError(f"unknown arm {arm!r}")

    x = np.asarray(episode.x0, dtype=np.float64).copy()
    snapshots = [x.copy()]

    # t0: context tick
    if disable_a:
        _passive(x, set(), config, freeze_decision=False)
    else:
        x[MEMORY] = np.tanh(config.memory_self * x[MEMORY] + config.acquire_gain * episode.context_event)
        x[DECISION] = x[MEMORY]
        _passive(x, {MEMORY, DECISION}, config, freeze_decision=False)
    early_state = x.copy()
    early_decision = _sign(float(x[DECISION]))
    snapshots.append(x.copy())

    # handoff boundary
    if arm == "reset":
        x[MEMORY] = float(episode.x0[MEMORY])
    handoff_state = x.copy()
    snapshots.append(x.copy())

    # t1: payload tick
    _payload_tick(
        x,
        episode.payload_event,
        a_active=(arm == "tonic" and not disable_a),
        b_active=(not disable_b),
        cfg=config,
        operator_order=operator_order,
    )
    precompute_state = x.copy()
    snapshots.append(x.copy())

    # t2: no event, B computes
    if clamp_memory_to_context:
        x[MEMORY] = float(episode.context)
    written: set[int] = set()
    if not disable_b:
        x[OUTPUT] = np.tanh(config.output_gain * x[MEMORY] * x[PAYLOAD])
        written.add(OUTPUT)
    _passive(x, written, config, freeze_decision=True)
    snapshots.append(x.copy())

    final_answer = _sign(float(x[OUTPUT]))
    max_abs_state = max(float(np.max(np.abs(s))) for s in snapshots)
    return EpisodeResult(
        arm=arm,
        early_decision=early_decision,
        final_answer=final_answer,
        early_state=_state_tuple(early_state),
        handoff_state=_state_tuple(handoff_state),
        precompute_state=_state_tuple(precompute_state),
        final_state=_state_tuple(x),
        memory_before_compute=float(precompute_state[MEMORY]),
        overlap_event_ticks=1 if arm == "tonic" and not disable_a and not disable_b else 0,
        event_count=2,
        event_abs_sum=abs(episode.context_event) + abs(episode.payload_event),
        max_abs_state=max_abs_state,
        early_correct=(early_decision == episode.context),
        final_correct=(final_answer == episode.context * episode.payload),
    )
```

- [ ] **Step 8: Add destructive-control tests**

Append to `tests/test_core.py`:

```python
def exact_balanced_tape():
    pairs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] * 64
    return [EpisodeInput(c, p, (0.01, -0.02, 0.03, -0.04), 0.0, 0.0) for c, p in pairs]


def test_disabling_a_leaves_early_decision_at_chance_on_balanced_tape():
    rs = [run_episode(ep, "transient", FrozenConfig(), disable_a=True) for ep in exact_balanced_tape()]
    assert sum(r.early_correct for r in rs) / len(rs) == 0.5


def test_disabling_b_leaves_final_parity_at_chance_on_balanced_tape():
    rs = [run_episode(ep, "transient", FrozenConfig(), disable_b=True) for ep in exact_balanced_tape()]
    assert sum(r.final_correct for r in rs) / len(rs) == 0.5


def test_correct_memory_clamp_allows_b_to_solve():
    rs = [
        run_episode(ep, "reset", FrozenConfig(), clamp_memory_to_context=True)
        for ep in exact_balanced_tape()
    ]
    assert all(r.final_correct for r in rs)
```

- [ ] **Step 9: Run Task 1 tests and commit**

```bash
pytest tests/test_core.py -q
git add pyproject.toml src/eaton/__init__.py src/eaton/core.py tests/test_core.py
git commit -m "feat: add EATON transient handoff core"
```

Expected: tests PASS before commit.

---

### Task 2: Deterministic matched world tape

**Files:**
- Create: `src/eaton/world.py`
- Create: `tests/test_world.py`
- Modify: `src/eaton/__init__.py`

**Interfaces:**
- Consumes `FrozenConfig`, `EpisodeInput`.
- Produces `build_seed_tape(seed: int, config: FrozenConfig) -> tuple[EpisodeInput, ...]`.

- [ ] **Step 1: Write failing balance and determinism tests**

`tests/test_world.py`:

```python
from collections import Counter
from dataclasses import asdict

from eaton.core import FrozenConfig
from eaton.world import build_seed_tape


def test_seed_tape_has_exact_balance():
    tape = build_seed_tape(0, FrozenConfig())
    assert len(tape) == 256
    assert Counter((e.context, e.payload) for e in tape) == {
        (-1, -1): 64,
        (-1, 1): 64,
        (1, -1): 64,
        (1, 1): 64,
    }


def test_seed_tape_is_repeatable_and_seed_specific():
    cfg = FrozenConfig()
    a = [asdict(e) for e in build_seed_tape(17, cfg)]
    b = [asdict(e) for e in build_seed_tape(17, cfg)]
    c = [asdict(e) for e in build_seed_tape(18, cfg)]
    assert a == b
    assert a != c
```

- [ ] **Step 2: Run and verify RED**

```bash
pytest tests/test_world.py -q
```

Expected: FAIL because `eaton.world` does not exist.

- [ ] **Step 3: Implement one-RNG-per-seed tape generation**

`src/eaton/world.py`:

```python
from __future__ import annotations

import numpy as np

from .core import EpisodeInput, FrozenConfig, STATE_DIM

PAIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def build_seed_tape(seed: int, config: FrozenConfig) -> tuple[EpisodeInput, ...]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    labels = [pair for pair in PAIRS for _ in range(config.repetitions_per_pair)]
    if len(labels) != config.episodes_per_seed:
        raise RuntimeError("frozen label budget does not equal episodes_per_seed")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    tape = []
    for index in order:
        context, payload = labels[int(index)]
        x0_values = rng.normal(0.0, config.init_sigma, size=STATE_DIM)
        tape.append(
            EpisodeInput(
                context=context,
                payload=payload,
                x0=tuple(float(v) for v in x0_values),  # type: ignore[arg-type]
                eta_context=float(rng.normal(0.0, config.event_noise_sigma)),
                eta_payload=float(rng.normal(0.0, config.event_noise_sigma)),
            )
        )
    return tuple(tape)
```

Export `build_seed_tape` from `src/eaton/__init__.py`.

- [ ] **Step 4: Add a replay-through-arms test proving no arm-specific RNG**

Append to `tests/test_world.py`:

```python
from eaton.core import run_episode


def test_one_episode_object_is_replayed_across_all_arms():
    ep = build_seed_tape(3, FrozenConfig())[0]
    rs = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    expected_sum = abs(ep.context_event) + abs(ep.payload_event)
    assert all(r.event_count == 2 for r in rs)
    assert all(r.event_abs_sum == expected_sum for r in rs)
```

- [ ] **Step 5: Run and commit**

```bash
pytest tests/test_core.py tests/test_world.py -q
git add src/eaton/__init__.py src/eaton/world.py tests/test_world.py
git commit -m "feat: add frozen EATON world tape"
```

Expected: PASS.

---

### Task 3: Matched-arm metrics, invariant separation, and classifier

**Files:**
- Create: `src/eaton/experiment.py`
- Create: `tests/test_experiment.py`
- Modify: `src/eaton/__init__.py`

**Interfaces:**
- Produces `run_seed(seed, config) -> dict`, `aggregate(seed_results, config) -> dict`, `classify(summary, config) -> str`, and `run_canonical(config=None) -> dict`.
- `summary["matched_invariants_valid"]` covers only stage-1 equality and external-budget matching; numerical finiteness/boundedness stays separate so classification precedence matches the spec.

- [ ] **Step 1: Write failing seed-level tests**

`tests/test_experiment.py`:

```python
from eaton.core import FrozenConfig
from eaton.experiment import run_seed


def test_seed_keeps_matched_arm_invariants_separate_from_stability():
    r = run_seed(0, FrozenConfig())
    assert r["matched_invariants"]["stage1_transient_tonic_equal"] is True
    assert r["matched_invariants"]["stage1_transient_reset_equal"] is True
    assert r["matched_invariants"]["event_counts_equal"] is True
    assert r["matched_invariants"]["event_amplitude_sums_equal"] is True
    assert isinstance(r["numerical_checks"]["all_finite"], bool)
    assert isinstance(r["numerical_checks"]["all_bounded"], bool)


def test_seed_records_all_paired_episode_outcomes():
    r = run_seed(0, FrozenConfig())
    for key in ("transient_vs_tonic", "transient_vs_reset"):
        counts = r["paired_episode_disagreements"][key]
        assert sum(counts.values()) == 256
        assert set(counts) == {
            "transient_only_correct",
            "attacker_only_correct",
            "both_correct",
            "both_wrong",
        }
```

- [ ] **Step 2: Run and verify RED**

```bash
pytest tests/test_experiment.py -q
```

Expected: FAIL because `eaton.experiment` does not exist.

- [ ] **Step 3: Implement `run_seed` and arm metrics**

`src/eaton/experiment.py` begins:

```python
from __future__ import annotations

from statistics import median
import numpy as np

from .core import FrozenConfig, run_episode
from .world import build_seed_tape

ARMS = ("transient", "tonic", "reset")


def _corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _metrics(results, tape) -> dict:
    return {
        "early_accuracy": float(np.mean([r.early_correct for r in results])),
        "final_accuracy": float(np.mean([r.final_correct for r in results])),
        "handoff_memory_correlation": _corr(
            [r.memory_before_compute for r in results],
            [e.context for e in tape],
        ),
        "operator_overlap_event_ticks": int(sum(r.overlap_event_ticks for r in results)),
        "event_count": int(sum(r.event_count for r in results)),
        "event_abs_sum": float(sum(r.event_abs_sum for r in results)),
        "max_abs_state": float(max(r.max_abs_state for r in results)),
    }


def _paired_counts(transient, attacker) -> dict:
    out = {
        "transient_only_correct": 0,
        "attacker_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for t, a in zip(transient, attacker):
        if t.final_correct and not a.final_correct:
            out["transient_only_correct"] += 1
        elif a.final_correct and not t.final_correct:
            out["attacker_only_correct"] += 1
        elif t.final_correct:
            out["both_correct"] += 1
        else:
            out["both_wrong"] += 1
    return out


def run_seed(seed: int, config: FrozenConfig) -> dict:
    tape = build_seed_tape(seed, config)
    raw = {arm: [run_episode(ep, arm, config) for ep in tape] for arm in ARMS}
    metrics = {arm: _metrics(raw[arm], tape) for arm in ARMS}

    sums = [metrics[arm]["event_abs_sum"] for arm in ARMS]
    matched = {
        "stage1_transient_tonic_equal": all(
            a.early_state == b.early_state and a.early_decision == b.early_decision
            for a, b in zip(raw["transient"], raw["tonic"])
        ),
        "stage1_transient_reset_equal": all(
            a.early_state == b.early_state and a.early_decision == b.early_decision
            for a, b in zip(raw["transient"], raw["reset"])
        ),
        "event_counts_equal": len({metrics[a]["event_count"] for a in ARMS}) == 1,
        "event_amplitude_sums_equal": max(sums) - min(sums) <= 1e-12,
    }
    numerical = {
        "all_finite": all(np.isfinite(metrics[a]["max_abs_state"]) for a in ARMS),
        "all_bounded": all(metrics[a]["max_abs_state"] <= config.state_bound for a in ARMS),
    }
    return {
        "seed": seed,
        "arms": metrics,
        "matched_invariants": matched,
        "numerical_checks": numerical,
        "paired_episode_disagreements": {
            "transient_vs_tonic": _paired_counts(raw["transient"], raw["tonic"]),
            "transient_vs_reset": _paired_counts(raw["transient"], raw["reset"]),
        },
    }
```

- [ ] **Step 4: Run seed tests and verify GREEN**

```bash
pytest tests/test_experiment.py -q
```

Expected: current tests PASS.

- [ ] **Step 5: Add failing classifier-precedence tests**

Append to `tests/test_experiment.py`:

```python
from eaton.experiment import classify


def passing_summary():
    return {
        "matched_invariants_valid": True,
        "median_early_accuracy": {"transient": 1.0, "tonic": 1.0, "reset": 1.0},
        "median_final_accuracy": {"transient": 0.98, "tonic": 0.50, "reset": 0.50},
        "median_transient_minus_tonic": 0.48,
        "median_transient_minus_reset": 0.48,
        "transient_seed_wins_vs_tonic": 64,
        "transient_seed_wins_vs_reset": 64,
        "all_finite": True,
        "max_abs_state": 0.99,
    }


def test_classifier_matched_failure_has_highest_precedence():
    s = passing_summary()
    s["matched_invariants_valid"] = False
    s["max_abs_state"] = 99.0
    assert classify(s, FrozenConfig()) == "INVALID_MATCHED_ARMS"


def test_classifier_tonic_failure_precedes_reset_failure():
    s = passing_summary()
    s["median_transient_minus_tonic"] = 0.19
    s["median_transient_minus_reset"] = 0.0
    assert classify(s, FrozenConfig()) == "NO_HANDOFF_ADVANTAGE"


def test_classifier_reset_failure():
    s = passing_summary()
    s["median_transient_minus_reset"] = 0.29
    assert classify(s, FrozenConfig()) == "HANDOFF_WITHOUT_RESIDENT_MEMORY"


def test_classifier_instability_is_not_misreported_as_matched_failure():
    s = passing_summary()
    s["max_abs_state"] = 1.1
    assert classify(s, FrozenConfig()) == "UNSTABLE_OR_WEAK_TRANSIENT"


def test_classifier_pass():
    assert classify(passing_summary(), FrozenConfig()) == "PASS_TRANSIENT_HANDOFF"
```

- [ ] **Step 6: Implement aggregate and classifier exactly in frozen precedence order**

Append to `src/eaton/experiment.py`:

```python
def classify(summary: dict, config: FrozenConfig) -> str:
    if not summary["matched_invariants_valid"]:
        return "INVALID_MATCHED_ARMS"
    if (
        summary["median_transient_minus_tonic"] < 0.20
        or summary["transient_seed_wins_vs_tonic"] < 48
    ):
        return "NO_HANDOFF_ADVANTAGE"
    if (
        summary["median_transient_minus_reset"] < 0.30
        or summary["transient_seed_wins_vs_reset"] < 56
    ):
        return "HANDOFF_WITHOUT_RESIDENT_MEMORY"
    if (
        min(summary["median_early_accuracy"].values()) < 0.95
        or summary["median_final_accuracy"]["transient"] < 0.90
        or not summary["all_finite"]
        or summary["max_abs_state"] > config.state_bound
    ):
        return "UNSTABLE_OR_WEAK_TRANSIENT"
    return "PASS_TRANSIENT_HANDOFF"


def aggregate(seed_results: list[dict], config: FrozenConfig) -> dict:
    early = {
        arm: float(median(r["arms"][arm]["early_accuracy"] for r in seed_results))
        for arm in ARMS
    }
    final = {
        arm: float(median(r["arms"][arm]["final_accuracy"] for r in seed_results))
        for arm in ARMS
    }
    memory_corr = {
        arm: float(median(r["arms"][arm]["handoff_memory_correlation"] for r in seed_results))
        for arm in ARMS
    }
    dt = [
        r["arms"]["transient"]["final_accuracy"] - r["arms"]["tonic"]["final_accuracy"]
        for r in seed_results
    ]
    dr = [
        r["arms"]["transient"]["final_accuracy"] - r["arms"]["reset"]["final_accuracy"]
        for r in seed_results
    ]
    summary = {
        "matched_invariants_valid": all(
            all(r["matched_invariants"].values()) for r in seed_results
        ),
        "median_early_accuracy": early,
        "median_final_accuracy": final,
        "median_handoff_memory_correlation": memory_corr,
        "median_transient_minus_tonic": float(median(dt)),
        "median_transient_minus_reset": float(median(dr)),
        "transient_seed_wins_vs_tonic": int(sum(x > 0.0 for x in dt)),
        "transient_seed_wins_vs_reset": int(sum(x > 0.0 for x in dr)),
        "all_finite": all(r["numerical_checks"]["all_finite"] for r in seed_results),
        "max_abs_state": float(max(r["arms"][a]["max_abs_state"] for r in seed_results for a in ARMS)),
    }
    summary["classification"] = classify(summary, config)
    return summary


def run_canonical(config: FrozenConfig | None = None) -> dict:
    cfg = FrozenConfig() if config is None else config
    seeds = [run_seed(seed, cfg) for seed in range(cfg.seed_count)]
    return {"config": cfg.__dict__, "seed_results": seeds, "summary": aggregate(seeds, cfg)}
```

Export `run_seed`, `aggregate`, `classify`, and `run_canonical` from `src/eaton/__init__.py`.

- [ ] **Step 7: Run Task 3 tests and commit**

```bash
pytest tests/test_experiment.py tests/test_world.py tests/test_core.py -q
git add src/eaton/__init__.py src/eaton/experiment.py tests/test_experiment.py
git commit -m "feat: add EATON matched-arm classifier"
```

Expected: PASS.

---

### Task 4: Canonical runner, immutable receipt, and receipt regression

**Files:**
- Create: `experiments/run_v0.py`
- Create: `results/v0_handoff.json`
- Create: `tests/test_v0_gate.py`

**Interfaces:**
- Consumes `run_seed`, `aggregate`, `FrozenConfig`.
- Produces an incrementally written receipt with frozen config, all 64 seed results, summary, classification, and claim boundary.

- [ ] **Step 1: Write failing receipt test**

`tests/test_v0_gate.py`:

```python
import json
import math
from pathlib import Path

from eaton.experiment import run_canonical

RECEIPT = Path("results/v0_handoff.json")


def test_committed_receipt_matches_fresh_scientific_summary():
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    fresh = run_canonical()
    cs = committed["summary"]
    fs = fresh["summary"]
    assert committed["config"] == fresh["config"]
    assert committed["classification"] == cs["classification"] == fs["classification"]
    for key in (
        "median_early_accuracy",
        "median_final_accuracy",
        "median_transient_minus_tonic",
        "median_transient_minus_reset",
        "transient_seed_wins_vs_tonic",
        "transient_seed_wins_vs_reset",
        "matched_invariants_valid",
        "all_finite",
    ):
        assert cs[key] == fs[key]
    assert math.isclose(cs["max_abs_state"], fs["max_abs_state"], abs_tol=1e-12)
    for arm in ("transient", "tonic", "reset"):
        assert math.isclose(
            cs["median_handoff_memory_correlation"][arm],
            fs["median_handoff_memory_correlation"][arm],
            abs_tol=1e-12,
        )
```

- [ ] **Step 2: Run and verify RED**

```bash
pytest tests/test_v0_gate.py -q
```

Expected: FAIL because `results/v0_handoff.json` does not exist.

- [ ] **Step 3: Implement incremental runner**

`experiments/run_v0.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eaton.core import FrozenConfig
from eaton.experiment import aggregate, run_seed

CLAIM_BOUNDARY = (
    "Synthetic existence proof only: v0 can establish operator lifetime as a "
    "computational variable in this matched resident-state machine. It does not "
    "establish a biological mechanism, general superiority of phasic control, "
    "or equivalence to transformer attention."
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v0_handoff.json")
    args = parser.parse_args()
    out = Path(args.out)
    cfg = FrozenConfig()
    payload = {
        "experiment": "EATON v0 transient operator handoff",
        "config": cfg.__dict__,
        "frozen_parameters": dict(cfg.__dict__),
        "seed_results": [],
        "summary": None,
        "classification": "INCOMPLETE",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write(out, payload)
    for seed in range(cfg.seed_count):
        payload["seed_results"].append(run_seed(seed, cfg))
        write(out, payload)
    payload["summary"] = aggregate(payload["seed_results"], cfg)
    payload["classification"] = payload["summary"]["classification"]
    write(out, payload)

    s = payload["summary"]
    print("classification:", payload["classification"])
    print("median early:", s["median_early_accuracy"])
    print("median final:", s["median_final_accuracy"])
    print("transient-tonic margin:", s["median_transient_minus_tonic"])
    print("transient-reset margin:", s["median_transient_minus_reset"])
    print("seed wins tonic/reset:", s["transient_seed_wins_vs_tonic"], s["transient_seed_wins_vs_reset"])
    print("receipt:", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the canonical experiment exactly once before any scientific parameter change**

```bash
python experiments/run_v0.py --out results/v0_handoff.json
```

Expected: 64 complete seed results and one frozen classification. Do not tune the gate after seeing this output.

- [ ] **Step 5: Check mechanical integrity before interpretation**

```bash
python - <<'PY'
import json
p=json.load(open('results/v0_handoff.json'))
assert len(p['seed_results']) == 64
assert p['classification'] == p['summary']['classification']
assert p['summary']['matched_invariants_valid'] is True
print(p['classification'])
print(p['summary'])
PY
```

If this fails because of matched-arm plumbing, fix code only. If it fails because the frozen scientific thresholds are not met, keep the negative result.

- [ ] **Step 6: Run receipt regression and full tests**

```bash
pytest -q
```

Expected: PASS, with continuous diagnostics compared at `1e-12` tolerance and discrete scientific metrics exact.

- [ ] **Step 7: Commit the untouched canonical receipt**

```bash
git add experiments/run_v0.py results/v0_handoff.json tests/test_v0_gate.py
git commit -m "experiment: freeze EATON v0 handoff result"
```

---

### Task 5: README and CI grounded in the actual result

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`
- Modify: `tests/test_v0_gate.py`

**Interfaces:**
- Consumes `results/v0_handoff.json`.
- Produces the public repo narrative and CI; README must report the actual classification without upgrading a negative result.

- [ ] **Step 1: Add a failing repo-surface test**

Append to `tests/test_v0_gate.py`:

```python
def test_readme_names_actual_classification_and_claim_boundary():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Event-Addressed Transient Operator Networks" in text
    assert receipt["classification"] in text
    assert "operator lifetime" in text.lower()
    assert "synthetic" in text.lower()
```

Run:

```bash
pytest tests/test_v0_gate.py::test_readme_names_actual_classification_and_claim_boundary -q
```

Expected: FAIL because README does not exist.

- [ ] **Step 2: Write README using exact receipt values**

Create `README.md` with these sections, copying exact values from `results/v0_handoff.json` rather than predicting them:

```markdown
# EATON

> **The machine does not run one algorithm. Sparse events temporarily instantiate operators in resident state; computation is the time-ordered handoff between those episodes.**

EATON = **Event-Addressed Transient Operator Networks**.

## V0 question

When two sequential jobs require different local operators, does releasing the first operator after its useful boundary preserve downstream computation better than leaving it tonically active, while still requiring resident state to survive the handoff?

## Minimal mechanism

```text
context ping -> A/acquire -> resident memory -> handoff
payload ping -> B/store -> B/compute -> final answer
```

The three matched arms are `transient`, `tonic`, and `reset`.

## Frozen v0 result

Report the receipt's exact classification, the three median early accuracies, the three median final accuracies, transient-minus-tonic and transient-minus-reset median margins, and paired seed-win counts. If classification is not `PASS_TRANSIENT_HANDOFF`, state the failure classification plainly and do not describe v0 as supporting EATON.

## Claim boundary

This is a synthetic mechanism experiment. It does not establish a biological dendritic mechanism, a transformer mechanism, general superiority of phasic control, or a benchmark advantage over conventional recurrent/gated models.

## Lineage

`SimpleNeuron` -> resident state and tiny events.  
`NSSN2` -> sparse recurrent state-conditioned receivers.  
`FrequencyAddressedNonlinearModalCell` -> addressing and resident operators.  
`AnotherOddThing` -> active pokes into resident operators.  
`EvoX` / GAx -> temporary procedures and reuse.  
`AdaptiveObserverCache` -> motivating local-vs-tail control result.

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --out results/v0_handoff.json
```

## Deferred gates

V1 asynchronous/noncommuting overlap, V2 active-set-as-state, and V3 slow residue remain deferred until v0 is frozen and interpreted.
```

The finished README must replace the result-reporting instruction with actual receipt values; do not leave generic result text in the committed README.

- [ ] **Step 3: Add Python 3.11/3.12 CI**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[test]'
      - run: pytest -q
```

- [ ] **Step 4: Verify docs and receipt reproducibility**

```bash
pytest -q
python experiments/run_v0.py --out /tmp/eaton-v0-check.json
python - <<'PY'
import json, math
c=json.load(open('results/v0_handoff.json'))
f=json.load(open('/tmp/eaton-v0-check.json'))
assert c['classification'] == f['classification']
for key in ('median_early_accuracy','median_final_accuracy','median_transient_minus_tonic','median_transient_minus_reset','transient_seed_wins_vs_tonic','transient_seed_wins_vs_reset','matched_invariants_valid','all_finite'):
    assert c['summary'][key] == f['summary'][key]
assert math.isclose(c['summary']['max_abs_state'], f['summary']['max_abs_state'], abs_tol=1e-12)
print(c['classification'])
PY
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md .github/workflows/ci.yml tests/test_v0_gate.py
git commit -m "docs: publish EATON v0 result and CI"
```

---

### Task 6: Final verification, PR, and merge gate

**Files:**
- Review all files from Tasks 1-5; no new mechanism files unless verification exposes a code defect.

**Interfaces:**
- Produces one reviewable implementation PR and verified `main` after merge.

- [ ] **Step 1: Verify frozen constants directly**

```bash
python - <<'PY'
from eaton.core import FrozenConfig
c=FrozenConfig()
assert (c.retention,c.memory_self,c.acquire_gain,c.output_gain) == (0.98,0.40,1.00,2.00)
assert (c.init_sigma,c.event_noise_sigma) == (0.05,0.08)
assert (c.seed_count,c.episodes_per_seed,c.repetitions_per_pair) == (64,256,64)
print(c)
PY
```

- [ ] **Step 2: Run complete verification from the package surface**

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --out /tmp/eaton-v0-final.json
```

Expected: PASS and fresh classification equal to committed receipt.

- [ ] **Step 3: Inspect scope**

```bash
git status --short
git diff main...HEAD --stat
git diff main...HEAD
```

Expected: only v0 core/world/experiment, runner, frozen receipt, tests, README, packaging, and CI. No v1-v3 code and no new biological or transformer claim.

- [ ] **Step 4: Push and open implementation PR**

```bash
git push -u origin HEAD
```

PR title:

```text
Build EATON v0 transient operator handoff gate
```

PR body must state the actual frozen classification and distinguish engineering verification from scientific interpretation.

- [ ] **Step 5: Require both CI versions green**

If CI exposes a numerical portability issue, preserve the scientific thresholds and diagnose the implementation or comparison tolerance. Do not change the gate to obtain green CI.

- [ ] **Step 6: Squash-merge after whole-branch review and verify main**

Squash title:

```text
Build EATON v0 transient operator handoff gate
```

After merge, verify `main` contains the same `results/v0_handoff.json` classification and README result.

---

## Plan Self-Review

- **Spec coverage:** Exact dynamics, schedule, arm differences, matched nuisance tape, destructive controls, metrics, thresholds, classification precedence, receipt contents, claim boundary, dependencies, and CI all map to explicit tasks.
- **Placeholder scan:** No `TBD`, `TODO`, generic error-handling instruction, or unspecified test remains. README result values are intentionally sourced from the not-yet-existing canonical receipt rather than invented in advance.
- **Type consistency:** `FrozenConfig`, `EpisodeInput`, `EpisodeResult`, `run_episode`, `build_seed_tape`, `run_seed`, `aggregate`, `classify`, and `run_canonical` use one signature throughout.
- **Review Focus:** Noise reversal, operator-order independence, reset isolation, matched tape replay, and classification precedence each have an explicit owning test.
- **Corrected precedence:** Matched-arm invariants are separate from numerical stability, so a boundedness/finiteness failure cannot be mislabeled `INVALID_MATCHED_ARMS`.
