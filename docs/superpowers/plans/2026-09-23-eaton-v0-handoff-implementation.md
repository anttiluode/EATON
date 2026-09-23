# EATON v0 Transient Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze the first EATON mechanism test: a resident-state machine in which transient release of an acquisition operator must preserve a later computation better than tonic persistence, while a reset attacker proves the handoff depends on resident state.

**Architecture:** Keep the implementation deliberately small and pure. `core.py` owns the three-tick state transition and operator-lifetime semantics; `world.py` produces deterministic matched episode tapes; `experiment.py` runs the three arms, checks invariants, aggregates seed-level metrics, and classifies the frozen gate. The canonical runner writes one JSON receipt, and the README reports exactly what that receipt says without expanding the claim.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-23-eaton-v0-handoff-design.md`

## Global Constraints

- Mechanism dependencies: NumPy only; tests use pytest.
- Python CI versions: exactly 3.11 and 3.12.
- Canonical seeds: exactly integers `0..63`.
- Canonical episodes per seed: exactly 256, with exactly 64 repetitions of each `(c,p)` pair in `{(-1,-1), (-1,+1), (+1,-1), (+1,+1)}`.
- Initial resident state: each of four coordinates sampled once per episode as `Normal(0, 0.05)` and replayed identically across arms.
- Context nuisance: `eta_context ~ Normal(0, 0.08)`; payload nuisance: `eta_payload ~ Normal(0, 0.08)`; no clipping.
- Passive retention: `0.98` for coordinates not actively written, except `decision_trace`, which is frozen after the stage-1 decision.
- Operator A write: `memory = tanh(0.40 * memory + 1.00 * u)` and `decision_trace = memory`.
- Operator B payload write: `payload_trace = tanh(u)`.
- Operator B compute write: `output = tanh(2.00 * memory * payload_trace)`.
- Exact logical schedule: `t0=context/A`, handoff, `t1=payload/B (+ A only for tonic)`, `t2=no event/B compute`.
- `transient` and `tonic` differ only in A eligibility at `t1`.
- `reset` differs from `transient` only by resetting `memory` to that episode's original nuisance `x0[memory]` at the handoff boundary.
- Canonical receipt path: `results/v0_handoff.json`.
- Do not implement v1-v3 mechanisms in this plan.
- Do not retune coefficients, seeds, episode counts, thresholds, noise scales, retention, or gains after observing the canonical v0 receipt.

## Review Focus

1. **Noise sign reversals:** delivered events are not clipped; a nuisance sample that flips the sign must count naturally as an error. Task 1 pins this with an explicit negative-context sign-reversal test.
2. **A/B execution order at `t1`:** tonic A and B write disjoint coordinates, so swapping invocation order must produce identical state. Task 1 tests both orders bit-for-bit.
3. **Reset isolation:** the reset arm must restore only `memory` to that episode's original `x0[memory]`, leaving decision trace and other coordinates identical to transient at the handoff. Task 1 tests the exact coordinate diff.
4. **Matched tape replay:** all arms must consume the exact same labels, `x0`, and nuisance values for an episode; no arm-specific RNG calls are allowed. Task 2 tests deterministic serialization and arm replay.
5. **Classification precedence:** invariant failure must outrank scientific outcomes; tonic failure must outrank reset failure; weak/unstable transient must outrank PASS. Task 3 tests every classifier branch with synthetic aggregates.

---

### Task 1: Scaffold the package and implement the three-tick resident-state machine

**Files:**
- Create: `pyproject.toml`
- Create: `src/eaton/__init__.py`
- Create: `src/eaton/core.py`
- Create: `tests/test_core.py`

**Interfaces:**
- Consumes: only Python stdlib plus NumPy.
- Produces:
  - `FrozenConfig` dataclass with all frozen numerical constants.
  - `EpisodeInput` dataclass carrying one fully sampled episode.
  - `EpisodeResult` dataclass carrying stage-1, handoff, and final diagnostics.
  - `run_episode(episode: EpisodeInput, arm: str, config: FrozenConfig, *, disable_a: bool = False, disable_b: bool = False, clamp_memory_to_context: bool = False, operator_order: tuple[str, str] = ("A", "B")) -> EpisodeResult`.

- [ ] **Step 1: Add packaging and a failing import test**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eaton"
version = "0.0.1"
description = "Event-Addressed Transient Operator Networks mechanism tests"
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

Create `src/eaton/__init__.py`:

```python
from .core import EpisodeInput, EpisodeResult, FrozenConfig, run_episode

__all__ = ["EpisodeInput", "EpisodeResult", "FrozenConfig", "run_episode"]
```

Create the first `tests/test_core.py` test:

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
```

- [ ] **Step 2: Run the import/config test and verify RED**

Run:

```bash
python -m pip install -e '.[test]'
pytest tests/test_core.py::test_frozen_config_matches_spec -q
```

Expected: FAIL because `eaton.core` / `FrozenConfig` does not exist yet.

- [ ] **Step 3: Implement frozen types and constants**

Create `src/eaton/core.py` with these exact public types:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Arm = Literal["transient", "tonic", "reset"]

MEMORY = 0
DECISION = 1
PAYLOAD = 2
OUTPUT = 3
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


def _tuple_state(x: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in x)  # type: ignore[return-value]
```

- [ ] **Step 4: Run the config test and verify GREEN**

Run:

```bash
pytest tests/test_core.py::test_frozen_config_matches_spec -q
```

Expected: PASS.

- [ ] **Step 5: Write failing tests for the exact three-tick semantics**

Append to `tests/test_core.py`:

```python
import math

import numpy as np

from eaton.core import EpisodeInput, run_episode


def canonical_episode(context=1, payload=-1):
    return EpisodeInput(
        context=context,
        payload=payload,
        x0=(0.01, -0.02, 0.03, -0.04),
        eta_context=0.0,
        eta_payload=0.0,
    )


def test_stage1_is_identical_across_all_arms():
    ep = canonical_episode()
    results = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    assert results[0].early_state == results[1].early_state == results[2].early_state
    assert results[0].early_decision == results[1].early_decision == results[2].early_decision == 1


def test_tonic_differs_from_transient_only_after_handoff():
    ep = canonical_episode(context=1, payload=-1)
    transient = run_episode(ep, "transient", FrozenConfig())
    tonic = run_episode(ep, "tonic", FrozenConfig())
    assert transient.early_state == tonic.early_state
    assert transient.handoff_state == tonic.handoff_state
    assert transient.precompute_state[0] != tonic.precompute_state[0]
    assert transient.precompute_state[2] == tonic.precompute_state[2]
    assert transient.overlap_event_ticks == 0
    assert tonic.overlap_event_ticks == 1


def test_reset_changes_only_memory_at_handoff():
    ep = canonical_episode()
    transient = run_episode(ep, "transient", FrozenConfig())
    reset = run_episode(ep, "reset", FrozenConfig())
    assert transient.early_state == reset.early_state
    diffs = [i for i, (a, b) in enumerate(zip(transient.handoff_state, reset.handoff_state)) if a != b]
    assert diffs == [0]
    assert reset.handoff_state[0] == ep.x0[0]


def test_payload_tick_is_operator_order_independent():
    ep = canonical_episode(context=-1, payload=1)
    ab = run_episode(ep, "tonic", FrozenConfig(), operator_order=("A", "B"))
    ba = run_episode(ep, "tonic", FrozenConfig(), operator_order=("B", "A"))
    assert ab == ba


def test_external_event_budget_is_identical_across_arms():
    ep = EpisodeInput(1, -1, (0.0, 0.0, 0.0, 0.0), 0.1, -0.2)
    results = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    assert {r.event_count for r in results} == {2}
    expected_sum = abs(1.1) + abs(-1.2)
    assert all(math.isclose(r.event_abs_sum, expected_sum, abs_tol=1e-15) for r in results)


def test_noise_sign_reversal_counts_as_a_real_error_without_clipping():
    ep = EpisodeInput(1, 1, (0.0, 0.0, 0.0, 0.0), -1.5, 0.0)
    result = run_episode(ep, "transient", FrozenConfig())
    assert ep.context_event == -0.5
    assert result.early_decision == -1
    assert result.early_correct is False


def test_state_is_finite_and_bounded_by_tanh():
    ep = EpisodeInput(1, 1, (0.04, -0.04, 0.04, -0.04), 1000.0, -1000.0)
    result = run_episode(ep, "tonic", FrozenConfig())
    assert np.isfinite(result.max_abs_state)
    assert result.max_abs_state <= FrozenConfig().state_bound
```

- [ ] **Step 6: Run the mechanism tests and verify RED**

Run:

```bash
pytest tests/test_core.py -q
```

Expected: FAIL because `run_episode` is not implemented.

- [ ] **Step 7: Implement `run_episode` with disjoint payload-tick writes**

Add to `src/eaton/core.py`:

```python
def _passive(x: np.ndarray, written: set[int], cfg: FrozenConfig, *, freeze_decision: bool) -> None:
    for i in range(STATE_DIM):
        if i in written:
            continue
        if freeze_decision and i == DECISION:
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
    pre = x.copy()
    writes: dict[int, float] = {}
    for operator in operator_order:
        if operator == "A" and a_active:
            writes[MEMORY] = float(np.tanh(cfg.memory_self * pre[MEMORY] + cfg.acquire_gain * event))
        elif operator == "B" and b_active:
            writes[PAYLOAD] = float(np.tanh(event))
        elif operator not in {"A", "B"}:
            raise ValueError(f"unknown operator {operator!r}")
    if len(set(writes)) != len(writes):
        raise RuntimeError("payload-tick operators attempted to write the same coordinate")
    _passive(x, set(writes), cfg, freeze_decision=True)
    for index, value in writes.items():
        x[index] = value


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
    if sorted(operator_order) != ["A", "B"]:
        raise ValueError("operator_order must contain A and B exactly once")

    x = np.asarray(episode.x0, dtype=np.float64).copy()
    snapshots = [x.copy()]

    # t0: context / acquire
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

    # t1: payload / store; tonic A remains eligible
    _payload_tick(
        x,
        episode.payload_event,
        a_active=(arm == "tonic" and not disable_a),
        b_active=not disable_b,
        cfg=config,
        operator_order=operator_order,
    )
    precompute_state = x.copy()
    snapshots.append(x.copy())

    # t2: no external event; B computes after optional destructive/control clamp
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
        early_state=_tuple_state(early_state),
        handoff_state=_tuple_state(handoff_state),
        precompute_state=_tuple_state(precompute_state),
        final_state=_tuple_state(x),
        memory_before_compute=float(precompute_state[MEMORY]),
        overlap_event_ticks=1 if arm == "tonic" and not disable_a and not disable_b else 0,
        event_count=2,
        event_abs_sum=abs(episode.context_event) + abs(episode.payload_event),
        max_abs_state=max_abs_state,
        early_correct=(early_decision == episode.context),
        final_correct=(final_answer == episode.context * episode.payload),
    )
```

- [ ] **Step 8: Run the mechanism tests and verify GREEN**

Run:

```bash
pytest tests/test_core.py -q
```

Expected: all Task 1 tests PASS.

- [ ] **Step 9: Add destructive-control tests for A, B, and correct-memory clamp**

Append to `tests/test_core.py`:

```python
def balanced_exact_tape():
    pairs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] * 64
    return [EpisodeInput(c, p, (0.01, -0.02, 0.03, -0.04), 0.0, 0.0) for c, p in pairs]


def test_disabling_a_removes_context_signal_from_early_decision():
    results = [run_episode(ep, "transient", FrozenConfig(), disable_a=True) for ep in balanced_exact_tape()]
    assert sum(r.early_correct for r in results) / len(results) == 0.5


def test_disabling_b_removes_parity_solution():
    results = [run_episode(ep, "transient", FrozenConfig(), disable_b=True) for ep in balanced_exact_tape()]
    assert sum(r.final_correct for r in results) / len(results) == 0.5


def test_correct_memory_clamp_lets_b_solve_second_stage():
    results = [
        run_episode(ep, "reset", FrozenConfig(), clamp_memory_to_context=True)
        for ep in balanced_exact_tape()
    ]
    assert all(r.final_correct for r in results)
```

- [ ] **Step 10: Run Task 1 tests, then commit**

Run:

```bash
pytest tests/test_core.py -q
```

Expected: PASS.

Commit:

```bash
git add pyproject.toml src/eaton/__init__.py src/eaton/core.py tests/test_core.py
git commit -m "feat: add EATON transient handoff core"
```

---

### Task 2: Build the frozen deterministic world tape

**Files:**
- Create: `src/eaton/world.py`
- Create: `tests/test_world.py`
- Modify: `src/eaton/__init__.py`

**Interfaces:**
- Consumes: `FrozenConfig`, `EpisodeInput` from Task 1.
- Produces: `build_seed_tape(seed: int, config: FrozenConfig) -> tuple[EpisodeInput, ...]`.

- [ ] **Step 1: Write failing balance/determinism/replay tests**

Create `tests/test_world.py`:

```python
from collections import Counter

from eaton.core import FrozenConfig
from eaton.world import build_seed_tape


def test_seed_tape_has_exact_frozen_balance():
    cfg = FrozenConfig()
    tape = build_seed_tape(0, cfg)
    assert len(tape) == 256
    counts = Counter((ep.context, ep.payload) for ep in tape)
    assert counts == {
        (-1, -1): 64,
        (-1, 1): 64,
        (1, -1): 64,
        (1, 1): 64,
    }


def test_seed_tape_is_deterministic_and_seed_specific():
    cfg = FrozenConfig()
    assert build_seed_tape(17, cfg) == build_seed_tape(17, cfg)
    assert build_seed_tape(17, cfg) != build_seed_tape(18, cfg)


def test_tape_stores_noise_once_for_arm_replay():
    ep = build_seed_tape(3, FrozenConfig())[0]
    snapshot = (ep.context, ep.payload, ep.x0, ep.eta_context, ep.eta_payload)
    assert snapshot == (ep.context, ep.payload, ep.x0, ep.eta_context, ep.eta_payload)
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/test_world.py -q
```

Expected: FAIL because `eaton.world` does not exist.

- [ ] **Step 3: Implement exact tape generation with one RNG per seed**

Create `src/eaton/world.py`:

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
        raise RuntimeError("frozen pair count does not match episodes_per_seed")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    tape: list[EpisodeInput] = []
    for index in order:
        context, payload = labels[int(index)]
        x0 = tuple(float(v) for v in rng.normal(0.0, config.init_sigma, size=STATE_DIM))
        eta_context = float(rng.normal(0.0, config.event_noise_sigma))
        eta_payload = float(rng.normal(0.0, config.event_noise_sigma))
        tape.append(
            EpisodeInput(
                context=context,
                payload=payload,
                x0=x0,  # type: ignore[arg-type]
                eta_context=eta_context,
                eta_payload=eta_payload,
            )
        )
    return tuple(tape)
```

Update `src/eaton/__init__.py` to export `build_seed_tape`.

- [ ] **Step 4: Add a no-hidden-RNG regression test**

Append to `tests/test_world.py`:

```python
from dataclasses import asdict


def test_serialized_tape_replays_exact_episode_values():
    cfg = FrozenConfig()
    first = [asdict(ep) for ep in build_seed_tape(9, cfg)]
    second = [asdict(ep) for ep in build_seed_tape(9, cfg)]
    assert first == second
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/test_world.py tests/test_core.py -q
```

Expected: PASS.

Commit:

```bash
git add src/eaton/__init__.py src/eaton/world.py tests/test_world.py
git commit -m "feat: add frozen EATON world tape"
```

---

### Task 3: Implement matched-arm evaluation, invariants, metrics, and frozen classification

**Files:**
- Create: `src/eaton/experiment.py`
- Create: `tests/test_experiment.py`
- Modify: `src/eaton/__init__.py`

**Interfaces:**
- Consumes: `run_episode`, `FrozenConfig`, `EpisodeResult`, `build_seed_tape`.
- Produces:
  - `run_seed(seed: int, config: FrozenConfig) -> dict`.
  - `aggregate(seed_results: list[dict], config: FrozenConfig) -> dict`.
  - `classify(summary: dict, config: FrozenConfig) -> str`.
  - `run_canonical(config: FrozenConfig | None = None) -> dict`.

- [ ] **Step 1: Write failing seed-level matched-arm tests**

Create `tests/test_experiment.py`:

```python
from eaton.core import FrozenConfig
from eaton.experiment import run_seed


def test_run_seed_replays_identical_external_budget_and_stage1():
    result = run_seed(0, FrozenConfig())
    inv = result["invariants"]
    assert inv["stage1_transient_tonic_equal"] is True
    assert inv["stage1_transient_reset_equal"] is True
    assert inv["event_counts_equal"] is True
    assert inv["event_amplitude_sums_equal"] is True
    assert result["arms"]["transient"]["event_count"] == 512


def test_seed_records_paired_episode_disagreements():
    result = run_seed(0, FrozenConfig())
    paired = result["paired_episode_disagreements"]
    assert set(paired) == {"transient_vs_tonic", "transient_vs_reset"}
    for counts in paired.values():
        assert set(counts) == {"transient_only_correct", "attacker_only_correct", "both_correct", "both_wrong"}
        assert sum(counts.values()) == 256
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/test_experiment.py -q
```

Expected: FAIL because `eaton.experiment` does not exist.

- [ ] **Step 3: Implement arm and seed aggregation**

Create `src/eaton/experiment.py` with these helpers and public functions:

```python
from __future__ import annotations

from statistics import median

import numpy as np

from .core import FrozenConfig, run_episode
from .world import build_seed_tape

ARMS = ("transient", "tonic", "reset")


def _safe_corr(a: list[float], b: list[int]) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def _arm_metrics(results, tape):
    return {
        "early_accuracy": float(np.mean([r.early_correct for r in results])),
        "final_accuracy": float(np.mean([r.final_correct for r in results])),
        "handoff_memory_correlation": _safe_corr(
            [r.memory_before_compute for r in results],
            [ep.context for ep in tape],
        ),
        "operator_overlap_event_ticks": int(sum(r.overlap_event_ticks for r in results)),
        "event_count": int(sum(r.event_count for r in results)),
        "event_abs_sum": float(sum(r.event_abs_sum for r in results)),
        "max_abs_state": float(max(r.max_abs_state for r in results)),
    }


def _paired_counts(transient, attacker):
    counts = {
        "transient_only_correct": 0,
        "attacker_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for t, a in zip(transient, attacker):
        if t.final_correct and not a.final_correct:
            counts["transient_only_correct"] += 1
        elif a.final_correct and not t.final_correct:
            counts["attacker_only_correct"] += 1
        elif t.final_correct and a.final_correct:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def run_seed(seed: int, config: FrozenConfig) -> dict:
    tape = build_seed_tape(seed, config)
    raw = {arm: [run_episode(ep, arm, config) for ep in tape] for arm in ARMS}

    stage1_tt = all(
        a.early_state == b.early_state and a.early_decision == b.early_decision
        for a, b in zip(raw["transient"], raw["tonic"])
    )
    stage1_tr = all(
        a.early_state == b.early_state and a.early_decision == b.early_decision
        for a, b in zip(raw["transient"], raw["reset"])
    )
    arm_metrics = {arm: _arm_metrics(raw[arm], tape) for arm in ARMS}
    counts = {arm_metrics[a]["event_count"] for a in ARMS}
    sums = [arm_metrics[a]["event_abs_sum"] for a in ARMS]

    return {
        "seed": seed,
        "arms": arm_metrics,
        "paired_episode_disagreements": {
            "transient_vs_tonic": _paired_counts(raw["transient"], raw["tonic"]),
            "transient_vs_reset": _paired_counts(raw["transient"], raw["reset"]),
        },
        "invariants": {
            "stage1_transient_tonic_equal": stage1_tt,
            "stage1_transient_reset_equal": stage1_tr,
            "event_counts_equal": len(counts) == 1,
            "event_amplitude_sums_equal": max(sums) - min(sums) <= 1e-12,
            "all_finite": all(np.isfinite(arm_metrics[a]["max_abs_state"]) for a in ARMS),
            "all_bounded": all(arm_metrics[a]["max_abs_state"] <= config.state_bound for a in ARMS),
        },
    }
```

- [ ] **Step 4: Run seed-level tests and verify GREEN**

Run:

```bash
pytest tests/test_experiment.py::test_run_seed_replays_identical_external_budget_and_stage1 tests/test_experiment.py::test_seed_records_paired_episode_disagreements -q
```

Expected: PASS.

- [ ] **Step 5: Write failing tests for all frozen classifier branches**

Append to `tests/test_experiment.py`:

```python
from copy import deepcopy

from eaton.experiment import classify


def passing_summary():
    return {
        "all_invariants_valid": True,
        "median_early_accuracy": {"transient": 1.0, "tonic": 1.0, "reset": 1.0},
        "median_final_accuracy": {"transient": 0.98, "tonic": 0.50, "reset": 0.50},
        "median_transient_minus_tonic": 0.48,
        "median_transient_minus_reset": 0.48,
        "transient_seed_wins_vs_tonic": 64,
        "transient_seed_wins_vs_reset": 64,
        "all_finite": True,
        "max_abs_state": 0.99,
    }


def test_classifier_precedence_invalid_matched_arms():
    s = passing_summary()
    s["all_invariants_valid"] = False
    assert classify(s, FrozenConfig()) == "INVALID_MATCHED_ARMS"


def test_classifier_no_handoff_advantage_precedes_reset_failure():
    s = passing_summary()
    s["median_transient_minus_tonic"] = 0.19
    s["median_transient_minus_reset"] = 0.0
    assert classify(s, FrozenConfig()) == "NO_HANDOFF_ADVANTAGE"


def test_classifier_resident_memory_failure():
    s = passing_summary()
    s["median_transient_minus_reset"] = 0.29
    assert classify(s, FrozenConfig()) == "HANDOFF_WITHOUT_RESIDENT_MEMORY"


def test_classifier_weak_or_unstable_transient():
    s = passing_summary()
    s["median_final_accuracy"]["transient"] = 0.89
    assert classify(s, FrozenConfig()) == "UNSTABLE_OR_WEAK_TRANSIENT"


def test_classifier_pass():
    assert classify(passing_summary(), FrozenConfig()) == "PASS_TRANSIENT_HANDOFF"
```

- [ ] **Step 6: Implement aggregate and exact classification thresholds**

Append to `src/eaton/experiment.py`:

```python
def aggregate(seed_results: list[dict], config: FrozenConfig) -> dict:
    arm_medians = {
        metric: {
            arm: float(median([r["arms"][arm][metric] for r in seed_results]))
            for arm in ARMS
        }
        for metric in ("early_accuracy", "final_accuracy", "handoff_memory_correlation")
    }
    transient_minus_tonic = [
        r["arms"]["transient"]["final_accuracy"] - r["arms"]["tonic"]["final_accuracy"]
        for r in seed_results
    ]
    transient_minus_reset = [
        r["arms"]["transient"]["final_accuracy"] - r["arms"]["reset"]["final_accuracy"]
        for r in seed_results
    ]
    all_invariants_valid = all(all(r["invariants"].values()) for r in seed_results)
    summary = {
        "all_invariants_valid": all_invariants_valid,
        "median_early_accuracy": arm_medians["early_accuracy"],
        "median_final_accuracy": arm_medians["final_accuracy"],
        "median_handoff_memory_correlation": arm_medians["handoff_memory_correlation"],
        "median_transient_minus_tonic": float(median(transient_minus_tonic)),
        "median_transient_minus_reset": float(median(transient_minus_reset)),
        "transient_seed_wins_vs_tonic": int(sum(v > 0.0 for v in transient_minus_tonic)),
        "transient_seed_wins_vs_reset": int(sum(v > 0.0 for v in transient_minus_reset)),
        "all_finite": all(r["invariants"]["all_finite"] for r in seed_results),
        "max_abs_state": float(max(r["arms"][arm]["max_abs_state"] for r in seed_results for arm in ARMS)),
    }
    summary["classification"] = classify(summary, config)
    return summary


def classify(summary: dict, config: FrozenConfig) -> str:
    if not summary["all_invariants_valid"]:
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


def run_canonical(config: FrozenConfig | None = None) -> dict:
    cfg = FrozenConfig() if config is None else config
    seed_results = [run_seed(seed, cfg) for seed in range(cfg.seed_count)]
    return {
        "config": cfg.__dict__,
        "seed_results": seed_results,
        "summary": aggregate(seed_results, cfg),
    }
```

- [ ] **Step 7: Run all experiment tests and commit**

Run:

```bash
pytest tests/test_experiment.py tests/test_world.py tests/test_core.py -q
```

Expected: PASS.

Commit:

```bash
git add src/eaton/__init__.py src/eaton/experiment.py tests/test_experiment.py
git commit -m "feat: add EATON matched-arm gate"
```

---

### Task 4: Add the canonical runner, frozen receipt, and receipt regression tests

**Files:**
- Create: `experiments/run_v0.py`
- Create: `results/v0_handoff.json`
- Create: `tests/test_v0_gate.py`

**Interfaces:**
- Consumes: `run_canonical()` from Task 3.
- Produces: deterministic human-readable console summary and `results/v0_handoff.json` containing config, frozen parameter block, seed metrics, aggregate summary, invariant checks, classification, and claim boundary.

- [ ] **Step 1: Write a failing runner/receipt schema test**

Create `tests/test_v0_gate.py`:

```python
import json
from pathlib import Path

from eaton.core import FrozenConfig
from eaton.experiment import run_canonical


RECEIPT = Path("results/v0_handoff.json")


def test_canonical_run_has_frozen_seed_count_and_required_fields():
    payload = run_canonical()
    assert len(payload["seed_results"]) == 64
    assert payload["config"]["episodes_per_seed"] == 256
    assert set(payload["summary"]["median_final_accuracy"]) == {"transient", "tonic", "reset"}
    assert payload["summary"]["classification"] in {
        "INVALID_MATCHED_ARMS",
        "NO_HANDOFF_ADVANTAGE",
        "HANDOFF_WITHOUT_RESIDENT_MEMORY",
        "UNSTABLE_OR_WEAK_TRANSIENT",
        "PASS_TRANSIENT_HANDOFF",
    }


def test_committed_receipt_matches_fresh_canonical_summary():
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    fresh = run_canonical()
    assert committed["config"] == fresh["config"]
    assert committed["summary"] == fresh["summary"]
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/test_v0_gate.py -q
```

Expected: first test may pass after Tasks 1-3; second test FAILS because the receipt does not exist yet.

- [ ] **Step 3: Create a runner that writes partial then final receipt safely**

Create `experiments/run_v0.py`:

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
        "frozen_parameters": {
            "retention": cfg.retention,
            "memory_self": cfg.memory_self,
            "acquire_gain": cfg.acquire_gain,
            "output_gain": cfg.output_gain,
            "init_sigma": cfg.init_sigma,
            "event_noise_sigma": cfg.event_noise_sigma,
            "seeds": list(range(cfg.seed_count)),
            "episodes_per_seed": cfg.episodes_per_seed,
            "repetitions_per_pair": cfg.repetitions_per_pair,
        },
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
    print("EATON v0")
    print("classification:", payload["classification"])
    print("median final accuracy:", s["median_final_accuracy"])
    print("transient-tonic margin:", s["median_transient_minus_tonic"])
    print("transient-reset margin:", s["median_transient_minus_reset"])
    print("seed wins vs tonic:", s["transient_seed_wins_vs_tonic"], "/ 64")
    print("seed wins vs reset:", s["transient_seed_wins_vs_reset"], "/ 64")
    print("receipt:", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the exact canonical experiment once**

Run:

```bash
python experiments/run_v0.py --out results/v0_handoff.json
```

Expected: completes all 64 seeds, prints the frozen classification, and writes a complete receipt. **Do not change any frozen coefficient or threshold in response to this output.**

- [ ] **Step 5: Inspect the receipt mechanically before interpreting it**

Run:

```bash
python - <<'PY'
import json
p=json.load(open('results/v0_handoff.json'))
assert len(p['seed_results']) == 64
assert p['classification'] == p['summary']['classification']
assert p['summary']['all_invariants_valid'] is True
print(p['classification'])
print(p['summary'])
PY
```

Expected: assertions PASS. If an invariant assertion fails, fix implementation only; do not tune scientific parameters.

- [ ] **Step 6: Run receipt regression and full test suite**

Run:

```bash
pytest -q
```

Expected: PASS, including fresh canonical summary matching the committed receipt summary exactly.

- [ ] **Step 7: Commit the result without editing it by hand**

Commit:

```bash
git add experiments/run_v0.py results/v0_handoff.json tests/test_v0_gate.py
git commit -m "experiment: freeze EATON v0 handoff result"
```

---

### Task 5: Document the actual result and add CI without broadening the claim

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`
- Modify: `tests/test_v0_gate.py`

**Interfaces:**
- Consumes: committed canonical receipt from Task 4.
- Produces: repository-facing explanation and CI that guards mechanism tests plus receipt reproducibility.

- [ ] **Step 1: Write a failing README/CI presence test**

Append to `tests/test_v0_gate.py`:

```python
def test_repo_surface_names_the_frozen_result_and_claim_boundary():
    readme = Path("README.md").read_text(encoding="utf-8")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert "Event-Addressed Transient Operator Networks" in readme
    assert receipt["classification"] in readme
    assert "synthetic" in readme.lower()
    assert "operator lifetime" in readme.lower()
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/test_v0_gate.py::test_repo_surface_names_the_frozen_result_and_claim_boundary -q
```

Expected: FAIL because `README.md` does not exist.

- [ ] **Step 3: Write README from the receipt, not from expectation**

Create `README.md` with this structure and replace no scientific result by inference; copy the exact numeric values and classification from `results/v0_handoff.json`:

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

Three matched arms differ only at the handoff:

- `transient`: A releases before payload;
- `tonic`: A remains eligible and interprets payload too;
- `reset`: A releases but resident memory is reset before B.

## Frozen v0 result

Read `results/v0_handoff.json`; report its exact classification, median early accuracies, median final accuracies, transient-minus-tonic/reset margins, and paired seed wins here.

Do not describe a FAIL-classified result as support for EATON. If the classification is `PASS_TRANSIENT_HANDOFF`, state only that this synthetic matched machine earns operator lifetime as a computational variable.

## Claim boundary

This is a synthetic mechanism experiment. It does not establish a biological dendritic mechanism, a transformer mechanism, general superiority of phasic control, or a benchmark advantage over conventional recurrent/gated models.

## Lineage

- `SimpleNeuron`: resident state + tiny routed events.
- `NSSN2`: sparse recurrent state-conditioned receivers.
- `FrequencyAddressedNonlinearModalCell`: addressing and reduced resident operators.
- `AnotherOddThing`: active pokes into hidden resident operators.
- `EvoX` / GAx: temporary procedures, alternatives, and reuse.
- `AdaptiveObserverCache`: persistent observer state and the motivating local-vs-tail control result.

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --out results/v0_handoff.json
```

## Next gates

V1 asynchronous/noncommuting overlap, V2 active-set-as-state, and V3 slow residue remain deferred until v0 is frozen and interpreted.
```

The implementation step must replace the result-reporting instruction in that template with the receipt's actual numbers; the finished README must contain no bracketed placeholders or speculative values.

- [ ] **Step 4: Add Python 3.11/3.12 CI**

Create `.github/workflows/ci.yml`:

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

- [ ] **Step 5: Run local verification and commit**

Run:

```bash
pytest -q
python experiments/run_v0.py --out /tmp/eaton-v0-check.json
python - <<'PY'
import json
committed=json.load(open('results/v0_handoff.json'))
fresh=json.load(open('/tmp/eaton-v0-check.json'))
assert committed['summary'] == fresh['summary']
assert committed['classification'] == fresh['classification']
print(committed['classification'])
PY
```

Expected: all tests PASS and the fresh classification/summary exactly match the committed receipt.

Commit:

```bash
git add README.md .github/workflows/ci.yml tests/test_v0_gate.py
git commit -m "docs: publish EATON v0 result and CI"
```

---

### Task 6: Final branch verification and integration

**Files:**
- Review all files created in Tasks 1-5.
- No new mechanism files unless verification exposes an implementation defect.

**Interfaces:**
- Consumes: complete branch.
- Produces: one reviewable PR whose result is reproducible from the frozen spec.

- [ ] **Step 1: Verify no scientific constants drifted from the spec**

Run:

```bash
python - <<'PY'
from eaton.core import FrozenConfig
c=FrozenConfig()
assert c.retention == 0.98
assert c.memory_self == 0.40
assert c.acquire_gain == 1.00
assert c.output_gain == 2.00
assert c.init_sigma == 0.05
assert c.event_noise_sigma == 0.08
assert c.seed_count == 64
assert c.episodes_per_seed == 256
assert c.repetitions_per_pair == 64
print(c)
PY
```

Expected: PASS and exact frozen config printed.

- [ ] **Step 2: Run complete verification from a clean install**

Run:

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --out /tmp/eaton-v0-final.json
python - <<'PY'
import json
committed=json.load(open('results/v0_handoff.json'))
fresh=json.load(open('/tmp/eaton-v0-final.json'))
assert committed['summary'] == fresh['summary']
assert committed['classification'] == fresh['classification']
print('verified', committed['classification'])
PY
```

Expected: PASS.

- [ ] **Step 3: Inspect diff for scope creep**

Run:

```bash
git status --short
git diff main...HEAD --stat
git diff main...HEAD
```

Expected: only v0 handoff implementation, frozen receipt, tests, docs, packaging, and CI. No v1-v3 implementation and no biological/transformer mechanism claim.

- [ ] **Step 4: Push branch and open the implementation PR**

Run:

```bash
git push -u origin HEAD
```

Open a PR titled:

```text
Build EATON v0 transient operator handoff gate
```

PR body must state the actual frozen classification and explicitly separate implementation verification from the scientific interpretation.

- [ ] **Step 5: Require green CI before merge**

Both Python 3.11 and 3.12 jobs must pass. If CI changes numerical summary values, diagnose platform/numerical behavior; do not relax the scientific thresholds to make CI green.

- [ ] **Step 6: Squash-merge only after final review**

Use squash title:

```text
Build EATON v0 transient operator handoff gate
```

Then verify `main` contains the exact committed receipt and README classification.

---

## Plan Self-Review

- **Spec coverage:** Every frozen coefficient, schedule detail, arm difference, destructive control, metric, classification threshold, receipt requirement, dependency constraint, and claim boundary maps to Tasks 1-5. V1-v3 are explicitly excluded.
- **Placeholder scan:** No implementation step contains `TBD`/`TODO` or an unspecified error-handling instruction. The README task requires copying actual receipt values after the canonical run rather than inventing values in advance.
- **Type consistency:** `FrozenConfig`, `EpisodeInput`, `EpisodeResult`, `run_episode`, `build_seed_tape`, `run_seed`, `aggregate`, `classify`, and `run_canonical` retain the same signatures wherever referenced.
- **Review Focus:** Noise reversal, operator-order independence, reset isolation, matched tape replay, and classifier precedence each have explicit tests in the owning task.
