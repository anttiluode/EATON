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
