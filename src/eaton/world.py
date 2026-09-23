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
                x0=tuple(float(v) for v in x0_values),
                eta_context=float(rng.normal(0.0, config.event_noise_sigma)),
                eta_payload=float(rng.normal(0.0, config.event_noise_sigma)),
            )
        )
    return tuple(tape)
