from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RULE_OFFSETS = np.array([1, 2, 3], dtype=np.int64)
SPLIT_OFFSETS = {"train": 1000, "validation": 2000, "test": 3000}


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


def apply_rules(start_symbol: int, rules: np.ndarray, alphabet_size: int = 4) -> np.ndarray:
    rules = np.asarray(rules, dtype=np.int64)
    if alphabet_size != 4:
        raise ValueError("v1 rule offsets are frozen for alphabet_size=4")
    if np.any((rules < 0) | (rules >= len(RULE_OFFSETS))):
        raise ValueError("rules must be in 0..2")
    symbols = np.empty(rules.size + 1, dtype=np.int64)
    symbols[0] = int(start_symbol) % alphabet_size
    for t, rule in enumerate(rules):
        symbols[t + 1] = (symbols[t] + RULE_OFFSETS[int(rule)]) % alphabet_size
    return symbols


def _sequence_count(config: WorldConfig, split: str) -> int:
    if split == "train":
        return config.train_sequences
    if split == "validation":
        return config.validation_sequences
    if split == "test":
        return config.test_sequences
    raise ValueError("split must be train, validation, or test")


def _draw_structured_rules(rng: np.random.Generator, length: int, config: WorldConfig) -> np.ndarray:
    rules = np.empty(length, dtype=np.int64)
    rules[0] = int(rng.integers(config.rule_count))
    for t in range(1, length):
        if rng.random() < config.structured_switch_probability:
            choices = [r for r in range(config.rule_count) if r != rules[t - 1]]
            rules[t] = int(choices[int(rng.integers(len(choices)))])
        else:
            rules[t] = rules[t - 1]
    return rules


def generate_world_split(
    config: WorldConfig,
    seed: int,
    split: Literal["train", "validation", "test"] | str,
    volatile: bool,
) -> WorldBatch:
    if config.alphabet_size != 4 or config.rule_count != 3:
        raise ValueError("EATON v1 freezes alphabet_size=4 and rule_count=3")
    count = _sequence_count(config, split)
    offset = SPLIT_OFFSETS[split] + (100_000 if volatile else 0)
    rng = np.random.default_rng(int(seed) + offset)
    symbols = np.empty((count, config.sequence_length + 1), dtype=np.int64)
    rules = np.empty((count, config.sequence_length), dtype=np.int64)
    for n in range(count):
        start = int(rng.integers(config.alphabet_size))
        if volatile:
            seq_rules = rng.integers(config.rule_count, size=config.sequence_length, dtype=np.int64)
        else:
            seq_rules = _draw_structured_rules(rng, config.sequence_length, config)
        rules[n] = seq_rules
        symbols[n] = apply_rules(start, seq_rules, config.alphabet_size)
    return WorldBatch(symbols=symbols, rules=rules)


def current_symbol_baseline(train: WorldBatch, test: WorldBatch, smoothing: float = 0.5) -> dict[str, float]:
    alphabet_size = int(max(train.symbols.max(), test.symbols.max())) + 1
    counts = np.full((alphabet_size, alphabet_size), float(smoothing), dtype=np.float64)
    current = train.symbols[:, :-1].ravel()
    nxt = train.symbols[:, 1:].ravel()
    np.add.at(counts, (current, nxt), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    test_current = test.symbols[:, :-1].ravel()
    test_next = test.symbols[:, 1:].ravel()
    pred = np.argmax(probs[test_current], axis=1)
    p = probs[test_current, test_next]
    return {
        "accuracy": float(np.mean(pred == test_next)),
        "nll": float(-np.mean(np.log(np.maximum(p, 1e-300)))),
    }
