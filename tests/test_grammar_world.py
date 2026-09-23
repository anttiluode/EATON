import numpy as np
import pytest

from eaton.grammar_world import WorldConfig, apply_rules, current_symbol_baseline, generate_world_split


def test_rule_offsets_match_definition():
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
    assert np.array_equal(a.rules, b.rules)
    assert not np.array_equal(a.symbols, c.symbols)


def test_rule_label_governs_same_index_transition():
    cfg = WorldConfig(sequence_length=8, train_sequences=2, validation_sequences=1, test_sequences=1)
    batch = generate_world_split(cfg, 0, "train", False)
    offsets = np.array([1, 2, 3])
    expected = (batch.symbols[:, :-1] + offsets[batch.rules]) % 4
    assert np.array_equal(expected, batch.symbols[:, 1:])


def test_current_symbol_baseline_returns_finite_metrics():
    cfg = WorldConfig(train_sequences=32, validation_sequences=8, test_sequences=8, sequence_length=12)
    train = generate_world_split(cfg, 4, "train", False)
    test = generate_world_split(cfg, 4, "test", False)
    metrics = current_symbol_baseline(train, test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert np.isfinite(metrics["nll"])


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        generate_world_split(WorldConfig(), 0, "oops", False)
