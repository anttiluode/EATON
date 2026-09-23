import numpy as np

from eaton.grammar_world import WorldConfig, generate_world_split
from eaton.rnn import RNNConfig, init_params, rollout_states, sequence_metrics, step, train_rnn


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
    cfg = RNNConfig(hidden_size=8, epochs=3, batch_size=8)
    a = train_rnn(train, cfg, seed=9)
    b = train_rnn(train, cfg, seed=9)
    assert all(np.array_equal(x, y) for x, y in zip(a.params.arrays(), b.params.arrays()))
    assert a.loss_history == b.loss_history


def test_training_reduces_loss_on_persistent_rule_world():
    w = WorldConfig(train_sequences=64, validation_sequences=16, test_sequences=16, sequence_length=20)
    train = generate_world_split(w, 1, "train", False)
    r = train_rnn(train, RNNConfig(hidden_size=8, epochs=10, batch_size=16), seed=1)
    assert r.loss_history[-1] < 0.8 * r.loss_history[0]


def test_sequence_metrics_are_finite_and_bounded():
    w = WorldConfig(train_sequences=16, validation_sequences=8, test_sequences=8, sequence_length=8)
    batch = generate_world_split(w, 7, "test", False)
    p = init_params(RNNConfig(hidden_size=5), seed=4)
    m = sequence_metrics(p, batch.symbols)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert np.isfinite(m["nll"])


def test_gradient_clipping_keeps_training_finite_with_small_clip():
    w = WorldConfig(train_sequences=32, validation_sequences=8, test_sequences=8, sequence_length=12)
    train = generate_world_split(w, 6, "train", False)
    r = train_rnn(train, RNNConfig(hidden_size=8, epochs=2, batch_size=8, gradient_clip_norm=1e-4), seed=2)
    assert np.all(np.isfinite(r.loss_history))
    assert all(np.all(np.isfinite(a)) for a in r.params.arrays())
