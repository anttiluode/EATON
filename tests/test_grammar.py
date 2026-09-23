import math
import numpy as np

from eaton.grammar import (
    fit_gaussian_cluster_model,
    fit_transition_model,
    fit_world_readout,
    gaussian_nll,
    mdl_score,
    permute_labels,
    time_bin_labels,
    transition_nll,
    world_readout_metrics,
)


def test_transition_counts_do_not_cross_sequence_boundaries():
    z = np.array([[0, 1], [1, 0]])
    s = np.array([[0, 1], [2, 3]])
    model = fit_transition_model(z, s, k=2, alphabet_size=4, alpha=0.5)
    assert model.raw_counts.sum() == 2
    assert model.raw_counts[0, 0, 1] == 1
    assert model.raw_counts[1, 2, 0] == 1


def test_transition_smoothing_makes_unseen_next_states_finite():
    z = np.array([[0, 0, 0]])
    s = np.array([[1, 1, 1]])
    model = fit_transition_model(z, s, k=2, alphabet_size=4, alpha=0.5)
    assert np.all(model.probs > 0.0)
    assert np.isfinite(transition_nll(model, z, s))


def test_world_metrics_are_invariant_to_coordinate_label_permutation():
    z = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    symbols = np.array([[0, 1, 3, 0, 2], [2, 1, 2, 0, 1]])
    a = fit_world_readout(z, symbols, k=2, alphabet_size=4, alpha=0.5)
    ma = world_readout_metrics(a, z, symbols)
    perm = np.array([1, 0])
    zp = permute_labels(z, perm)
    b = fit_world_readout(zp, symbols, k=2, alphabet_size=4, alpha=0.5)
    mb = world_readout_metrics(b, zp, symbols)
    assert ma == mb


def test_transition_nll_is_invariant_to_coordinate_label_permutation():
    z = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    s = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    a = fit_transition_model(z, s, 2, 4, 0.5)
    perm = np.array([1, 0])
    zp = permute_labels(z, perm)
    b = fit_transition_model(zp, s, 2, 4, 0.5)
    assert math.isclose(transition_nll(a, z, s), transition_nll(b, zp, s), abs_tol=1e-15)


def test_mdl_score_matches_frozen_arithmetic():
    score = mdl_score(
        validation_feature_nll_sum=10.0,
        validation_transition_nll_sum=5.0,
        k=2,
        feature_dim=3,
        alphabet_size=4,
        n_train_points=100,
    )
    params = 2 * 3 + 1 + 2 * 4 * (2 - 1)
    expected = 15.0 + 0.5 * params * np.log(100)
    assert math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-12)


def test_gaussian_model_is_finite_for_constant_features():
    x = np.ones((8, 3))
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = fit_gaussian_cluster_model(x, labels, k=2, variance_floor=1e-8)
    assert m.variance >= 1e-8
    assert np.isfinite(gaussian_nll(m, x, labels))


def test_time_bins_reset_at_every_sequence_boundary():
    labels = time_bin_labels(n_sequences=2, sequence_length=5, k=2)
    assert labels.tolist() == [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]
