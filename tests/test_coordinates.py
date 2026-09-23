import numpy as np

from eaton.coordinates import (
    CoordinateConfig,
    analytic_response_signatures,
    assign_clusters,
    fit_kmeans,
    fit_response_reconstruction,
    make_probe_bank,
    response_nrmse,
    response_signatures,
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
    x = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    cfg = CoordinateConfig(kmeans_restarts=3, kmeans_max_iter=20)
    a = fit_kmeans(x, 4, seed=3, config=cfg)
    b = fit_kmeans(x, 4, seed=3, config=cfg)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.centroids, b.centroids)
    assert np.all(np.isfinite(a.centroids))
    assert a.centroids.shape == (4, 1)


def test_constant_features_stay_finite():
    x = np.ones((10, 5), dtype=np.float64)
    m = fit_kmeans(x, 3, seed=4, config=CoordinateConfig())
    labels = assign_clusters(m, x)
    assert np.all(np.isfinite(m.centroids))
    assert labels.shape == (10,)


def test_response_reconstruction_is_exact_for_cluster_constants():
    r = np.array([[1.0, 2.0], [1.0, 2.0], [-1.0, 3.0], [-1.0, 3.0]])
    labels = np.array([0, 0, 1, 1])
    means = fit_response_reconstruction(labels, r, 2)
    recon = means[labels]
    assert response_nrmse(r, recon) == 0.0


def test_probe_bank_is_repeatable_and_unit_norm():
    a = make_probe_bank(7, 5, seed=11)
    b = make_probe_bank(7, 5, seed=11)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
