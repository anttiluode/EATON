from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rnn import RNNParams, step_batch


@dataclass(frozen=True)
class CoordinateConfig:
    probe_count: int = 8
    probe_epsilon: float = 1e-5
    k_candidates: tuple[int, ...] = tuple(range(1, 9))
    kmeans_max_iter: int = 100
    kmeans_restarts: int = 4
    feature_std_floor: float = 1e-8
    gaussian_variance_floor: float = 1e-8
    transition_smoothing: float = 0.5


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std


@dataclass(frozen=True)
class KMeansModel:
    centroids: np.ndarray
    scaler: FeatureScaler
    labels: np.ndarray
    inertia: float


def make_probe_bank(hidden_size: int, probe_count: int, seed: int) -> np.ndarray:
    if hidden_size <= 0 or probe_count <= 0:
        raise ValueError("hidden_size and probe_count must be positive")
    rng = np.random.default_rng(seed)
    probes = rng.normal(size=(probe_count, hidden_size)).astype(np.float64)
    norms = np.linalg.norm(probes, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return probes / norms


def _validate_states_symbols(states: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    symbols = np.asarray(symbols, dtype=np.int64)
    if states.ndim != 2 or symbols.ndim != 1 or states.shape[0] != symbols.shape[0]:
        raise ValueError("states must be (N,H) and symbols must be (N,)")
    return states, symbols


def response_signatures(
    params: RNNParams,
    states: np.ndarray,
    symbols: np.ndarray,
    probes: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    states, symbols = _validate_states_symbols(states, symbols)
    probes = np.asarray(probes, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[1] != states.shape[1]:
        raise ValueError("probes must have shape (P,H)")
    base, _ = step_batch(params, states, symbols)
    chunks: list[np.ndarray] = []
    for v in probes:
        moved, _ = step_batch(params, states + float(epsilon) * v[None, :], symbols)
        chunks.append((moved - base) / float(epsilon))
    return np.concatenate(chunks, axis=1)


def analytic_response_signatures(
    params: RNNParams,
    states: np.ndarray,
    symbols: np.ndarray,
    probes: np.ndarray,
) -> np.ndarray:
    states, symbols = _validate_states_symbols(states, symbols)
    probes = np.asarray(probes, dtype=np.float64)
    nxt, _ = step_batch(params, states, symbols)
    deriv = 1.0 - nxt * nxt
    chunks: list[np.ndarray] = []
    for v in probes:
        linear = params.W_h @ v
        chunks.append(deriv * linear[None, :])
    return np.concatenate(chunks, axis=1)


def _fit_scaler(x: np.ndarray, floor: float) -> FeatureScaler:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.maximum(std, floor)
    return FeatureScaler(mean=mean, std=std)


def _sqdist(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # BLAS-backed squared distances avoid materializing (N,K,D).
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    d = x2 + c2 - 2.0 * (x @ centroids.T)
    return np.maximum(d, 0.0)


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    first = int(rng.integers(n))
    centroids[0] = x[first]
    min_dist = np.sum((x - centroids[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(np.sum(min_dist))
        if total <= 0.0 or not np.isfinite(total):
            idx = int(j % n)
        else:
            idx = int(rng.choice(n, p=min_dist / total))
        centroids[j] = x[idx]
        min_dist = np.minimum(min_dist, np.sum((x - centroids[j]) ** 2, axis=1))
    return centroids


def _fit_one_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    centroids = _kmeans_plus_plus(x, k, rng)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances = _sqdist(x, centroids)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        new_centroids = centroids.copy()
        represented = np.zeros(k, dtype=bool)
        for j in range(k):
            mask = new_labels == j
            if np.any(mask):
                new_centroids[j] = np.mean(x[mask], axis=0)
                represented[j] = True
        if not np.all(represented):
            min_d = np.min(distances, axis=1)
            order = np.argsort(-min_d, kind="mergesort")
            used: set[int] = set()
            for j in np.flatnonzero(~represented):
                idx = next((int(i) for i in order if int(i) not in used), int(j % x.shape[0]))
                used.add(idx)
                new_centroids[j] = x[idx]
        converged = np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids, rtol=0.0, atol=0.0)
        labels, centroids = new_labels, new_centroids
        if converged:
            break
    distances = _sqdist(x, centroids)
    labels = np.argmin(distances, axis=1).astype(np.int64)
    inertia = float(np.sum(distances[np.arange(x.shape[0]), labels]))
    return centroids, labels, inertia


def fit_kmeans(x: np.ndarray, k: int, seed: int, config: CoordinateConfig | None = None) -> KMeansModel:
    cfg = CoordinateConfig() if config is None else config
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a nonempty 2D array")
    if not 1 <= k <= x.shape[0]:
        raise ValueError("k must be between 1 and number of samples")
    scaler = _fit_scaler(x, cfg.feature_std_floor)
    xs = scaler.transform(x)
    best = None
    for restart in range(cfg.kmeans_restarts):
        result = _fit_one_kmeans(xs, k, seed + 10_007 * restart, cfg.kmeans_max_iter)
        if best is None or result[2] < best[2] - 1e-15:
            best = result
    assert best is not None
    centroids, labels, inertia = best
    return KMeansModel(centroids=centroids, scaler=scaler, labels=labels, inertia=inertia)


def assign_clusters(model: KMeansModel, x: np.ndarray) -> np.ndarray:
    xs = model.scaler.transform(np.asarray(x, dtype=np.float64))
    return np.argmin(_sqdist(xs, model.centroids), axis=1).astype(np.int64)


def fit_response_reconstruction(labels: np.ndarray, responses: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    responses = np.asarray(responses, dtype=np.float64)
    if labels.shape != (responses.shape[0],):
        raise ValueError("labels length must match responses")
    global_mean = np.mean(responses, axis=0)
    means = np.repeat(global_mean[None, :], k, axis=0)
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            means[j] = np.mean(responses[mask], axis=0)
    return means


def response_nrmse(target: np.ndarray, reconstructed: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((target - reconstructed) ** 2)))
    denom = float(np.sqrt(np.mean((target - np.mean(target, axis=0, keepdims=True)) ** 2)))
    if denom <= 1e-15:
        return 0.0 if rmse <= 1e-15 else float("inf")
    return rmse / denom
