from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionModel:
    raw_counts: np.ndarray
    probs: np.ndarray


@dataclass(frozen=True)
class WorldReadout:
    raw_counts: np.ndarray
    probs: np.ndarray


@dataclass(frozen=True)
class GaussianClusterModel:
    means: np.ndarray
    variance: float


def _validate_coordinate_grid(z: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    if z.ndim != 2 or symbols.ndim != 2 or z.shape != symbols.shape:
        raise ValueError("z and current symbols must have identical (N,T) shape")
    return z, symbols


def fit_transition_model(
    z: np.ndarray,
    symbols: np.ndarray,
    k: int,
    alphabet_size: int,
    alpha: float = 0.5,
) -> TransitionModel:
    z, symbols = _validate_coordinate_grid(z, symbols)
    counts = np.zeros((k, alphabet_size, k), dtype=np.float64)
    if z.shape[1] >= 2:
        origins = z[:, :-1].ravel()
        events = symbols[:, :-1].ravel()
        targets = z[:, 1:].ravel()
        np.add.at(counts, (origins, events, targets), 1.0)
    smooth = counts + float(alpha)
    probs = smooth / smooth.sum(axis=2, keepdims=True)
    return TransitionModel(raw_counts=counts, probs=probs)


def transition_nll(model: TransitionModel, z: np.ndarray, symbols: np.ndarray, *, reduction: str = "mean") -> float:
    z, symbols = _validate_coordinate_grid(z, symbols)
    if z.shape[1] < 2:
        return 0.0
    origins = z[:, :-1].ravel()
    events = symbols[:, :-1].ravel()
    targets = z[:, 1:].ravel()
    losses = -np.log(np.maximum(model.probs[origins, events, targets], 1e-300))
    if reduction == "sum":
        return float(np.sum(losses))
    if reduction == "mean":
        return float(np.mean(losses))
    raise ValueError("reduction must be mean or sum")


def fit_world_readout(
    z: np.ndarray,
    symbols: np.ndarray,
    k: int,
    alphabet_size: int,
    alpha: float = 0.5,
) -> WorldReadout:
    z = np.asarray(z, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    if z.ndim != 2 or symbols.ndim != 2 or symbols.shape != (z.shape[0], z.shape[1] + 1):
        raise ValueError("symbols must have shape (N,T+1) for z shape (N,T)")
    counts = np.zeros((k, alphabet_size, alphabet_size), dtype=np.float64)
    origins = z.ravel()
    current = symbols[:, :-1].ravel()
    nxt = symbols[:, 1:].ravel()
    np.add.at(counts, (origins, current, nxt), 1.0)
    smooth = counts + float(alpha)
    probs = smooth / smooth.sum(axis=2, keepdims=True)
    return WorldReadout(raw_counts=counts, probs=probs)


def world_readout_metrics(model: WorldReadout, z: np.ndarray, symbols: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.int64)
    if symbols.shape != (z.shape[0], z.shape[1] + 1):
        raise ValueError("symbols must have shape (N,T+1)")
    origins = z.ravel()
    current = symbols[:, :-1].ravel()
    nxt = symbols[:, 1:].ravel()
    probs = model.probs[origins, current]
    pred = np.argmax(probs, axis=1)
    p = probs[np.arange(len(nxt)), nxt]
    return {
        "accuracy": float(np.mean(pred == nxt)),
        "nll": float(-np.mean(np.log(np.maximum(p, 1e-300)))),
    }


def fit_gaussian_cluster_model(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    variance_floor: float = 1e-8,
) -> GaussianClusterModel:
    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or labels.shape != (x.shape[0],):
        raise ValueError("features must be (N,D), labels (N,)")
    global_mean = np.mean(x, axis=0)
    means = np.repeat(global_mean[None, :], k, axis=0)
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            means[j] = np.mean(x[mask], axis=0)
    residual = x - means[labels]
    variance = max(float(np.mean(residual * residual)), float(variance_floor))
    return GaussianClusterModel(means=means, variance=variance)


def gaussian_nll(
    model: GaussianClusterModel,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    reduction: str = "mean",
) -> float:
    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    d = x.shape[1]
    residual = x - model.means[labels]
    per_point = 0.5 * d * np.log(2.0 * np.pi * model.variance) + 0.5 * np.sum(residual * residual, axis=1) / model.variance
    if reduction == "sum":
        return float(np.sum(per_point))
    if reduction == "mean":
        return float(np.mean(per_point))
    raise ValueError("reduction must be mean or sum")


def mdl_score(
    *,
    validation_feature_nll_sum: float,
    validation_transition_nll_sum: float,
    k: int,
    feature_dim: int,
    alphabet_size: int,
    n_train_points: int,
) -> float:
    gaussian_parameters = k * feature_dim + 1
    transition_parameters = k * alphabet_size * (k - 1)
    parameter_count = gaussian_parameters + transition_parameters
    penalty = 0.5 * parameter_count * np.log(max(int(n_train_points), 1))
    return float(validation_feature_nll_sum + validation_transition_nll_sum + penalty)


def time_bin_labels(n_sequences: int, sequence_length: int, k: int) -> np.ndarray:
    if n_sequences <= 0 or sequence_length <= 0 or k <= 0:
        raise ValueError("n_sequences, sequence_length, and k must be positive")
    bins = np.floor(np.arange(sequence_length, dtype=np.float64) * k / sequence_length).astype(np.int64)
    bins = np.minimum(bins, k - 1)
    return np.repeat(bins[None, :], n_sequences, axis=0)


def permute_labels(labels: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(len(permutation))):
        raise ValueError("permutation must contain each label exactly once")
    return permutation[labels]
