from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RNNConfig:
    hidden_size: int = 16
    alphabet_size: int = 4
    init_scale: float = 0.15
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class RNNParams:
    W_h: np.ndarray
    W_x: np.ndarray
    b_h: np.ndarray
    W_y: np.ndarray
    b_y: np.ndarray

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.W_h, self.W_x, self.b_h, self.W_y, self.b_y)


@dataclass(frozen=True)
class TrainResult:
    params: RNNParams
    loss_history: tuple[float, ...]


def init_params(config: RNNConfig, seed: int) -> RNNParams:
    rng = np.random.default_rng(seed)
    h = config.hidden_size
    a = config.alphabet_size
    return RNNParams(
        W_h=rng.normal(0.0, config.init_scale / np.sqrt(h), size=(h, h)).astype(np.float64),
        W_x=rng.normal(0.0, config.init_scale / np.sqrt(a), size=(h, a)).astype(np.float64),
        b_h=np.zeros(h, dtype=np.float64),
        W_y=rng.normal(0.0, config.init_scale / np.sqrt(h), size=(a, h)).astype(np.float64),
        b_y=np.zeros(a, dtype=np.float64),
    )


def step(params: RNNParams, hidden: np.ndarray, symbol: int) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(hidden, dtype=np.float64)
    new_h = np.tanh(params.W_h @ h + params.W_x[:, int(symbol)] + params.b_h)
    logits = params.W_y @ new_h + params.b_y
    return new_h, logits


def step_batch(params: RNNParams, hidden: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(hidden, dtype=np.float64)
    s = np.asarray(symbols, dtype=np.int64)
    new_h = np.tanh(h @ params.W_h.T + params.W_x[:, s].T + params.b_h)
    logits = new_h @ params.W_y.T + params.b_y
    return new_h, logits


def rollout_states(params: RNNParams, symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int64)
    if symbols.ndim != 2 or symbols.shape[1] < 2:
        raise ValueError("symbols must have shape (N, T+1) with T>=1")
    n, width = symbols.shape
    t_steps = width - 1
    hdim = params.b_h.size
    out = np.empty((n, t_steps, hdim), dtype=np.float64)
    h = np.zeros((n, hdim), dtype=np.float64)
    for t in range(t_steps):
        out[:, t] = h
        h, _ = step_batch(params, h, symbols[:, t])
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sequence_metrics(params: RNNParams, symbols: np.ndarray) -> dict[str, float]:
    symbols = np.asarray(symbols, dtype=np.int64)
    n, width = symbols.shape
    h = np.zeros((n, params.b_h.size), dtype=np.float64)
    losses: list[np.ndarray] = []
    correct = 0
    total = 0
    for t in range(width - 1):
        h, logits = step_batch(params, h, symbols[:, t])
        probs = _softmax(logits)
        targets = symbols[:, t + 1]
        losses.append(-np.log(np.maximum(probs[np.arange(n), targets], 1e-300)))
        correct += int(np.sum(np.argmax(logits, axis=1) == targets))
        total += n
    return {"accuracy": float(correct / total), "nll": float(np.mean(np.concatenate(losses)))}


def _batch_loss_and_grads(params: RNNParams, symbols: np.ndarray) -> tuple[float, tuple[np.ndarray, ...]]:
    symbols = np.asarray(symbols, dtype=np.int64)
    batch, width = symbols.shape
    steps = width - 1
    hdim = params.b_h.size
    states = np.empty((steps + 1, batch, hdim), dtype=np.float64)
    states[0] = 0.0
    probs_list: list[np.ndarray] = []
    loss_sum = 0.0
    for t in range(steps):
        states[t + 1], logits = step_batch(params, states[t], symbols[:, t])
        probs = _softmax(logits)
        probs_list.append(probs)
        targets = symbols[:, t + 1]
        loss_sum += float(np.sum(-np.log(np.maximum(probs[np.arange(batch), targets], 1e-300))))

    dWh = np.zeros_like(params.W_h)
    dWx = np.zeros_like(params.W_x)
    dbh = np.zeros_like(params.b_h)
    dWy = np.zeros_like(params.W_y)
    dby = np.zeros_like(params.b_y)
    dh_next = np.zeros((batch, hdim), dtype=np.float64)
    scale = 1.0 / (batch * steps)

    for t in range(steps - 1, -1, -1):
        probs = probs_list[t].copy()
        targets = symbols[:, t + 1]
        probs[np.arange(batch), targets] -= 1.0
        dlogits = probs * scale
        dWy += dlogits.T @ states[t + 1]
        dby += np.sum(dlogits, axis=0)
        dh = dlogits @ params.W_y + dh_next
        dz = dh * (1.0 - states[t + 1] ** 2)
        dWh += dz.T @ states[t]
        np.add.at(dWx.T, symbols[:, t], dz)
        dbh += np.sum(dz, axis=0)
        dh_next = dz @ params.W_h

    return loss_sum / (batch * steps), (dWh, dWx, dbh, dWy, dby)


def _replace(params: RNNParams, arrays: list[np.ndarray]) -> RNNParams:
    return RNNParams(*arrays)


def train_rnn(batch, config: RNNConfig | None = None, seed: int = 0) -> TrainResult:
    cfg = RNNConfig() if config is None else config
    params = init_params(cfg, seed)
    rng = np.random.default_rng(seed + 50_000)
    m = [np.zeros_like(a) for a in params.arrays()]
    v = [np.zeros_like(a) for a in params.arrays()]
    update = 0
    losses: list[float] = []
    symbols = np.asarray(batch.symbols, dtype=np.int64)
    n = symbols.shape[0]

    for _epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        epoch_weighted_loss = 0.0
        seen = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            x = symbols[idx]
            loss, grads = _batch_loss_and_grads(params, x)
            norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
            if norm > cfg.gradient_clip_norm:
                factor = cfg.gradient_clip_norm / norm
                grads = tuple(g * factor for g in grads)
            update += 1
            new_arrays: list[np.ndarray] = []
            for i, (a, g) in enumerate(zip(params.arrays(), grads)):
                m[i] = cfg.adam_beta1 * m[i] + (1.0 - cfg.adam_beta1) * g
                v[i] = cfg.adam_beta2 * v[i] + (1.0 - cfg.adam_beta2) * (g * g)
                mhat = m[i] / (1.0 - cfg.adam_beta1**update)
                vhat = v[i] / (1.0 - cfg.adam_beta2**update)
                new_arrays.append(a - cfg.learning_rate * mhat / (np.sqrt(vhat) + cfg.adam_epsilon))
            params = _replace(params, new_arrays)
            epoch_weighted_loss += loss * len(idx)
            seen += len(idx)
        losses.append(float(epoch_weighted_loss / seen))
    return TrainResult(params=params, loss_history=tuple(losses))
