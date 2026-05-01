"""MLX hypersphere training loop for AAGNN (paper §3.2 Eqs. 5-6)."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from layer import AbnormalityAwareLayer


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def compute_pseudo_labels(layer: AbnormalityAwareLayer, X: mx.array, neigh_lists: list[list[int]], *, pseudo_label_pct: float = 50.0, train_val_split: float = 0.6, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, mx.array]:
    if X.shape[0] < 3:
        raise ValueError("AAGNN needs at least 3 nodes so R, D, and T are non-empty.")
    if not (0.0 < pseudo_label_pct < 100.0):
        raise ValueError(f"pseudo_label_pct must be in (0, 100), got {pseudo_label_pct}")
    if not (0.0 < train_val_split < 1.0):
        raise ValueError(f"train_val_split must be in (0, 1), got {train_val_split}")
    H = layer(X, neigh_lists)
    c = mx.stop_gradient(mx.mean(H, axis=0))
    d = _np(mx.sum((H - c) ** 2, axis=-1))
    n = int(X.shape[0])
    s_size = int(round(pseudo_label_pct / 100.0 * n))
    s_size = max(2, min(n - 1, s_size))
    order = np.argsort(d).astype(np.int64)
    S = order[:s_size]
    rng = np.random.default_rng(seed)
    S_shuf = S[rng.permutation(s_size)]
    r_size = int(round(train_val_split * s_size))
    r_size = max(1, min(s_size - 1, r_size))
    R_idx = np.sort(S_shuf[:r_size]).astype(np.int64)
    D_idx = np.sort(S_shuf[r_size:]).astype(np.int64)
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True
    T_idx = np.sort(np.nonzero(~in_S)[0]).astype(np.int64)
    return R_idx, D_idx, T_idx, c


def train_aagnn(layer: AbnormalityAwareLayer, X: mx.array, neigh_lists: list[list[int]], R_idx: np.ndarray, D_idx: np.ndarray, c: mx.array, *, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 5e-4, optimizer: str = "adam", verbose: bool = True) -> dict[str, list[float]]:
    if optimizer not in {"adam", "sgd"}:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
    opt = optim.Adam(learning_rate=lr) if optimizer == "adam" else optim.SGD(learning_rate=lr)
    R = mx.array(R_idx, dtype=mx.int32)
    D = mx.array(D_idx, dtype=mx.int32)
    c = mx.stop_gradient(c)
    train_losses: list[float] = []
    val_losses: list[float] = []
    log_every = max(1, epochs // 10)

    def loss_fn(model: AbnormalityAwareLayer) -> mx.array:
        H = model(X, neigh_lists)
        loss = mx.mean(mx.sum((H[R] - c) ** 2, axis=-1))
        if weight_decay:
            l2 = mx.array(0.0)
            for p in model.trainable_parameters().values():
                if isinstance(p, dict):
                    for q in p.values():
                        l2 = l2 + mx.sum(q * q)
                else:
                    l2 = l2 + mx.sum(p * p)
            loss = loss + 0.5 * weight_decay * l2
        return loss

    grad_fn = nn.value_and_grad(layer, loss_fn)
    for epoch in range(epochs):
        loss, grads = grad_fn(layer)
        opt.update(layer, grads)
        mx.eval(layer.parameters(), opt.state)
        train_losses.append(float(loss))
        H_val = layer(X, neigh_lists)
        v = mx.mean(mx.sum((H_val[D] - c) ** 2, axis=-1))
        val_losses.append(float(v))
        if verbose and (epoch < 5 or epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch + 1}/{epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")
    return {"train_losses": train_losses, "val_losses": val_losses}


def anomaly_scores(layer: AbnormalityAwareLayer, X: mx.array, neigh_lists: list[list[int]], c: mx.array) -> np.ndarray:
    H = layer(X, neigh_lists)
    s = mx.sum((H - c) ** 2, axis=-1)
    return np.array(s).astype(np.float64)
