"""Hypersphere training loop for AAGNN (paper §3.2 Eqs. 5-6, §3.3 Algorithm 1).

Reference: Zhou et al., "Subtractive Aggregation for Attributed Network Anomaly
Detection", CIKM 2021.

Implements Algorithm 1:
    Lines 1-5  -> compute_pseudo_labels: random-init forward, fix centre c,
                  pick the p% closest-to-c nodes as pseudo-normal S, split into
                  R (train) and D (val); test set T = V - S.
    Lines 6-10 -> train_aagnn: SGD/Adam minimisation of Eq. 5
                      L = (1/|R|) * sum_{i in R} ||h_i - c||^2 + (lambda/2) ||Theta||_F^2
                  with the L2 term delegated to the optimiser's weight_decay
                  arg, per the standard Deep SVDD treatment (Ruff et al. 2018).
    Eq. 6      -> anomaly_scores: s(i) = ||h_i - c||^2.

The hypersphere centre c is computed once from the random-init forward pass
and HELD CONSTANT throughout training (Ruff 2018; AAGNN follows it).
"""

from __future__ import annotations

import numpy as np
import torch

from layer import AbnormalityAwareLayer


def _forward_no_grad(
    layer: AbnormalityAwareLayer,
    X: torch.Tensor,
    neigh_lists: list[list[int]],
) -> torch.Tensor:
    """Run the layer in eval-mode under no-grad, restoring train-mode after."""
    was_training = layer.training
    layer.eval()
    with torch.no_grad():
        H = layer(X, neigh_lists)
    if was_training:
        layer.train()
    return H


def compute_pseudo_labels(
    layer: AbnormalityAwareLayer,
    X: torch.Tensor,
    neigh_lists: list[list[int]],
    *,
    pseudo_label_pct: float = 50.0,
    train_val_split: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Algorithm 1 lines 1-5.

    Forward-pass through the random-init layer, take the p% closest-to-centre
    nodes as the pseudo-normal set S, split S into R (train) and D (val),
    and return T = V - S as the test set.

    Returns (R_idx, D_idx, T_idx, c) — sorted ascending int64 arrays plus a
    detached (out_dim,) tensor on the same device as ``X``.
    """
    if not (0.0 < pseudo_label_pct < 100.0):
        raise ValueError(
            f"pseudo_label_pct must be in (0, 100), got {pseudo_label_pct}"
        )
    if not (0.0 < train_val_split < 1.0):
        raise ValueError(
            f"train_val_split must be in (0, 1), got {train_val_split}"
        )

    n = X.shape[0]
    if n < 3:
        raise ValueError(
            f"Need at least 3 nodes to keep R, D, and T non-empty; got {n}."
        )
    H = _forward_no_grad(layer, X, neigh_lists)
    c = H.mean(dim=0).detach()
    d = ((H - c) ** 2).sum(dim=-1)

    # |S| = round(p% * n); clamp to [1, n-1] so R, D, T are all non-empty.
    s_size = int(round(pseudo_label_pct / 100.0 * n))
    s_size = max(1, min(n - 1, s_size))

    order = torch.argsort(d).cpu().numpy().astype(np.int64)
    S = order[:s_size]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(s_size)
    S_shuf = S[perm]

    r_size = int(round(train_val_split * s_size))
    r_size = max(1, min(s_size - 1, r_size))

    R_idx = np.sort(S_shuf[:r_size]).astype(np.int64)
    D_idx = np.sort(S_shuf[r_size:]).astype(np.int64)

    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True
    T_idx = np.sort(np.nonzero(~in_S)[0]).astype(np.int64)

    return R_idx, D_idx, T_idx, c


def train_aagnn(
    layer: AbnormalityAwareLayer,
    X: torch.Tensor,
    neigh_lists: list[list[int]],
    R_idx: np.ndarray,
    D_idx: np.ndarray,
    c: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    optimizer: str = "adam",
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Algorithm 1 lines 6-10.

    Minimise L_R = mean_{i in R} ||h_i - c||^2 (Eq. 5; Frobenius L2 handled by
    weight_decay), validating on D each epoch. Returns per-epoch losses.
    """
    if optimizer not in {"adam", "sgd"}:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
    if R_idx.size == 0:
        raise ValueError("R_idx is empty — nothing to train on.")
    if D_idx.size == 0:
        raise ValueError("D_idx is empty — nothing to validate on.")

    device = X.device
    c = c.detach().to(device)
    R_t = torch.as_tensor(R_idx, dtype=torch.long, device=device)
    D_t = torch.as_tensor(D_idx, dtype=torch.long, device=device)

    if optimizer == "adam":
        opt = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(layer.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    log_every = max(1, epochs // 10)

    for epoch in range(epochs):
        layer.train()
        H = layer(X, neigh_lists)
        loss = ((H[R_t] - c) ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))

        # Run a fresh forward pass for validation. The graph is small (a few
        # hundred nodes); this avoids retain_graph subtleties and gives a
        # clean train/eval mode split.
        H_val = _forward_no_grad(layer, X, neigh_lists)
        v = ((H_val[D_t] - c) ** 2).sum(dim=-1).mean()
        val_losses.append(float(v.item()))

        if verbose and (epoch < 5 or epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}"
            )

    return {"train_losses": train_losses, "val_losses": val_losses}


def anomaly_scores(
    layer: AbnormalityAwareLayer,
    X: torch.Tensor,
    neigh_lists: list[list[int]],
    c: torch.Tensor,
) -> np.ndarray:
    """Eq. 6 — squared L2 distance from each node's representation to ``c``.

    Higher = more anomalous. Returns a (n,) float64 array (sklearn AUC needs
    float).
    """
    H = _forward_no_grad(layer, X, neigh_lists)
    s = ((H - c.to(H.device)) ** 2).sum(dim=-1)
    return s.detach().cpu().numpy().astype(np.float64)
