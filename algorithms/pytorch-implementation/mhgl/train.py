"""Multi-hypersphere training loop for MHGL (paper §3.2 Eqs. 3.5-3.8, Algorithm 2).

Reference: Zhou et al., "Unseen Anomaly Detection on Networks via
Multi-Hpersphere Learning", SIAM SDM 2022.

Implements Algorithm 2 end-to-end:
    Lines 1-7:  initialise centres + radii + high-confidence sets H^i.
                Centres c_i are computed ONCE from the random-init forward
                pass (Eq. 3.5) and FROZEN across all T epochs — only Theta
                (encoder weights) is updated. This matches the Deep-SVDD
                discipline AAGNN follows.
    Lines 8-15: per epoch, sample alpha * |H^i| mixup pseudo-labels via
                Eq. 3.8 inside each H^i, compute the multi-hypersphere
                objective (Eq. 3.6), step Adam.
    Eq. 3.6:    sum_i mean_{j in D_i} ||h_j - c_i||^2  (real members + mixups)
                + (sigma / (q*p)) sum_{r,i} (||h_r - c_i||^2 + eps)^{-1}
                The L2 / Frobenius regulariser is delegated to the optimiser's
                weight_decay arg per the standard treatment. ``eps`` guards
                against gradient blow-up at init when an anomaly happens to
                land on a centre.
    Eq. 3.7:    s(v_j) = min_i ||h_j - c_i||^2 over normal centres only.

Note on the abnormal PDE call: Algorithm 2 line 2 mentions running PDE on
abnormal nodes too, but Eq. 3.6 only uses normal centres (the second term
uses labelled-anomaly raw indices, not abnormal patterns). To match the loss
equation exactly we run PDE only on labelled normals; labelled anomalies
enter the loss as raw node indices.
"""

from __future__ import annotations

import numpy as np
import torch

from gcn import GCNEncoder
from pde import Pattern


# --- Forward helper -------------------------------------------------------------

def _forward_no_grad(
    encoder: GCNEncoder,
    X: torch.Tensor,
    A_hat: torch.Tensor,
) -> torch.Tensor:
    """Run the encoder under no_grad in inference mode, restoring train mode after."""
    was_training = encoder.training
    encoder.train(False)
    with torch.no_grad():
        H = encoder(X, A_hat)
    if was_training:
        encoder.train(True)
    return H


# --- Eq. 3.5 — centres ----------------------------------------------------------

def compute_centres(
    H: torch.Tensor,
    patterns: list[Pattern],
) -> torch.Tensor:
    """Eq. 3.5: c_i = mean(H[indices_i]). Returns (p, d) tensor on H.device.

    Centres are computed ONCE after PDE fitting and HELD CONSTANT across all
    training epochs (Algorithm 2 line 3 vs lines 14-15: only Theta updates).
    """
    if not patterns:
        raise ValueError("patterns is empty — nothing to compute centres for")
    centres = []
    for p in patterns:
        idx = torch.as_tensor(p.indices, dtype=torch.long, device=H.device)
        centres.append(H[idx].mean(dim=0))
    return torch.stack(centres, dim=0).detach()


# --- Algorithm 2 lines 4-7 — high-confidence sets -------------------------------

def compute_high_confidence(
    H: torch.Tensor,
    pattern: Pattern,
    centre: torch.Tensor,
    *,
    threshold_t: float,
    radius_quantile: float = 1.0,
) -> np.ndarray:
    """Algorithm 2 lines 4-7 for one pattern.

    F^i = {v_j ∈ S^i : ξ_j > t}                     (line 5: posterior gate)
    r_i = quantile_{v_j ∈ F^i} d_euc(F^i_j, c_i)    (line 6: radius)
    H^i = {v_j ∈ V : d_euc(c_i, h_j) ≤ r_i}         (line 7: full membership)

    The paper specifies r_i = max distance (line 6); ``radius_quantile=1.0``
    matches that exactly. Lower quantiles tighten the hypersphere and reduce
    the chance of capturing unlabelled-anomaly nodes when the encoder is
    randomly initialised — a practical issue on small synthetic graphs where
    one labelled-normal can be a clustering outlier.

    Falls back to S^i itself if F^i is empty (no member clears the threshold)
    so we never produce a zero-radius hypersphere.
    """
    if not (0.0 < radius_quantile <= 1.0):
        raise ValueError(f"radius_quantile must be in (0, 1], got {radius_quantile}")

    members = pattern.indices
    posteriors = pattern.posteriors

    high_post_mask = posteriors > threshold_t
    F_indices = members[high_post_mask] if high_post_mask.any() else members

    F_t = torch.as_tensor(F_indices, dtype=torch.long, device=H.device)
    F_dists = torch.linalg.norm(H[F_t] - centre, dim=-1)
    if radius_quantile >= 1.0:
        r_i = F_dists.max().item()
    else:
        r_i = float(torch.quantile(F_dists, radius_quantile).item())

    # Distance from every node in V to centre (no_grad: this is just a filter).
    with torch.no_grad():
        all_dists = torch.linalg.norm(H - centre, dim=-1)
    H_mask = (all_dists <= r_i).cpu().numpy()
    return np.asarray(np.nonzero(H_mask)[0], dtype=np.int64)


# --- Eq. 3.8 — mixup pseudo-labels ----------------------------------------------

def mixup_pseudo_labels(
    H: torch.Tensor,
    high_conf_indices: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Eq. 3.8: h_new = (1-β)·h_a + β·h_b with β ~ U(0,1).

    Pairs (a, b) are sampled WITH replacement from ONE pattern's H^i — never
    cross-pollinated across patterns (the pseudo-label is meant to belong to
    this pattern's centre, not someone else's).

    Returns (n_samples, d) tensor sharing gradient with ``H`` so backprop
    flows through the encoder for the mixed-up vectors.
    """
    m = high_conf_indices.shape[0]
    if m < 2 or n_samples <= 0:
        return torch.empty((0, H.shape[1]), device=H.device, dtype=H.dtype)

    a = rng.integers(0, m, size=n_samples)
    b = rng.integers(0, m, size=n_samples)
    a_idx = torch.as_tensor(high_conf_indices[a], dtype=torch.long, device=H.device)
    b_idx = torch.as_tensor(high_conf_indices[b], dtype=torch.long, device=H.device)
    beta = torch.as_tensor(rng.random(size=n_samples), dtype=H.dtype, device=H.device).unsqueeze(-1)
    return (1.0 - beta) * H[a_idx] + beta * H[b_idx]


# --- Eq. 3.6 — multi-hypersphere loss -------------------------------------------

def mhgl_loss(
    H: torch.Tensor,                           # (n, d) — live encoder output
    centres: torch.Tensor,                     # (p, d) — frozen normal centres
    high_conf: list[np.ndarray],               # H^i per normal pattern
    mixup_per_pattern: list[torch.Tensor],     # h_new per normal pattern (live)
    anom_indices: np.ndarray,                  # labelled-anomaly node indices
    *,
    sigma: float,
    eps: float,
) -> torch.Tensor:
    """Eq. 3.6 (Frobenius L2 term delegated to optimiser weight_decay):

        loss = (1/p) * mean_i [ mean_{j in D_i} ||h_j - c_i||^2 ]
             + (sigma / (q*p)) * sum_{r in anom} sum_{i in 1..p} (||h_r - c_i||^2 + eps)^{-1}

    The first term is the per-pattern *mean* squared distance over D_i (real
    H^i members + mixup pseudo-labels), then averaged over patterns. This is
    a faithful but stable reading of the paper's "(1/(p*n))" prefactor —
    weighting per-pattern means equally regardless of |H^i| imbalance.
    """
    p = centres.shape[0]
    if p == 0:
        raise ValueError("centres must have at least one row")
    if len(high_conf) != p or len(mixup_per_pattern) != p:
        raise ValueError(
            f"high_conf ({len(high_conf)}) and mixup_per_pattern "
            f"({len(mixup_per_pattern)}) must each have length p={p}"
        )

    # First term — contraction.
    contraction_terms = []
    for i in range(p):
        c_i = centres[i]
        members_real = high_conf[i]
        mixup_i = mixup_per_pattern[i]
        chunks = []
        if members_real.size > 0:
            idx = torch.as_tensor(members_real, dtype=torch.long, device=H.device)
            chunks.append(H[idx])
        if mixup_i.shape[0] > 0:
            chunks.append(mixup_i)
        if not chunks:
            continue
        D_i = torch.cat(chunks, dim=0)  # (n_i, d)
        contraction_terms.append(((D_i - c_i) ** 2).sum(dim=-1).mean())
    if not contraction_terms:
        contraction = H.new_zeros(())
    else:
        contraction = torch.stack(contraction_terms).mean()

    # Second term — repulsion (only if there are labelled anomalies).
    if anom_indices.size > 0:
        anom_t = torch.as_tensor(anom_indices, dtype=torch.long, device=H.device)
        H_anom = H[anom_t]                                    # (q, d)
        diff = H_anom.unsqueeze(1) - centres.unsqueeze(0)     # (q, p, d)
        d2 = (diff ** 2).sum(dim=-1)                           # (q, p)
        repulsion = (1.0 / (d2 + eps)).mean()                  # mean over q*p pairs
    else:
        repulsion = H.new_zeros(())

    return contraction + sigma * repulsion


# --- Algorithm 2 outer loop -----------------------------------------------------

def train_mhgl(
    encoder: GCNEncoder,
    X: torch.Tensor,
    A_hat: torch.Tensor,
    centres: torch.Tensor,
    high_conf: list[np.ndarray],
    anom_indices: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    sigma: float,
    eps: float,
    augmentation_alpha: int,
    seed: int,
    verbose: bool,
) -> dict[str, list[float]]:
    """Algorithm 2 lines 8-16. Returns per-epoch train / val losses.

    Validation is a single-shot loss recomputation under no_grad on the same
    (H^i, anomaly) split — the synthetic graph is small and we don't carve
    out a separate held-out fold mid-training because Algorithm 2 doesn't
    define one (R/D split is AAGNN-specific, not MHGL).
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if augmentation_alpha < 0:
        raise ValueError(f"augmentation_alpha must be >= 0, got {augmentation_alpha}")

    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    log_every = max(1, epochs // 10)

    for epoch in range(epochs):
        encoder.train(True)
        H = encoder(X, A_hat)

        mixup_per_pattern = [
            mixup_pseudo_labels(H, hi, augmentation_alpha * hi.shape[0], rng)
            for hi in high_conf
        ]
        loss = mhgl_loss(
            H, centres, high_conf, mixup_per_pattern, anom_indices,
            sigma=sigma, eps=eps,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))

        # Validation: recompute the loss on a fresh forward pass with a fresh
        # mixup draw, no_grad. This is just a "is the loss surface stable"
        # signal, not a held-out generalisation metric.
        H_val = _forward_no_grad(encoder, X, A_hat)
        val_mixup = [
            mixup_pseudo_labels(H_val, hi, augmentation_alpha * hi.shape[0], rng)
            for hi in high_conf
        ]
        with torch.no_grad():
            v = mhgl_loss(
                H_val, centres, high_conf, val_mixup, anom_indices,
                sigma=sigma, eps=eps,
            )
        val_losses.append(float(v.item()))

        if verbose and (epoch < 5 or epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}"
            )

    return {"train_losses": train_losses, "val_losses": val_losses}


# --- Eq. 3.7 — anomaly scoring --------------------------------------------------

def anomaly_scores(
    encoder: GCNEncoder,
    X: torch.Tensor,
    A_hat: torch.Tensor,
    centres: torch.Tensor,
) -> np.ndarray:
    """Eq. 3.7: s(v_j) = min_i ||h_j - c_i||^2 over normal centres only.

    Returns float64 (sklearn ROC-AUC needs float).
    """
    H = _forward_no_grad(encoder, X, A_hat)
    diff = H.unsqueeze(1) - centres.to(H.device).unsqueeze(0)   # (n, p, d)
    d2 = (diff ** 2).sum(dim=-1)                                 # (n, p)
    s, _ = d2.min(dim=-1)                                        # (n,)
    return s.detach().cpu().numpy().astype(np.float64)
