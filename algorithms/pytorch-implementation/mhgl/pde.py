"""Pattern Distribution Estimator (paper §3.1, Algorithm 1).

Reference: Zhou et al., "Unseen Anomaly Detection on Networks via
Multi-Hpersphere Learning", SIAM SDM 2022.

The PDE fits a Gaussian Mixture Model (paper Eq. 3.3) on the encoder output
of a candidate node set, assigns each node to its argmax-posterior component
(Eq. 3.4), and recursively re-splits any component that holds more than u
nodes. The output is a list of fine-grained ``Pattern`` records — each one a
disjoint slice of the candidate set with per-member posterior confidences.

The MHGL trainer calls ``fit_pde`` twice per fit:
    - once on labelled-normal node indices (k=k_normal, paper §4.2 default 10)
    - once on labelled-anomaly node indices (k=k_abnormal, default 2 in this
      module — q is small enough at training time that k=10 produces
      degenerate covariance estimates).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class Pattern:
    """One fine-grained pattern: a set of node indices and their GMM posteriors.

    ``indices`` is ascending int64. ``posteriors`` is the max-component
    posterior for each member (paper Eq. 3.4 ξ_j) — used by Algorithm 2 to
    select high-confidence members.
    """

    indices: np.ndarray         # (m,) int64
    posteriors: np.ndarray      # (m,) float32


def _fit_gmm(
    X: np.ndarray,
    n_components: int,
    *,
    seed: int,
) -> GaussianMixture:
    """Fit a GMM, falling back to simpler covariance types on numerical failure.

    'full' is paper-faithful but needs many samples per component for the
    feature dim; 'diag' tolerates lower sample-to-feature ratios; 'spherical'
    is the absolute fallback. We try them in order and take the first that
    converges without a singular-covariance warning.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if X.shape[0] < n_components:
        raise ValueError(
            f"GMM requires at least {n_components} samples, got {X.shape[0]}"
        )

    last_err: Exception | None = None
    for cov_type in ("full", "diag", "spherical"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=ConvergenceWarning)
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov_type,
                    random_state=seed,
                    reg_covar=1e-4,
                    max_iter=200,
                )
                gmm.fit(X)
                return gmm
        except (ConvergenceWarning, ValueError, FloatingPointError) as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"GMM failed under all covariance types (full / diag / spherical). "
        f"Last error: {last_err!r}"
    )


def fit_pde(
    H: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    k: int = 10,
    u: int = 30,
    max_recursion: int = 3,
    seed: int = 0,
) -> list[Pattern]:
    """Algorithm 1: hierarchical GMM-based pattern estimation.

    Args:
        H: (n, d) encoder output for ALL nodes (we only index into candidates,
            but indices stay global so downstream code can use them directly).
        candidate_indices: (m,) int64 — which rows of H to cluster (e.g.
            labelled-normal indices, or labelled-anomaly indices).
        k: initial GMM component count. Auto-clamped down if there are too
            few candidates to support k components.
        u: node-set-size threshold. Any resulting pattern with |S^i| > u is
            recursively re-fit, up to ``max_recursion`` levels deep.
        max_recursion: hard cap on recursion depth.
        seed: RNG seed for sklearn.

    Returns:
        Disjoint list of ``Pattern`` records covering ``candidate_indices``.
    """
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D, got shape {H.shape}")
    if candidate_indices.dtype != np.int64:
        raise ValueError(f"candidate_indices must be int64, got {candidate_indices.dtype}")
    if candidate_indices.size == 0:
        return []

    return _split(H, np.sort(candidate_indices), k=k, u=u, depth=0, max_depth=max_recursion, seed=seed)


def _split(
    H: np.ndarray,
    indices: np.ndarray,
    *,
    k: int,
    u: int,
    depth: int,
    max_depth: int,
    seed: int,
) -> list[Pattern]:
    """Recursive helper. ``indices`` is sorted int64.

    Algorithm 1 always fits a GMM with k components; the u-threshold only
    gates the *recursive* re-split of any resulting component bigger than u
    (lines 4-5). So we only short-circuit when k_eff collapses to 1 (too few
    samples to support more than one component meaningfully).
    """
    m = indices.shape[0]
    # Floor on samples-per-component: GMM needs at least n_components samples,
    # but ~3 per component is the practical lower bound for a meaningful fit.
    k_eff = max(1, min(k, m // 3))
    if k_eff <= 1:
        posteriors = np.ones(m, dtype=np.float32)
        return [Pattern(indices=indices, posteriors=posteriors)]

    H_sub = H[indices]
    gmm = _fit_gmm(H_sub, n_components=k_eff, seed=seed)
    resp = gmm.predict_proba(H_sub)               # (m, k_eff) — γ_ji
    assignments = np.argmax(resp, axis=1)         # Eq. 3.4
    posteriors = resp[np.arange(m), assignments]  # ξ_j

    out: list[Pattern] = []
    for comp in range(k_eff):
        mask = assignments == comp
        if not mask.any():
            continue
        sub_indices = indices[mask]
        sub_post = posteriors[mask].astype(np.float32)
        if sub_indices.size > u and depth + 1 < max_depth:
            # Algorithm 1 line 6: split number k' = |S^i| // u (with same u-floor).
            k_next = max(2, sub_indices.size // u)
            child_seed = seed + 1 + comp
            out.extend(
                _split(
                    H,
                    sub_indices,
                    k=k_next,
                    u=u,
                    depth=depth + 1,
                    max_depth=max_depth,
                    seed=child_seed,
                )
            )
        else:
            out.append(Pattern(indices=sub_indices, posteriors=sub_post))
    return out
