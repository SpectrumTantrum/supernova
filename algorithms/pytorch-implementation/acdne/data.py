"""Cross-network data structures and synthetic generator for ACDNE.

Reference: Shen, Dai, Chung, Lu, Choi — "Adversarial Deep Network Embedding
for Cross-network Node Classification", AAAI 2020.

Defines:
    - CrossNetwork  : frozen record holding (X, edges, labels) for a labelled
                      source network and an unlabelled target network.
    - ppmi_matrix   : K-step PPMI topological proximity (Levy & Goldberg
                      2014), used in paper Eq. 3 (FE2 input) and Eq. 5
                      (pairwise constraint).
    - neighbour_input : Eq. 3 — row-normalised PPMI-weighted aggregation of
                        attributes; this is the input to FE2.
    - SyntheticCrossNetwork : two SBMs sharing a community→label assignment
                              but with planted domain shift in attribute
                              statistics (analogous to paper §Datasets'
                              30% bit-flip corruption protocol).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Defaults ------------------------------------------------------------------

DEFAULT_K = 3                     # paper §Implementation Details: K-step = 3
DEFAULT_FEAT_DIM = 64
DEFAULT_N_CLASSES = 4
DEFAULT_NODES_SOURCE = 60         # per class -> 240 source nodes total
DEFAULT_NODES_TARGET = 50         # per class -> 200 target nodes total


# --- Records -------------------------------------------------------------------

@dataclass(frozen=True)
class CrossNetwork:
    """Paper §Problem Definition: labelled source G^s and unlabelled target G^t.

    Both networks share the same label categories (n_classes). No nodes or
    edges overlap across the two — they are entirely separate graphs that
    happen to live in a shared attribute space.

    y_t is held only for evaluation; `ACDNE.fit` never reads it.
    """

    X_s: np.ndarray             # (n_s, w) float32
    edges_s: np.ndarray         # (E_s, 2) int64, i < j
    y_s: np.ndarray             # (n_s,) int64, in [0, n_classes)
    X_t: np.ndarray             # (n_t, w) float32
    edges_t: np.ndarray         # (E_t, 2) int64, i < j
    y_t: np.ndarray | None      # (n_t,) int64, evaluation-only
    n_classes: int

    @property
    def n_s(self) -> int:
        return int(self.X_s.shape[0])

    @property
    def n_t(self) -> int:
        return int(self.X_t.shape[0])

    @property
    def feat_dim(self) -> int:
        return int(self.X_s.shape[1])

    def __post_init__(self) -> None:
        if self.X_s.dtype != np.float32 or self.X_t.dtype != np.float32:
            raise ValueError("X_s and X_t must be float32")
        if self.X_s.shape[1] != self.X_t.shape[1]:
            raise ValueError(
                f"feature dim mismatch: source {self.X_s.shape[1]} vs target {self.X_t.shape[1]}"
            )
        for tag, edges, n in [("source", self.edges_s, self.n_s), ("target", self.edges_t, self.n_t)]:
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(f"{tag} edges must be (E, 2), got {edges.shape}")
            if edges.dtype != np.int64:
                raise ValueError(f"{tag} edges must be int64, got {edges.dtype}")
            if edges.size:
                lo, hi = int(edges.min()), int(edges.max())
                if lo < 0 or hi >= n:
                    raise ValueError(
                        f"{tag} edge indices out of range [0, {n}): [{lo}, {hi}]"
                    )
                if (edges[:, 0] >= edges[:, 1]).any():
                    raise ValueError(f"{tag} edges must satisfy i < j")
        if self.y_s.shape != (self.n_s,) or self.y_s.dtype != np.int64:
            raise ValueError(f"y_s must be int64 with shape ({self.n_s},)")
        if self.y_s.min() < 0 or self.y_s.max() >= self.n_classes:
            raise ValueError(
                f"y_s values must lie in [0, {self.n_classes}); got [{self.y_s.min()}, {self.y_s.max()}]"
            )
        if self.y_t is not None:
            if self.y_t.shape != (self.n_t,) or self.y_t.dtype != np.int64:
                raise ValueError(f"y_t must be int64 with shape ({self.n_t},)")
            if self.y_t.min() < 0 or self.y_t.max() >= self.n_classes:
                raise ValueError(
                    f"y_t values must lie in [0, {self.n_classes})"
                )


# --- PPMI (paper Eq. 3 / Eq. 5 weights) ----------------------------------------

def _adjacency(edges: np.ndarray, n: int) -> np.ndarray:
    """Dense symmetric (n, n) float32 adjacency from a canonical (E, 2) edge list."""
    A = np.zeros((n, n), dtype=np.float32)
    if edges.size:
        i, j = edges[:, 0], edges[:, 1]
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def ppmi_matrix(edges: np.ndarray, n: int, K: int = DEFAULT_K) -> np.ndarray:
    """K-step PPMI proximity matrix used as a_ij in paper Eq. 3 and Eq. 5.

    Builds the K-step random-walk co-occurrence M and converts it to PPMI
    per Levy & Goldberg 2014 (cited by the paper §3.1):

        P     = D^{-1} A          (row-stochastic transition; isolated
                                   nodes are self-looped so D_ii > 0)
        M     = (1/K) sum_{k=1..K} P^k
        PMI   = log(M_ij * sum(M) / (sum_k M_ik * sum_k M_kj))
        PPMI  = max(PMI, 0)

    The diagonal is zeroed so that Eq. 3's exclusion of self (j != i) is
    enforced at the data level rather than relied upon downstream.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    A = _adjacency(edges, n)
    deg = A.sum(axis=1)
    # Self-loop isolated nodes so the row-normalisation stays well-defined.
    iso = deg == 0
    if iso.any():
        A = A.copy()
        idx = np.nonzero(iso)[0]
        A[idx, idx] = 1.0
        deg = A.sum(axis=1)

    P = A / deg[:, None]                 # (n, n)
    M = np.zeros_like(P, dtype=np.float64)
    Pk = np.eye(n, dtype=np.float64)
    P64 = P.astype(np.float64)
    for _ in range(K):
        Pk = Pk @ P64
        M += Pk
    M /= float(K)

    total = M.sum()
    row_sum = M.sum(axis=1, keepdims=True)
    col_sum = M.sum(axis=0, keepdims=True)
    # Outer product of marginals, with an epsilon to keep log finite.
    eps = 1e-12
    expected = (row_sum @ col_sum) / max(total, eps)
    ratio = np.divide(
        M,
        np.maximum(expected, eps),
        out=np.zeros_like(M),
        where=M > 0,
    )
    pmi = np.zeros_like(M)
    np.log(ratio, out=pmi, where=ratio > 0)
    ppmi = np.maximum(pmi, 0.0).astype(np.float32)
    np.fill_diagonal(ppmi, 0.0)
    return ppmi


def neighbour_input(X: np.ndarray, A_ppmi: np.ndarray) -> np.ndarray:
    """Paper Eq. 3 — n_i = sum_j (a_ij / sum_g a_ig) x_j (j, g != i).

    Diagonal is assumed already zero (`ppmi_matrix` enforces this).
    Nodes with all-zero PPMI rows (effectively isolated under K-step) get
    a zero neighbour vector, which is the right behaviour: FE2 then sees
    a zero input for those nodes.
    """
    if X.shape[0] != A_ppmi.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but PPMI is {A_ppmi.shape[0]}x{A_ppmi.shape[1]}"
        )
    row_sum = A_ppmi.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 0, row_sum, 1.0)
    weights = A_ppmi / safe                   # (n, n)
    return (weights @ X).astype(np.float32)


# --- Synthetic generator -------------------------------------------------------

@dataclass
class SyntheticCrossNetwork:
    """Two SBMs sharing community→label semantics but with planted domain shift.

    Source and target share class centroids `mu_c`. The target has an
    additional global mean offset and a per-class jitter sampled once;
    the result is two networks where attributes encode the same labels
    but the coordinate system is shifted — exactly the regime ACDNE is
    designed for. (Paper §Datasets corrupts attributes by random
    bit-flips; that's discrete-attribute-specific, so this Gaussian
    analogue stands in.)
    """

    n_classes: int = DEFAULT_N_CLASSES
    nodes_per_class_source: int = DEFAULT_NODES_SOURCE
    nodes_per_class_target: int = DEFAULT_NODES_TARGET
    feat_dim: int = DEFAULT_FEAT_DIM
    centroid_scale: float = 2.0          # separation between class centroids
    feature_noise_std: float = 0.5       # within-class spread
    p_in: float = 0.10                   # SBM intra-class edge probability
    p_out: float = 0.005                 # SBM inter-class edge probability
    domain_shift_strength: float = 0.6   # magnitude of target-side mean offset
    seed: int = 0

    def generate(self) -> CrossNetwork:
        rng = np.random.default_rng(self.seed)

        # Shared class centroids.
        mu = rng.standard_normal(size=(self.n_classes, self.feat_dim)).astype(np.float32)
        mu *= self.centroid_scale

        # Target-only domain shift: one global offset + small per-class jitter.
        d_global = (rng.standard_normal(size=(self.feat_dim,)) * self.domain_shift_strength).astype(np.float32)
        d_per_class = (
            rng.standard_normal(size=(self.n_classes, self.feat_dim))
            * (self.domain_shift_strength * 0.3)
        ).astype(np.float32)
        mu_t = mu + d_global[None, :] + d_per_class

        X_s, edges_s, y_s = self._build_one(
            rng, self.nodes_per_class_source, mu, name="source"
        )
        X_t, edges_t, y_t = self._build_one(
            rng, self.nodes_per_class_target, mu_t, name="target"
        )
        return CrossNetwork(
            X_s=X_s, edges_s=edges_s, y_s=y_s,
            X_t=X_t, edges_t=edges_t, y_t=y_t,
            n_classes=self.n_classes,
        )

    def _build_one(
        self,
        rng: np.random.Generator,
        nodes_per_class: int,
        mu: np.ndarray,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One SBM with class-conditional Gaussian attributes."""
        n = self.n_classes * nodes_per_class
        community = np.repeat(np.arange(self.n_classes), nodes_per_class)
        noise = rng.normal(0.0, self.feature_noise_std, size=(n, self.feat_dim))
        X = (mu[community] + noise).astype(np.float32)

        edge_set: set[tuple[int, int]] = set()
        for i in range(n):
            ci = community[i]
            js = np.arange(i + 1, n)
            if js.size == 0:
                continue
            same = community[js] == ci
            probs = np.where(same, self.p_in, self.p_out)
            draws = rng.random(size=js.size)
            for j in js[draws < probs].tolist():
                edge_set.add((i, j))

        if edge_set:
            edges = np.asarray(sorted(edge_set), dtype=np.int64)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
        y = community.astype(np.int64)
        return X, edges, y
