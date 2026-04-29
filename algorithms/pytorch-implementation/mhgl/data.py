"""Data structures and synthetic-data generator for MHGL.

Reference: Zhou et al., "Unseen Anomaly Detection on Networks via
Multi-Hpersphere Learning", SIAM SDM 2022, §2 (problem definition) and §4
(datasets & evaluation protocol).

Defines:
    - AttributedNetwork: paper §2 G = (V, E, X) plus anomaly_type tag and
      V_train / V_test / labelled-subset masks for the seen/unseen protocol.
    - degree, _adjacency_lists: graph helpers shared with AAGNN's pattern.
    - build_normalized_adj: D^{-1/2}(A+I)D^{-1/2} as a torch sparse COO
      tensor (paper §3 Eq. 3.2 — Kipf & Welling 2017 GCN propagation rule).
    - SyntheticAttributedNetwork: SBM graph + clique-injected SEEN anomalies
      + feature-swap-injected UNSEEN anomalies + paper-protocol train/test
      partition driving example.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


# --- Defaults (paper §4.1 / synthetic-friendly) ---------------------------------

DEFAULT_FEATURE_DIM = 32
DEFAULT_N_NORMAL_COMMUNITIES = 5
DEFAULT_NODES_PER_NORMAL_COMMUNITY = 60   # 300 normal nodes total
DEFAULT_NODES_SEEN_COMMUNITY = 30         # one rare-class community for seen anomalies
DEFAULT_NODES_UNSEEN_COMMUNITY = 30       # one rare-class community for unseen anomalies
DEFAULT_Q_LABELED_SEEN = 20               # paper §4.2 RQ2 default
DEFAULT_NORMAL_LABEL_RATIO = 0.10         # paper §4.2 RQ2 default


# --- Anomaly type codes ---------------------------------------------------------

ANOM_NORMAL = 0
ANOM_SEEN = 1        # structural-clique injection — labelled at training time
ANOM_UNSEEN = 2      # contextual feature-swap — never appears in V_train


# --- Records --------------------------------------------------------------------

@dataclass(frozen=True)
class AttributedNetwork:
    """Paper §2: G = (V, E, X) with seen/unseen anomaly tagging and partition.

    V_train (those rows with train_mask=True) holds nodes whose binary anomaly
    label MAY be revealed; label_mask marks the subset that is actually
    revealed (q seen anomalies + p% normals per paper §4.2). V_test = ~train_mask
    contains the rest of seen anomalies + ALL unseen anomalies + held-out normals.
    """

    X: np.ndarray                  # (n, f) float32
    edges: np.ndarray              # (E, 2) int64 with i < j
    n: int
    f: int
    labels: np.ndarray | None = None         # (n,) int64 0/1 (binary anomaly)
    anomaly_type: np.ndarray | None = None   # (n,) int64 0=normal, 1=seen, 2=unseen
    train_mask: np.ndarray | None = None     # (n,) bool — node belongs to V_train
    label_mask: np.ndarray | None = None     # (n,) bool — label is revealed (subset of train_mask)

    def __post_init__(self) -> None:
        if self.X.shape != (self.n, self.f):
            raise ValueError(
                f"X shape {self.X.shape} does not match (n={self.n}, f={self.f})"
            )
        if self.X.dtype != np.float32:
            raise ValueError(f"X must be float32, got {self.X.dtype}")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be (E, 2), got {self.edges.shape}")
        if self.edges.dtype != np.int64:
            raise ValueError(f"edges must be int64, got {self.edges.dtype}")
        if self.edges.size:
            lo, hi = int(self.edges.min()), int(self.edges.max())
            if lo < 0 or hi >= self.n:
                raise ValueError(f"edge indices out of range [0, {self.n}): [{lo}, {hi}]")
            if (self.edges[:, 0] >= self.edges[:, 1]).any():
                raise ValueError("edges must satisfy i < j (undirected canonical form, no self-loops)")
        for name, arr, dtype in (
            ("labels", self.labels, np.int64),
            ("anomaly_type", self.anomaly_type, np.int64),
        ):
            if arr is not None:
                if arr.shape != (self.n,):
                    raise ValueError(f"{name} shape {arr.shape} != ({self.n},)")
                if arr.dtype != dtype:
                    raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")
        for name, arr in (("train_mask", self.train_mask), ("label_mask", self.label_mask)):
            if arr is not None:
                if arr.shape != (self.n,):
                    raise ValueError(f"{name} shape {arr.shape} != ({self.n},)")
                if arr.dtype != np.bool_:
                    raise ValueError(f"{name} dtype must be bool, got {arr.dtype}")
        if self.train_mask is not None and self.label_mask is not None:
            if (self.label_mask & ~self.train_mask).any():
                raise ValueError("label_mask must be a subset of train_mask")


# --- Graph helpers --------------------------------------------------------------

def degree(edges: np.ndarray, n: int) -> np.ndarray:
    """Node degrees as int64 array of shape (n,)."""
    deg = np.zeros(n, dtype=np.int64)
    if edges.size:
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
    return deg


def _adjacency_lists(edges: np.ndarray, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges.tolist():
        adj[i].append(j)
        adj[j].append(i)
    return adj


def build_normalized_adj(
    edges: np.ndarray,
    n: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute the symmetric-normalised propagation matrix A_hat = D^{-1/2}(A+I)D^{-1/2}.

    Paper §3 Eq. 3.2 / Kipf & Welling 2017 (ICLR). Returns a coalesced
    ``torch.sparse_coo_tensor`` with shape (n, n) and dtype float32, suitable
    for ``torch.sparse.mm`` inside a GCN layer.

    A is undirected, so for each canonical edge (i, j) with i < j we emit the
    pair (i, j) and (j, i). Self-loops are added explicitly.
    """
    if edges.size:
        i = edges[:, 0]
        j = edges[:, 1]
        rows = np.concatenate([i, j, np.arange(n, dtype=np.int64)])
        cols = np.concatenate([j, i, np.arange(n, dtype=np.int64)])
    else:
        rows = np.arange(n, dtype=np.int64)
        cols = np.arange(n, dtype=np.int64)

    # Degree of A + I (count entries per row).
    deg_hat = np.zeros(n, dtype=np.float64)
    np.add.at(deg_hat, rows, 1.0)
    inv_sqrt = np.where(deg_hat > 0, deg_hat ** -0.5, 0.0)
    vals = (inv_sqrt[rows] * inv_sqrt[cols]).astype(np.float32)

    indices = torch.from_numpy(np.stack([rows, cols], axis=0))
    values = torch.from_numpy(vals)
    A_hat = torch.sparse_coo_tensor(indices, values, size=(n, n), device=device)
    return A_hat.coalesce()


# --- Synthetic generator (paper §4.1 protocol: rare classes as anomalies) -------

@dataclass
class SyntheticAttributedNetwork:
    """SBM graph following paper §4.1 protocol — rare classes as anomalies.

    Paper §4.1: "we follow related works by taking one rare category as seen
    anomalies while considering the remained rare categories as unseen
    anomalies". Real datasets (Computer / Photo / CS) have 8–15 classes total
    where the smallest class is "seen", the next-smallest are "unseen".

    We mirror this with:
        - n_normal_communities clusters of normal nodes (default 5 × 60 = 300)
        - one ``seen-anomaly`` community with its own feature centroid and
          intra-community edge structure (default 30 nodes)
        - one ``unseen-anomaly`` community, ditto, with a *different* centroid
          (default 30 nodes)

    Each community gets a distinct random Gaussian feature centroid in R^f
    plus per-node noise. Both anomaly communities have the same SBM edge
    probabilities as normals — they're "anomalous" by virtue of feature
    centroid divergence and small size, not by structural perturbation. This
    mirrors how a GCN encoder discriminates rare classes on the paper's real
    co-purchase / co-author graphs: aggregate the neighbourhood, project to
    a latent space, observe that rare-class neighbourhoods don't align with
    any normal centroid.

    Train/test split per paper §2: q labelled seen anomalies + p% labelled
    normals enter V_train; the rest of the seen anomalies, ALL unseen
    anomalies, and held-out normals make up V_test.
    """

    n_normal_communities: int = DEFAULT_N_NORMAL_COMMUNITIES
    nodes_per_normal_community: int = DEFAULT_NODES_PER_NORMAL_COMMUNITY
    nodes_seen_community: int = DEFAULT_NODES_SEEN_COMMUNITY
    nodes_unseen_community: int = DEFAULT_NODES_UNSEEN_COMMUNITY
    feat_dim: int = DEFAULT_FEATURE_DIM
    p_in: float = 0.10
    p_out: float = 0.005
    centroid_scale: float = 2.5            # widens inter-community feature gap
    feature_noise_std: float = 0.3
    q_labeled_seen: int = DEFAULT_Q_LABELED_SEEN
    normal_label_ratio: float = DEFAULT_NORMAL_LABEL_RATIO
    seed: int = 0

    def generate(self) -> AttributedNetwork:
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        n_normal = self.n_normal_communities * self.nodes_per_normal_community
        n_seen = self.nodes_seen_community
        n_unseen = self.nodes_unseen_community
        n = n_normal + n_seen + n_unseen
        f = self.feat_dim
        n_communities_total = self.n_normal_communities + 2  # +seen +unseen

        if self.q_labeled_seen > n_seen:
            raise ValueError(
                f"q_labeled_seen={self.q_labeled_seen} > nodes_seen_community={n_seen}"
            )

        # 1. Community assignment: ids 0..n_normal_communities-1 are normals,
        #    id n_normal_communities = seen anomaly community,
        #    id n_normal_communities+1 = unseen anomaly community.
        seen_id = self.n_normal_communities
        unseen_id = self.n_normal_communities + 1
        sizes = (
            [self.nodes_per_normal_community] * self.n_normal_communities
            + [n_seen, n_unseen]
        )
        community = np.repeat(np.arange(n_communities_total), sizes)

        # 2. Distinct feature centroids per community + per-node noise.
        # ``centroid_scale`` widens the inter-community feature gap so a
        # randomly-initialised GCN encoder produces a clean clustering on
        # the first forward pass — important for MHGL because centres are
        # frozen at random-init time.
        centroids = np_rng.standard_normal(size=(n_communities_total, f)) * self.centroid_scale
        noise = np_rng.normal(0.0, self.feature_noise_std, size=(n, f))
        X = (centroids[community] + noise).astype(np.float32)

        # 3. SBM edges — same p_in / p_out for all communities (anomaly
        #    communities aren't structurally different, just smaller and with
        #    different features).
        edge_set: set[tuple[int, int]] = set()
        for i in range(n):
            ci = community[i]
            js = np.arange(i + 1, n)
            if js.size == 0:
                continue
            same = community[js] == ci
            probs = np.where(same, self.p_in, self.p_out)
            draws = np_rng.random(size=js.size)
            for j in js[draws < probs].tolist():
                edge_set.add((i, j))

        # 4. Labels + anomaly_type by community membership
        labels = np.zeros(n, dtype=np.int64)
        anomaly_type = np.zeros(n, dtype=np.int64)
        seen_arr = np.nonzero(community == seen_id)[0].astype(np.int64)
        unseen_arr = np.nonzero(community == unseen_id)[0].astype(np.int64)
        normal_arr = np.nonzero(community < self.n_normal_communities)[0].astype(np.int64)
        labels[seen_arr] = 1
        labels[unseen_arr] = 1
        anomaly_type[seen_arr] = ANOM_SEEN
        anomaly_type[unseen_arr] = ANOM_UNSEEN

        # 5. V_train / V_test partition (paper §2)
        train_mask = np.zeros(n, dtype=np.bool_)
        label_mask = np.zeros(n, dtype=np.bool_)

        seen_perm = list(seen_arr)
        rng.shuffle(seen_perm)
        labelled_seen = np.asarray(seen_perm[: self.q_labeled_seen], dtype=np.int64)
        train_mask[labelled_seen] = True
        label_mask[labelled_seen] = True

        normal_perm = list(normal_arr)
        rng.shuffle(normal_perm)
        n_norm_label = max(1, int(round(self.normal_label_ratio * len(normal_perm))))
        labelled_normal = np.asarray(normal_perm[:n_norm_label], dtype=np.int64)
        train_mask[labelled_normal] = True
        label_mask[labelled_normal] = True

        # All unseen anomalies stay out of V_train (paper §2: U ∩ V_train = ∅).

        # 6. Edges → canonical (E, 2) int64
        if edge_set:
            edges = np.asarray(sorted(edge_set), dtype=np.int64)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)

        return AttributedNetwork(
            X=X,
            edges=edges,
            n=n,
            f=f,
            labels=labels,
            anomaly_type=anomaly_type,
            train_mask=train_mask,
            label_mask=label_mask,
        )
