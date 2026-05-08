"""Adapter from the Amazon Photo NPZ to MHGL's :class:`AttributedNetwork`.

Implements paper §4.1's rare-class anomaly protocol against the real
co-purchase graph:

* Sort the 8 classes by ascending frequency.
* Smallest class -> unseen anomaly (label=1, anomaly_type=2, NEVER in V_train).
* Second-smallest class -> seen anomaly (label=1, anomaly_type=1); randomly
  pick ``q_labeled_seen=20`` of these to enter ``train_mask`` and
  ``label_mask``.
* Remaining 6 classes -> normals (label=0, anomaly_type=0); randomly pick
  ``normal_label_ratio=0.10`` of them for ``train_mask`` and ``label_mask``.

Edges are returned in canonical undirected form (i < j, deduplicated)
after symmetrising the raw directed CSR adjacency.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

# Import shim — put the parent mhgl/ on sys.path so ``from data import …`` works
# regardless of where this module is invoked from. ``parents[2]`` resolves to
# .../mhgl from .../mhgl/datasets/amazon_photo/loader.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import ANOM_NORMAL, ANOM_SEEN, ANOM_UNSEEN, AttributedNetwork  # noqa: E402


DEFAULT_NPZ = pathlib.Path(__file__).resolve().parent / "raw" / "amazon_electronics_photo.npz"

DEFAULT_Q_LABELED_SEEN = 20
DEFAULT_NORMAL_LABEL_RATIO = 0.10


def _load_npz(path: pathlib.Path) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Amazon Photo NPZ not found at {path}. Run `python fetch.py` first."
        )
    data = np.load(path, allow_pickle=False)
    X = sp.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    A = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    labels = np.asarray(data["labels"]).astype(np.int64, copy=False)
    return X, A, labels


def _canonical_edges(A: sp.csr_matrix) -> np.ndarray:
    """Symmetrise A and return undirected canonical edges (E, 2) int64, i < j."""
    A_sym = A + A.T  # adds entries; we only care about non-zero pattern
    A_sym = A_sym.tocoo()
    keep = A_sym.row < A_sym.col  # drops self-loops AND lower triangle
    rows = A_sym.row[keep].astype(np.int64, copy=False)
    cols = A_sym.col[keep].astype(np.int64, copy=False)
    pairs = np.stack([rows, cols], axis=1)
    if pairs.size:
        # ``np.unique`` along axis=0 sorts lexicographically and dedupes.
        pairs = np.unique(pairs, axis=0)
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
    return pairs


def _class_order_by_frequency(labels: np.ndarray) -> np.ndarray:
    """Return class ids sorted by ascending count, ties broken by ascending class id."""
    counts = np.bincount(labels)
    classes = np.arange(counts.size)
    # lexsort: last key is primary -> sort by count, ties by class id.
    order = np.lexsort((classes, counts))
    return order.astype(np.int64, copy=False)


def load_amazon_photo(
    npz_path: pathlib.Path | str = DEFAULT_NPZ,
    *,
    q_labeled_seen: int = DEFAULT_Q_LABELED_SEEN,
    normal_label_ratio: float = DEFAULT_NORMAL_LABEL_RATIO,
    seed: int = 0,
) -> AttributedNetwork:
    """Load Amazon Photo and apply paper §4.1's rare-class anomaly protocol."""
    npz_path = pathlib.Path(npz_path)
    X_sp, A_sp, raw_labels = _load_npz(npz_path)

    n, f = X_sp.shape
    X = X_sp.toarray().astype(np.float32, copy=False)

    edges = _canonical_edges(A_sp)

    order = _class_order_by_frequency(raw_labels)
    if order.size < 3:
        raise ValueError(
            f"Amazon Photo expected >=3 classes for the rare-class protocol, got {order.size}"
        )
    unseen_class = int(order[0])
    seen_class = int(order[1])
    normal_classes = set(int(c) for c in order[2:])

    rng = np.random.default_rng(seed)

    labels = np.zeros(n, dtype=np.int64)
    anomaly_type = np.full(n, ANOM_NORMAL, dtype=np.int64)
    train_mask = np.zeros(n, dtype=np.bool_)
    label_mask = np.zeros(n, dtype=np.bool_)

    unseen_idx = np.nonzero(raw_labels == unseen_class)[0].astype(np.int64)
    seen_idx = np.nonzero(raw_labels == seen_class)[0].astype(np.int64)
    normal_idx = np.nonzero(np.isin(raw_labels, list(normal_classes)))[0].astype(np.int64)

    labels[unseen_idx] = 1
    anomaly_type[unseen_idx] = ANOM_UNSEEN
    # Unseen anomalies must NEVER appear in V_train (paper §2: U ∩ V_train = ∅).

    labels[seen_idx] = 1
    anomaly_type[seen_idx] = ANOM_SEEN
    if q_labeled_seen > seen_idx.size:
        raise ValueError(
            f"q_labeled_seen={q_labeled_seen} > available seen anomalies ({seen_idx.size})"
        )
    labelled_seen = rng.choice(seen_idx, size=q_labeled_seen, replace=False)
    train_mask[labelled_seen] = True
    label_mask[labelled_seen] = True

    n_norm_label = max(1, int(round(normal_label_ratio * normal_idx.size)))
    labelled_normal = rng.choice(normal_idx, size=n_norm_label, replace=False)
    train_mask[labelled_normal] = True
    label_mask[labelled_normal] = True

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=pathlib.Path, default=DEFAULT_NPZ)
    ap.add_argument("--q-labeled-seen", type=int, default=DEFAULT_Q_LABELED_SEEN)
    ap.add_argument("--normal-label-ratio", type=float, default=DEFAULT_NORMAL_LABEL_RATIO)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = load_amazon_photo(
        args.npz,
        q_labeled_seen=args.q_labeled_seen,
        normal_label_ratio=args.normal_label_ratio,
        seed=args.seed,
    )
    assert net.labels is not None
    assert net.anomaly_type is not None
    assert net.train_mask is not None
    assert net.label_mask is not None

    print(f"n              = {net.n}")
    print(f"f              = {net.f}")
    print(f"|E| (undir)    = {len(net.edges)}")
    print(f"label_mask sum = {int(net.label_mask.sum())}")
    print(f"train_mask sum = {int(net.train_mask.sum())}")
    print(f"seen anom (1)  = {int((net.anomaly_type == ANOM_SEEN).sum())}")
    print(f"unseen anom(2) = {int((net.anomaly_type == ANOM_UNSEEN).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
