"""Adapter: ACDNE-paper .mat files -> ``CrossNetwork`` (DBLPv7 -> ACMv9).

Source: Shen et al. AAAI 2020. The two .mat files share a 6,775-token
bag-of-words vocabulary derived from paper titles, plus a five-class
research-area taxonomy (DB / DM / AI / CV / IR).

Each .mat file holds:
    attrb   (n, w) uint8 — bag-of-words counts per node
    group   (n, c) uint8 — one-hot (occasionally multi-hot) class labels
    network (n, n) sparse COO float64 — undirected, symmetric adjacency

The original labels are multi-label (≈1.5% of DBLP and ≈6.2% of ACM nodes
carry two classes). ACDNE's ``CrossNetwork`` is single-label, so we collapse
via ``argmax`` along the class axis — the same convention the paper authors'
own training code uses for F1 evaluation.

CLI:
    python loader.py    # prints summary statistics for both networks
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

# Make the parent ``acdne/`` package importable so we can reach
# ``data.CrossNetwork`` without restructuring the repo.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import CrossNetwork  # noqa: E402


RAW_DIR = pathlib.Path(__file__).resolve().parent / "raw"
SOURCE_FILE = "dblpv7.mat"
TARGET_FILE = "acmv9.mat"


def _load_one_mat(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, edges, y) from a single ACDNE-paper .mat file.

    - ``X``:    (n, w) float32 dense bag-of-words attributes.
    - ``edges``:(E, 2) int64 with i < j, deduplicated, no self-loops.
    - ``y``:    (n,) int64 single-class labels via argmax over the
                paper's one/multi-hot label matrix.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run fetch.py first to download the .mat files."
        )

    mat = loadmat(str(path))
    for key in ("attrb", "group", "network"):
        if key not in mat:
            raise KeyError(
                f"{path.name} missing expected key {key!r}; got keys "
                f"{[k for k in mat if not k.startswith('__')]}"
            )

    attrb = mat["attrb"]
    group = mat["group"]
    network = mat["network"]

    n_attr = attrb.shape[0]
    n_lbl = group.shape[0]
    n_net = network.shape[0]
    if not (n_attr == n_lbl == n_net):
        raise ValueError(
            f"{path.name}: row-count mismatch — attrb={n_attr}, "
            f"group={n_lbl}, network={n_net}×{network.shape[1]}"
        )
    if network.shape[0] != network.shape[1]:
        raise ValueError(f"{path.name}: network must be square, got {network.shape}")

    X = np.ascontiguousarray(attrb, dtype=np.float32)

    # Multi-label rows exist (~1.5% DBLP, ~6.2% ACM). argmax matches the
    # paper authors' own F1 evaluation. Reject all-zero rows so we never
    # silently send unlabelled nodes to class 0.
    label_sums = np.asarray(group.sum(axis=1)).ravel()
    if (label_sums == 0).any():
        n_zero = int((label_sums == 0).sum())
        raise ValueError(
            f"{path.name}: {n_zero} nodes have no label — refusing to mask "
            "silently. Open an issue if upstream data has changed."
        )
    y = np.asarray(group).argmax(axis=1).astype(np.int64)

    # Canonical undirected edge list. ``network`` is symmetric in upstream
    # files; ``triu(k=1)`` already enforces i<j and discards self-loops.
    coo = sp.triu(sp.coo_matrix(network), k=1).tocoo()
    if coo.nnz == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
    else:
        edges = np.stack([coo.row.astype(np.int64), coo.col.astype(np.int64)], axis=1)
        # Defensive: dedupe in case upstream stored duplicate entries.
        edges = np.unique(edges, axis=0)

    return X, edges, y


def load_dblp_acmv9(
    raw_dir: pathlib.Path | str = RAW_DIR,
) -> CrossNetwork:
    """Build the DBLPv7 (source) -> ACMv9 (target) cross-network.

    The two .mat files share the same 6,775-token vocabulary. We assert
    that explicitly so downstream feature-dim errors surface here with a
    clear message rather than deep inside the model.
    """
    raw = pathlib.Path(raw_dir)
    X_s, edges_s, y_s = _load_one_mat(raw / SOURCE_FILE)
    X_t, edges_t, y_t = _load_one_mat(raw / TARGET_FILE)

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            f"feature-dim mismatch: source has w={X_s.shape[1]}, "
            f"target has w={X_t.shape[1]} — both .mat files must share "
            "the same vocabulary."
        )

    classes = np.union1d(np.unique(y_s), np.unique(y_t))
    n_classes = int(classes.max() + 1)
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError(
            f"label set {classes.tolist()} is not contiguous starting at 0"
        )

    return CrossNetwork(
        X_s=X_s, edges_s=edges_s, y_s=y_s,
        X_t=X_t, edges_t=edges_t, y_t=y_t,
        n_classes=n_classes,
    )


def _hist(y: np.ndarray, n_classes: int) -> list[int]:
    return np.bincount(y, minlength=n_classes).tolist()


def main() -> int:
    net = load_dblp_acmv9()
    print("DBLPv7 (source) -> ACMv9 (target)")
    print(f"  n_s        = {net.n_s}")
    print(f"  n_t        = {net.n_t}")
    print(f"  feat_dim   = {net.feat_dim}")
    print(f"  n_classes  = {net.n_classes}")
    print(f"  edges_s    = {len(net.edges_s)}")
    print(f"  edges_t    = {len(net.edges_t)}")
    print(f"  label hist (source) = {_hist(net.y_s, net.n_classes)}")
    assert net.y_t is not None
    print(f"  label hist (target) = {_hist(net.y_t, net.n_classes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
