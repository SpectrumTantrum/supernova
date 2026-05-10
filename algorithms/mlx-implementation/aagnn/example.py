"""End-to-end AAGNN demo on a synthetic stochastic-block-model city.

Run:
    cd algorithms/pytorch-implementation/aagnn
    python example.py
    python example.py --aggregator attention
    python example.py --seed 7

Smoke-test contract (exits non-zero unless ALL hold):
    1. Both train and validation losses trend down.
    2. Welch's t-test on the score distributions: anomaly scores are
       significantly larger than normal scores (p < 0.05, mean-gap > 0).
       Paper anchor: AAGNN §3 establishes that anomaly representations
       are "mapped to faraway regions" relative to normals; the Welch
       test is the paper-faithful operationalization of that claim.

Note on the AUC contract: AAGNN paper Table 2 (CIKM 2021) reports
single point estimates per real dataset (BlogCatalog 0.8171–0.8184,
Flickr 0.8299–0.8456, PubMed 0.8459–0.8564) — no per-seed distribution,
no error bars. A per-seed numeric AUC floor is therefore unanchored;
removed per CONTEXT.md D-13 (paper-cited threshold rule) — see
.planning/phases/00-mlx-in-flight-cleanup/00-11-RESEARCH.md. AUC is
still computed and printed below as informational.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from data import SyntheticAttributedNetwork
from model import AAGNN, AAGNNConfig


def _short(losses: list[float]) -> str:
    """Format losses as ``[first3, ..., last3]`` for compact display."""
    if len(losses) <= 6:
        return "[" + ", ".join(f"{l:.4f}" for l in losses) + "]"
    head = ", ".join(f"{l:.4f}" for l in losses[:3])
    tail = ", ".join(f"{l:.4f}" for l in losses[-3:])
    return f"[{head}, ..., {tail}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregator", choices=["mean", "attention"], default="mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden-dim", type=int, default=64,
                    help="Smaller than paper's 256 to keep the smoke test fast on CPU; "
                         "SBM doesn't need full capacity.")
    args = ap.parse_args()

    print("=" * 64)
    print("AAGNN — synthetic SBM smoke test")
    print("=" * 64)

    print("\n[1/4] Generating synthetic SBM…")
    net = SyntheticAttributedNetwork(seed=args.seed).generate()
    n_anom = int(net.labels.sum()) if net.labels is not None else 0
    print(f"  Nodes:             {net.n:,}")
    print(f"  Edges:             {len(net.edges):,}")
    print(f"  Anomalies (true):  {n_anom:,}")
    print(f"  Communities:       {SyntheticAttributedNetwork().n_communities}")
    print(f"  Feature dim:       {net.f}")

    tag = "M" if args.aggregator == "mean" else "A"
    print(f"\n[2/4] Training AAGNN-{tag}…")
    cfg = AAGNNConfig(
        hidden_dim=args.hidden_dim,
        aggregator=args.aggregator,
        epochs=args.epochs,
        seed=args.seed,
        verbose=True,
    )
    model = AAGNN(cfg).fit(net)

    print("\n[3/4] Loss curves:")
    print(f"  Train losses: {_short(model.history['train_losses'])}")
    print(f"  Val losses:   {_short(model.history['val_losses'])}")

    print("\n[4/4] Evaluation:")
    scores = model.score()
    T = model.split_indices()["T"]
    assert net.labels is not None
    anom_in_T = int(net.labels[T].sum())
    auc_T = float(roc_auc_score(net.labels[T], scores[T]))

    anom_mask = net.labels == 1
    norm_mask = net.labels == 0
    mean_anom = float(scores[anom_mask].mean())
    mean_norm = float(scores[norm_mask].mean())
    t_stat, p_value = stats.ttest_ind(
        scores[anom_mask], scores[norm_mask], equal_var=False
    )
    t_stat = float(t_stat)
    p_value = float(p_value)

    print(f"  Test-set size:        |T| = {len(T)}, of which {anom_in_T} are true anomalies")
    print(f"  ROC-AUC on T:         {auc_T:.4f}  (informational — see module docstring)")
    print(f"  Score gap (anomaly - normal):  {mean_anom - mean_norm:+.4f}")
    print(f"  Welch's t-test:       t = {t_stat:+.3f}   p = {p_value:.2e}")

    train_losses = model.history["train_losses"]
    val_losses = model.history["val_losses"]
    train_loss_down = train_losses[-1] < train_losses[0]
    val_loss_down = val_losses[-1] < val_losses[0]
    gap_ok = mean_anom > mean_norm
    p_ok = p_value < 0.05

    if gap_ok and p_ok and train_loss_down and val_loss_down:
        print("\n✓ PASS — AAGNN distinguishes injected anomalies on the synthetic SBM.")
        return 0

    if not train_loss_down:
        reason = (f"train loss did not trend down "
                  f"(start={train_losses[0]:.4f}, end={train_losses[-1]:.4f})")
    elif not val_loss_down:
        reason = (f"validation loss did not trend down "
                  f"(start={val_losses[0]:.4f}, end={val_losses[-1]:.4f})")
    elif not gap_ok:
        reason = (f"score gap not positive "
                  f"(anom={mean_anom:.4f}, norm={mean_norm:.4f})")
    else:
        reason = f"Welch's p-value {p_value:.2e} >= 0.05"
    print(f"\n✗ FAIL — {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
