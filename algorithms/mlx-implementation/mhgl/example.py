"""End-to-end MHGL demo on a synthetic SBM with seen + unseen anomalies.

Run:
    cd algorithms/pytorch-implementation/mhgl
    python example.py
    python example.py --epochs 300                   # paper-fidelity training length
    python example.py --hidden-dims 256,128,64,32    # paper-fidelity encoder

Smoke-test contract (exits non-zero unless ALL hold):
    1. Train loss trends down.
    2. ROC-AUC on V_test ≥ AUC_FLOOR (default 0.75).
    3. Welch's t-test on V_test scores: anomaly > normal, p < 0.05.
    4. Welch's t-test on V_test restricted to {normals, unseen-only}:
       p < 0.05 with positive mean-gap. This statistically gates the
       paper's main claim that MHGL detects unseen anomaly types, without
       baking in a brittle threshold-tuned AUC floor.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from data import ANOM_UNSEEN, SyntheticAttributedNetwork
from model import MHGL, MHGLConfig


AUC_FLOOR = 0.75


def _short(losses: list[float]) -> str:
    """Format losses as ``[first3, ..., last3]`` for compact display."""
    if len(losses) <= 6:
        return "[" + ", ".join(f"{l:.4f}" for l in losses) + "]"
    head = ", ".join(f"{l:.4f}" for l in losses[:3])
    tail = ", ".join(f"{l:.4f}" for l in losses[-3:])
    return f"[{head}, ..., {tail}]"


def _parse_dims(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80,
                    help="Smaller than paper's 300 to keep the smoke test fast on CPU.")
    ap.add_argument("--hidden-dims", type=_parse_dims, default=(64, 32, 16, 16),
                    help="Comma-separated GCN layer widths. Paper uses 256,128,64,32.")
    ap.add_argument("--k-normal", type=int, default=3,
                    help="GMM components for PDE. Paper uses 10 on graphs of 7k-18k nodes. "
                         "Our synthetic has ~30 labelled normals; k=3 keeps ~10 nodes per "
                         "GMM component (above the n_features-rank-deficient floor for diag "
                         "covariance) while still producing fine-grained patterns.")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--radius-quantile", type=float, default=0.25,
                    help="Quantile of F^i distances used as r_i. Paper uses 1.0 (max); "
                         "0.25 tightens H^i so unlabelled-anomaly nodes are less "
                         "likely to be captured at random-init time.")
    ap.add_argument("--auc-floor", type=float, default=AUC_FLOOR)
    args = ap.parse_args()

    print("=" * 64)
    print("MHGL — synthetic SBM smoke test (seen + unseen anomalies)")
    print("=" * 64)

    print("\n[1/4] Generating synthetic SBM…")
    net = SyntheticAttributedNetwork(seed=args.seed).generate()
    assert net.labels is not None and net.anomaly_type is not None
    assert net.train_mask is not None and net.label_mask is not None
    n_seen = int((net.anomaly_type == 1).sum())
    n_unseen = int((net.anomaly_type == 2).sum())
    n_train = int(net.train_mask.sum())
    n_label = int(net.label_mask.sum())
    print(f"  Nodes:                    {net.n:,}")
    print(f"  Edges:                    {len(net.edges):,}")
    print(f"  Feature dim:              {net.f}")
    print(f"  Seen anomalies (total):   {n_seen}")
    print(f"  Unseen anomalies (total): {n_unseen}")
    print(f"  V_train size:             {n_train}  ({n_label} labels revealed)")

    print("\n[2/4] Training MHGL…")
    cfg = MHGLConfig(
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        k_normal=args.k_normal,
        sigma=args.sigma,
        radius_quantile=args.radius_quantile,
        seed=args.seed,
        verbose=True,
    )
    model = MHGL(cfg).fit(net)

    print("\n[3/4] Loss curves:")
    print(f"  Train losses: {_short(model.history['train_losses'])}")
    print(f"  Val losses:   {_short(model.history['val_losses'])}")

    print("\n[4/4] Evaluation:")
    scores = model.score()
    test_mask = ~net.train_mask
    test_idx = np.nonzero(test_mask)[0]
    test_labels = net.labels[test_mask]
    test_scores = scores[test_mask]

    auc_overall = float(roc_auc_score(test_labels, test_scores))

    test_anom_type = net.anomaly_type[test_mask]
    is_normal = test_labels == 0
    is_anom = test_labels == 1
    is_unseen = test_anom_type == ANOM_UNSEEN

    mean_anom = float(test_scores[is_anom].mean())
    mean_norm = float(test_scores[is_normal].mean())
    t_all, p_all = stats.ttest_ind(
        test_scores[is_anom], test_scores[is_normal], equal_var=False
    )
    t_all = float(t_all)
    p_all = float(p_all)

    if is_unseen.any():
        mean_unseen = float(test_scores[is_unseen].mean())
        t_unseen, p_unseen = stats.ttest_ind(
            test_scores[is_unseen], test_scores[is_normal], equal_var=False
        )
        t_unseen = float(t_unseen)
        p_unseen = float(p_unseen)
        # AUC on the {normal, unseen-only} subset — informational, not gated.
        unseen_subset = is_normal | is_unseen
        auc_unseen = float(roc_auc_score(test_labels[unseen_subset], test_scores[unseen_subset]))
    else:
        mean_unseen = float("nan")
        t_unseen = float("nan")
        p_unseen = float("nan")
        auc_unseen = float("nan")

    print(f"  Test-set size:                |V_test| = {len(test_idx)}")
    print(f"  ROC-AUC on V_test:            {auc_overall:.4f}")
    print(f"  ROC-AUC on (V_test ∩ unseen): {auc_unseen:.4f}  (informational)")
    print(f"  Score gap (anom - normal):    {mean_anom - mean_norm:+.4f}")
    print(f"  Welch all-anom vs normal:     t = {t_all:+.3f}   p = {p_all:.2e}")
    print(f"  Welch unseen vs normal:       t = {t_unseen:+.3f}   p = {p_unseen:.2e}")

    train_losses = model.history["train_losses"]
    loss_down = train_losses[-1] < train_losses[0]
    auc_ok = auc_overall >= args.auc_floor
    gap_ok = mean_anom > mean_norm
    p_all_ok = p_all < 0.05
    p_unseen_ok = (
        is_unseen.any()
        and p_unseen < 0.05
        and mean_unseen > mean_norm
    )

    if loss_down and auc_ok and gap_ok and p_all_ok and p_unseen_ok:
        print("\n✓ PASS — MHGL distinguishes seen + unseen anomalies on the synthetic SBM.")
        return 0

    if not loss_down:
        reason = (f"train loss did not trend down "
                  f"(start={train_losses[0]:.4f}, end={train_losses[-1]:.4f})")
    elif not auc_ok:
        reason = f"ROC-AUC {auc_overall:.4f} < floor {args.auc_floor:.4f}"
    elif not gap_ok:
        reason = (f"score gap not positive "
                  f"(anom={mean_anom:.4f}, norm={mean_norm:.4f})")
    elif not p_all_ok:
        reason = f"Welch all-anom p-value {p_all:.2e} >= 0.05"
    elif not is_unseen.any():
        reason = "no unseen anomalies in V_test — generator misconfigured"
    elif p_unseen >= 0.05:
        reason = f"Welch unseen-vs-normal p-value {p_unseen:.2e} >= 0.05"
    else:
        reason = (f"unseen score gap not positive "
                  f"(unseen={mean_unseen:.4f}, norm={mean_norm:.4f})")
    print(f"\n✗ FAIL — {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
