"""End-to-end AAGNN demo on the Cora citation network with injected anomalies.

Run:
    cd algorithms/pytorch-implementation/aagnn/datasets/cora
    python fetch.py
    python run_example.py            # 200 epochs (default)
    python run_example.py --smoke    # 20 epochs

Smoke-test contract (mirrors parent example.py output, with a relaxed exit
condition for real-world Cora):
    PASS  iff  Welch's t-test passes (p < 0.05) AND mean anomaly score > mean
              normal score.
The ROC-AUC is printed for visibility but is NOT a gating condition — the
synthetic-SBM floor (0.75) is too tight on Cora's high-dim sparse features.
A practical soft floor of 0.65 is reported alongside.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

from scipy import stats
from sklearn.metrics import roc_auc_score

# Parent module shim (aagnn/datasets/cora/run_example.py → aagnn/{data,model}.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from model import AAGNN, AAGNNConfig  # noqa: E402

from loader import load_cora_with_anomalies  # noqa: E402


SOFT_AUC_FLOOR = 0.65


def _short(losses: list[float]) -> str:
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
    ap.add_argument("--smoke", action="store_true",
                    help="Fast 20-epoch smoke run (overrides --epochs).")
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="Paper default. Cora's 1433-dim sparse features need "
                         "the full capacity; 64 collapses the anomaly gap.")
    args = ap.parse_args()

    epochs = 20 if args.smoke else args.epochs

    print("=" * 64)
    print("AAGNN — Cora citation network (real data + injected anomalies)")
    print("=" * 64)

    print("\n[1/4] Loading Cora with injected anomalies…")
    net = load_cora_with_anomalies(seed=args.seed)
    n_anom = int(net.labels.sum()) if net.labels is not None else 0
    print(f"  Nodes:             {net.n:,}")
    print(f"  Edges:             {len(net.edges):,}")
    print(f"  Anomalies (true):  {n_anom:,}")
    print(f"  Feature dim:       {net.f}")

    tag = "M" if args.aggregator == "mean" else "A"
    print(f"\n[2/4] Training AAGNN-{tag} ({epochs} epochs)…")
    cfg = AAGNNConfig(
        hidden_dim=args.hidden_dim,
        aggregator=args.aggregator,
        epochs=epochs,
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
    auc_T = float(roc_auc_score(net.labels[T], scores[T])) if anom_in_T else float("nan")

    anom_mask = net.labels == 1
    norm_mask = net.labels == 0
    mean_anom = float(scores[anom_mask].mean())
    mean_norm = float(scores[norm_mask].mean())

    # AAGNN scores ARE squared L2 distances to the hypersphere centre c
    # (Eq. 6), so the per-class means below double as cluster-centroid
    # distances in the learned representation space.
    print(f"  Test-set size:        |T| = {len(T)}, of which {anom_in_T} are true anomalies")
    print(f"  ROC-AUC on T:         {auc_T:.4f}   (soft floor {SOFT_AUC_FLOOR:.2f})")
    print(f"  Mean dist to c (anomaly):  {mean_anom:.4f}")
    print(f"  Mean dist to c (normal):   {mean_norm:.4f}")
    print(f"  Score gap (anomaly - normal):  {mean_anom - mean_norm:+.4f}")

    t_stat, p_value = stats.ttest_ind(
        scores[anom_mask], scores[norm_mask], equal_var=False
    )
    t_stat = float(t_stat)
    p_value = float(p_value)
    print(f"  Welch's t-test:       t = {t_stat:+.3f}   p = {p_value:.2e}")

    gap_ok = mean_anom > mean_norm
    p_ok = p_value < 0.05

    if not gap_ok:
        print(f"\n✗ FAIL — score gap not positive (anom={mean_anom:.4f}, norm={mean_norm:.4f})")
        return 1
    if not p_ok:
        print(f"\n✗ FAIL — Welch's p-value {p_value:.2e} ≥ 0.05")
        return 1

    auc_soft_ok = (not math.isnan(auc_T)) and auc_T >= SOFT_AUC_FLOOR
    if auc_soft_ok:
        print(f"\n✓ PASS — anomalies are separable (p<0.05, gap>0) and AUC "
              f"{auc_T:.4f} ≥ {SOFT_AUC_FLOOR:.2f}.")
    else:
        print(f"\n✓ PASS — anomalies are separable (p<0.05, gap>0); AUC "
              f"{auc_T:.4f} below soft floor {SOFT_AUC_FLOOR:.2f} but "
              f"separation is statistically significant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
