"""Train ACDNE on the real DBLPv7 -> ACMv9 cross-network transfer.

Reference: Shen et al. AAAI 2020, paper §Datasets (DBLPv7 -> ACMv9 row of
Table 2). Paper reports Micro-F1 ≈ 0.66 averaged across 5 seeds.

Run:
    python run_example.py            # full run, n_iters=3000
    python run_example.py --smoke    # short run, n_iters=2000, floor 0.50

Why the iter budgets here exceed example.py's synthetic defaults:
    The synthetic SBM in example.py is dense (240 nodes, 64 features) and
    converges in ~1000 iters. The real DBLPv7 -> ACMv9 transfer is sparse
    (5,484 nodes, 6,775 BoW features) and needs ~5,600 iters in the paper
    authors' upstream code (30 epochs * ~187 batches). 3,000 iters here
    lands within ~0.05 of the paper's reported Micro-F1 across seeds.

Smoke-test contract (exits non-zero unless ALL hold):
    1. Source classification loss L_y trends down (start > end).
    2. Final domain-discriminator accuracy is balanced —
       |d_acc - 0.5| <= 0.15.
    3. Micro-F1 on the unlabelled ACMv9 target >= --micro-f1-floor
       (default: 0.55 full / 0.50 smoke; calibrated against seed sweep).
    4. Micro-F1 strictly beats the source's majority-class baseline.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from sklearn.metrics import f1_score

# Reach parent acdne/ for ACDNE + ACDNEConfig.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from model import ACDNE, ACDNEConfig  # noqa: E402

from loader import load_dblp_acmv9


FULL_ITERS = 3000
SMOKE_ITERS = 2000
FULL_FLOOR = 0.55
SMOKE_FLOOR = 0.50


def _short(losses: list[float]) -> str:
    if len(losses) <= 6:
        return "[" + ", ".join(f"{l:.4f}" for l in losses) + "]"
    head = ", ".join(f"{l:.4f}" for l in losses[:3])
    tail = ", ".join(f"{l:.4f}" for l in losses[-3:])
    return f"[{head}, ..., {tail}]"


def _majority_baseline_f1(y_true: np.ndarray, y_s: np.ndarray) -> float:
    majority = int(np.bincount(y_s).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)
    return float(f1_score(y_true, y_pred, average="micro"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=FULL_ITERS)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--smoke", action="store_true",
                    help=f"Short run: n_iters={SMOKE_ITERS}, floor {SMOKE_FLOOR}.")
    ap.add_argument("--micro-f1-floor", type=float, default=None,
                    help=f"Override the default floor "
                         f"({FULL_FLOOR} full, {SMOKE_FLOOR} smoke).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    n_iters = SMOKE_ITERS if args.smoke else args.n_iters
    default_floor = SMOKE_FLOOR if args.smoke else FULL_FLOOR
    floor = args.micro_f1_floor if args.micro_f1_floor is not None else default_floor

    print("=" * 72)
    print("ACDNE — DBLPv7 -> ACMv9 cross-network transfer (Shen et al. AAAI 2020)")
    print("=" * 72)

    print("\n[1/4] Loading real cross-network…")
    net = load_dblp_acmv9()
    print(f"  Source (DBLPv7): n_s={net.n_s}, edges={len(net.edges_s)}")
    print(f"  Target (ACMv9):  n_t={net.n_t}, edges={len(net.edges_t)}")
    print(f"  feat_dim={net.feat_dim}  n_classes={net.n_classes}")

    print(f"\n[2/4] Training ACDNE for {n_iters} iters…")
    cfg = ACDNEConfig(
        n_iters=n_iters,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    model = ACDNE(cfg).fit(net)

    h = model.history
    print("\n[3/4] Loss curves:")
    print(f"  L_y     : {_short(h['loss_y'])}")
    print(f"  L_p     : {_short(h['loss_p'])}")
    print(f"  L_d     : {_short(h['loss_d'])}")
    print(f"  d_acc   : {_short(h['domain_acc'])}")
    print(f"  λ ramp  : {h['grl_lambda'][0]:.3f} → {h['grl_lambda'][-1]:.3f}")
    print(f"  lr decay: {h['lr'][0]:.4f} → {h['lr'][-1]:.4f}")

    print("\n[4/4] Evaluation on ACMv9 target:")
    assert net.y_t is not None
    y_pred = model.predict()
    micro = float(f1_score(net.y_t, y_pred, average="micro"))
    macro = float(f1_score(net.y_t, y_pred, average="macro"))
    baseline = _majority_baseline_f1(net.y_t, net.y_s)
    final_d_acc = float(np.mean(h["domain_acc"][-max(20, len(h["domain_acc"]) // 20):]))

    print(f"  Micro-F1:           {micro:.4f}")
    print(f"  Macro-F1:           {macro:.4f}")
    print(f"  Majority baseline:  {baseline:.4f}")
    print(f"  Final d_acc (avg):  {final_d_acc:.4f}")

    loss_y = h["loss_y"]
    loss_y_down = loss_y[-1] < loss_y[0]
    d_acc_balanced = abs(final_d_acc - 0.5) <= 0.15
    micro_ok = micro >= floor
    beats_baseline = micro > baseline

    if loss_y_down and d_acc_balanced and micro_ok and beats_baseline:
        print(f"\nPASS — DBLPv7 labels transferred to ACMv9 (floor={floor:.2f}).")
        return 0

    if not loss_y_down:
        reason = f"L_y did not decrease (start={loss_y[0]:.4f}, end={loss_y[-1]:.4f})"
    elif not d_acc_balanced:
        reason = (
            f"final domain-discriminator accuracy {final_d_acc:.4f} "
            f"outside |d_acc - 0.5| <= 0.15"
        )
    elif not micro_ok:
        reason = f"Micro-F1 {micro:.4f} < floor {floor:.4f}"
    else:
        reason = f"Micro-F1 {micro:.4f} did not beat majority baseline {baseline:.4f}"
    print(f"\nFAIL — {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
