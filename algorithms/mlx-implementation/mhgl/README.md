# MHGL

MLX implementation of **Unseen Anomaly Detection on Networks via
Multi-Hpersphere Learning** [sic — paper title misspells "Hypersphere"]
— Shuang Zhou, Xiao Huang, Ninghao Liu, Qiaoyu Tan, Fu-Lai Chung,
*SIAM SDM 2022*.

The algorithm tackles a harder version of the AAGNN problem: detect
**unseen** anomalies (novel classes never labelled at training time)
mixed into the test set alongside **seen** anomalies (a known class with
q labelled training examples). Single-hypersphere methods like Deep SVDD
and AAGNN blur the diverse normal-pattern distribution into a single
centre, so unseen types camouflage as normals. MHGL clusters normals
into many fine-grained hyperspheres via a GMM-based Pattern Distribution
Estimator and scores by **min-distance to the nearest centre** (Eq. 3.7).

## Pipeline

```
                ┌──────────────────────────────────────────┐
                │ Attributed graph G=(V, E, X)             │
                │ + V_train (q labelled anomalies +        │
                │   p% labelled normals; paper §2)         │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │ GCN encoder  (paper Eq. 3.2 / §4.2)      │
                │  4 layers, dims [256, 128, 64, 32], ReLU │
                │  H = GCN^k(X, A_hat)                     │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │ PDE  (paper §3.1, Algorithm 1)            │
                │  Fit GMM with k_normal components on      │
                │  H[labelled-normals]; recursively split   │
                │  any component with |S^i| > u            │
                │  → fine-grained patterns S^1, …, S^p      │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │ Centres + high-confidence sets            │
                │  c_i = mean(H[S^i])      (Eq. 3.5)        │
                │  F^i = {j ∈ S^i : ξ_j > t}                │
                │  r_i = quantile(d(F^i, c_i))              │
                │  H^i = {v ∈ V : d(c_i, h_v) ≤ r_i}        │
                │  CENTRES ARE FROZEN ∀ T epochs            │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │ Multi-hypersphere training (Algorithm 2)  │
                │  Each epoch:                              │
                │  1. Sample α·|H^i| mixup pseudo-labels    │
                │     h_new = (1-β)h_a + β h_b   (Eq. 3.8)  │
                │  2. Loss (Eq. 3.6, optimiser handles L2): │
                │       Σ_i mean_{D_i} ||h - c_i||²         │
                │     + (σ/(qp)) Σ_{r,i} (||h_r-c_i||²+ε)⁻¹ │
                │  3. Adam step                             │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │ Anomaly score (Eq. 3.7)                  │
                │  s(v) = min_i ||h_v - c_i||²              │
                │  Higher = more anomalous                  │
                └──────────────────────────────────────────┘
```

## Files

| File              | Role                                                                            |
|-------------------|---------------------------------------------------------------------------------|
| `data.py`         | `AttributedNetwork` dataclass, `build_normalized_adj` (Eq. 3.2 / Kipf & Welling), `SyntheticAttributedNetwork` (rare-class anomaly protocol from paper §4.1) |
| `gcn.py`          | `GCNLayer`, `GCNEncoder` — 4-layer stack matching paper §4.2                    |
| `pde.py`          | `fit_pde` (Algorithm 1), GMM with adaptive covariance fallback                  |
| `train.py`        | Centres (Eq. 3.5), high-confidence sets (Algo 2 lines 4-7), mixup (Eq. 3.8), loss (Eq. 3.6), training loop, scoring (Eq. 3.7) |
| `model.py`        | `MHGL` orchestrator with `MHGLConfig` and `fit` / `score` / `predict` / `save` / `load` |
| `example.py`      | End-to-end synthetic-SBM smoke test with seen + unseen anomaly evaluation       |

## Install & run

```bash
pip install -r requirements.txt
python example.py                                         # default — seed 0, 80 epochs, AUC ≥ 0.75
python example.py --epochs 300                            # paper-fidelity training length
python example.py --hidden-dims 256,128,64,32 --epochs 300  # paper-fidelity encoder
python example.py --radius-quantile 1.0                   # paper-faithful r_i = max d(F^i, c_i)
```

The smoke test plants 5 normal communities (300 nodes, distinct feature
centroids) plus a 30-node *seen-anomaly* community and a 30-node
*unseen-anomaly* community. q=20 of the seen anomalies and 10% of the
normals enter V_train with revealed labels per paper §2; the remaining
seen, **all** unseen, and held-out normals form V_test.

It exits 0 only if **all four** hold:

1. Train loss trends down.
2. ROC-AUC on V_test ≥ `--auc-floor` (default 0.75).
3. Welch's t-test (V_test all-anomalies vs normals): `p < 0.05`,
   positive mean-gap.
4. Welch's t-test (V_test unseen vs normals): `p < 0.05`, positive
   mean-gap. This statistically gates the paper's main claim — detecting
   anomaly types never seen at training — without baking in a
   threshold-tuned AUC floor.

A typical seed-0 run should clear both the gated ROC-AUC and unseen-anomaly
t-test checks; exact AUCs vary with MLX, NumPy, and scikit-learn versions.

### Seed sensitivity

MHGL freezes hypersphere centres at the random-init forward pass (paper
Algorithm 2 line 3 — same Deep-SVDD discipline as AAGNN). On the
paper's real graphs (7k–18k nodes, strong inductive bias from class
structure) this is reliable. On a small synthetic with 360 nodes and
random GCN initialisation, the encoder occasionally produces an H₀ where
the unseen-anomaly community lands inside the radius of one normal
hypersphere — once captured, those nodes get pulled to the centre by
the contraction loss and the algorithm fails to detect them. The smoke
test is calibrated to seed 0; other seeds may need `--epochs 300` or a
tighter `--radius-quantile`.

## Library use

```python
from sklearn.metrics import roc_auc_score
import numpy as np

from data import SyntheticAttributedNetwork
from model import MHGL, MHGLConfig

net = SyntheticAttributedNetwork(seed=0).generate()
model = MHGL(MHGLConfig(hidden_dims=(64, 32, 16, 16), epochs=80, k_normal=3)).fit(net)

scores = model.score()                     # (n,) — Eq. 3.7
test_idx = ~net.train_mask
print("AUC on V_test:", roc_auc_score(net.labels[test_idx], scores[test_idx]))

# Apply the trained model to a fresh graph (must share feature dimensionality)
other = SyntheticAttributedNetwork(seed=1).generate()
scores_other = model.score(other)

model.save("./mhgl.npz")
loaded = MHGL.load("./mhgl.npz")
```

`AttributedNetwork` is a frozen dataclass holding `X` (n × f features),
`edges` (E × 2 undirected, canonical `i < j`), `labels`, `anomaly_type`
(0 normal, 1 seen, 2 unseen), and the `train_mask` / `label_mask` masks
defining the V_train ↔ V_test partition per paper §2.

## Hyperparameters (paper §4.2)

| Symbol        | Default | Paper | Meaning                                                |
|---------------|---------|-------|--------------------------------------------------------|
| `hidden_dims` | (256, 128, 64, 32) | same | GCN encoder layer widths                            |
| activation    | `relu`  | same  | per-layer non-linearity                                |
| `k_normal`    | 10      | 10    | initial GMM components on labelled normals             |
| `pde_split_threshold_u` | 30 | per-dataset | `\|S^i\| > u` triggers recursive split        |
| `pde_max_recursion` | 3 | — | hard cap on PDE recursion depth                          |
| `high_confidence_t` | 0.7 | "suitable threshold" | posterior gate on F^i              |
| `radius_quantile`   | 1.0 | 1.0 (max) | quantile of F^i distances used as r_i           |
| epochs `T`    | 300     | 300   | training epochs (Algorithm 2 line 8)                   |
| lr            | 1e-3    | "suitable" | Adam learning rate                                |
| `weight_decay` (λ) | 5e-4 | 5e-4 | L2 regularisation on Θ (Eq. 3.6 third term)        |
| `sigma` (σ)   | 1.0     | per-dataset | balance weight on Eq. 3.6 second term            |
| `augmentation_alpha` (α) | 2 | 2 | mixup pseudo-label count multiplier              |
| `eps_repulsion` | 1e-6 | —    | numerical guard on inverse-distance (paper silent)     |

The smoke test (`example.py`) overrides `hidden_dims=(64,32,16,16)`,
`k_normal=3`, `epochs=80`, and `radius_quantile=0.25` to keep the run
fast on CPU and to compensate for the small synthetic's sensitivity to
random init. Use `--hidden-dims 256,128,64,32 --epochs 300
--radius-quantile 1.0` for the paper-faithful configuration.

## Critical implementation notes

### `bias=True` (differs from AAGNN's `bias=False`)

AAGNN sets `bias=False` on its single linear projection because the
single-hypersphere Deep SVDD objective (Ruff et al. 2018) admits a
trivial collapse to a constant if any bias is present. MHGL is
multi-hypersphere with an additional inverse-distance repulsion term
(Eq. 3.6 second term) that **diverges** if all representations collapse
to a single point — making the trivial solution unreachable. The paper
therefore uses the Kipf & Welling default (`bias=True`); we follow it.
Don't "fix" this to `bias=False` to match AAGNN.

### Frozen centres

Hypersphere centres `c_i` are computed once from the random-init
forward pass (Algorithm 2 line 3) and held constant across all T
training epochs. Only Θ (encoder weights) updates. This is the
Deep-SVDD discipline — the same convention AAGNN follows for its single
centre. The contraction term then forces the encoder to *learn* to map
each labelled / high-confidence node near its assigned (random) centre.

### Numerical guard on the repulsion term

Eq. 3.6's second term is `sum_{r,i} (||h_r - c_i||²)⁻¹`. At init, before
training pulls anomalies away, a labelled anomaly can land arbitrarily
close to a normal centre — gradient explodes. We replace the bare
inverse with `1 / (||h_r - c_i||² + eps)` where `eps = 1e-6`. The paper
is silent on this; the guard is essential for stable optimisation.

### One PDE call, not two

Algorithm 2 line 2 reads "Adopt the PDE to obtain x normal patterns and
y abnormal patterns". On a literal reading you'd call PDE twice. But
Eq. 3.6 only uses *normal* centres — the second term references raw
labelled-anomaly node indices, never abnormal patterns or centres. To
match the loss exactly, we run PDE only on labelled-normal nodes.

## Out of scope

- The Computer / Photo / CS proprietary datasets from paper §4.1.
- Comparative baselines from §4.2 (SPARC, DeepSAD, OpenWGL, GDN,
  OCGNN). The portfolio implements algorithms, not benchmark suites.
- Ablation variants from §4.5 (MHGL-, HGL, HGL-).
- A PyG / DGL wrapping. The implementation uses a dense MLX array for the
  normalized GCN propagation matrix — fine for graphs up to a few thousand
  nodes; rewrite with sparse message passing if you need to scale.
