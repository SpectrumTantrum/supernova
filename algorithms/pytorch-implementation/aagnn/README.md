# AAGNN

PyTorch implementation of **Subtractive Aggregation for Attributed Network
Anomaly Detection** — Shuang Zhou, Qiaoyu Tan, Zhiming Xu, Xiao Huang, Fu-lai
Chung, *CIKM '21*, November 2021.
<https://doi.org/10.1145/3459637.3482195>

The algorithm scores each node of an attributed graph by **how much it
deviates from its local neighbourhood**. Normal nodes blend in with their
neighbours; anomalies stand out. AAGNN bakes this premise directly into its
single graph layer via *subtractive* aggregation, then learns a
hypersphere around the (presumed-normal) majority of nodes — distance to
the hypersphere centre is the anomaly score.

## Pipeline

```
                ┌─────────────────────────────┐
                │  Attributed graph G=(V,E,X) │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │ AAGNN layer  (paper §3.1)   │
                │  z_i = W x_i  (no bias)     │
                │  h_i = σ(z_i − Aggregate)   │
                │  Aggregate: mean | attention│
                └──────────────┬──────────────┘
                               │
   ┌───────────────────────────▼───────────────────────────┐
   │ Pseudo-labelling   (Algorithm 1, lines 1–5)           │
   │  c = mean(h)                                           │
   │  Take p% closest-to-c as pseudo-normal set S           │
   │  R = 60% of S (train), D = 40% of S (val), T = V − S   │
   └───────────────────────────┬───────────────────────────┘
                               │
   ┌───────────────────────────▼───────────────────────────┐
   │ Hypersphere training (paper §3.2, Eq. 5)               │
   │  L = (1/|R|) Σ_R ‖h_i − c‖²  +  (λ/2) ‖Θ‖²_F           │
   │  c is fixed; weight decay applied via the optimiser    │
   └───────────────────────────┬───────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │ Anomaly score (Eq. 6)       │
                │  s(i) = ‖h_i − c‖²          │
                └─────────────────────────────┘
```

## Files

| File              | Role                                                                            |
|-------------------|---------------------------------------------------------------------------------|
| `data.py`         | Graph dataclasses, SBM-based synthetic generator, paper-protocol anomaly injection |
| `layer.py`        | `AbnormalityAwareLayer` with mean and attention aggregators (paper §3.1)        |
| `train.py`        | `compute_pseudo_labels`, `train_aagnn`, `anomaly_scores` (Algorithm 1)          |
| `model.py`        | `AAGNN` orchestrator with `fit()` / `score()` / `predict()` / `save()` / `load()` |
| `example.py`      | End-to-end smoke test on a synthetic clustered SBM                              |

## Install & run

```bash
pip install -r requirements.txt
python example.py                          # default: mean aggregator, AUC ≥ 0.75
python example.py --aggregator attention   # AAGNN-A variant
python example.py --seed 7 --epochs 300    # different graph + longer training
```

The smoke test plants a 5-community SBM (300 nodes), injects 30 anomalies
(15 structural cliques + 15 contextual feature swaps per the paper-cited
Ding et al. 2019 / Song et al. 2007 protocol), trains AAGNN, then checks:

1. Train and validation losses trend down.
2. ROC-AUC on the held-out test set `T = V − S` is ≥ `--auc-floor` (0.75 by
   default; the paper reports 0.82–0.85 on real datasets, and a clean SBM
   typically lands in roughly the same range).
3. Welch's t-test on score distributions: anomaly scores are significantly
   higher than normal scores (`p < 0.05`, mean-gap > 0).

It exits 0 only if **all three** hold.

## Library use

```python
from sklearn.metrics import roc_auc_score
from data import SyntheticAttributedNetwork
from model import AAGNN, AAGNNConfig

net = SyntheticAttributedNetwork(seed=0).generate()
model = AAGNN(AAGNNConfig(aggregator="mean")).fit(net)

scores = model.score()                        # (n,) — Eq. 6
print("AUC:", roc_auc_score(net.labels, scores))

# Apply the trained model to a fresh graph (must share feature dimensionality)
other = SyntheticAttributedNetwork(seed=1).generate()
scores_other = model.score(other)

model.save("./aagnn.pt")
loaded = AAGNN.load("./aagnn.pt")
```

`AttributedNetwork` is a small frozen dataclass holding `X` (n × f
features), `edges` (E × 2 undirected, canonical `i < j`), and optional
ground-truth `labels`. Bring your own — anything you can express as those
arrays will work.

## Hyperparameters (paper §4.1)

| Symbol          | Default | Meaning                                              |
|-----------------|---------|------------------------------------------------------|
| `d`             | 256*    | hidden / embedding dimension                         |
| aggregator      | `mean`  | `mean` (Eq. 2) or `attention` (Eqs. 3–4)             |
| `k`             | 1       | hop count for the neighbour set `N_i^k` (paper §4.2) |
| `p`             | 50      | percent of nodes pseudo-labelled as normal           |
| R : D           | 3 : 2   | train / validation split inside the pseudo-normal set|
| `λ` (`weight_decay`) | 5e-4 | L2 regularisation on Θ (paper Eq. 5)            |
| optimiser       | `adam`  | `adam` or `sgd`                                      |
| epochs          | 200     | hypersphere training epochs                          |
| lr              | 1e-3    | optimiser learning rate                              |

\* `example.py` defaults to `hidden_dim=64` to keep the smoke test fast on
CPU. Override with `--hidden-dim 256` to match the paper.

## Critical: why `bias=False` matters

The hypersphere objective is essentially graph-flavoured Deep SVDD (Ruff et
al. 2018, the paper AAGNN cites as ref [21]). Any trainable bias on the
linear projection `z_i = W x_i + b` would let the optimiser push every
node's representation to the centre `c` and drive the loss to zero — a
trivial degenerate solution. `layer.py` uses `nn.Linear(..., bias=False)`
unconditionally, with a comment to that effect.

## Out of scope

This module ships the **algorithm** only. Things deliberately excluded:

- The BlogCatalog / Flickr / PubMed datasets (proprietary; the paper's §4
  evaluation graphs are not redistributed here).
- The five baseline anomaly-detection models compared in §4.1 (Radar,
  MADAN, DOMINANT, AnomalyDAE, AEGIS).
- Visualisation / parameter-analysis figures (§4.2–4.3).
- A PyG / DGL wrapping. The implementation uses plain Python neighbour
  lists + masked tensor ops — fast enough for graphs up to a few thousand
  nodes; rewrite with sparse message passing if you need to scale further.
