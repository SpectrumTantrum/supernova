# ACDNE

MLX implementation of **Adversarial Deep Network Embedding for
Cross-network Node Classification** — Xiao Shen, Quanyu Dai, Fu-lai
Chung, Wei Lu, Kup-Sze Choi, *AAAI 2020*.

The algorithm transfers node-classification labels from a fully labelled
*source* graph G^s to a topologically-disjoint, fully unlabelled *target*
graph G^t that shares the same label space but lives in a shifted
attribute distribution. It learns a **shared embedding** that (a)
classifies source nodes correctly, (b) preserves K-step PPMI proximity
on each side, and (c) is *domain-invariant* — a discriminator can no
longer tell source from target. Domain invariance is enforced via a
gradient-reversal layer (Ganin & Lempitsky 2015), so the encoder,
classifier, and discriminator all train under one SGD step.

## Pipeline

```
       G^s (labelled)                              G^t (unlabelled)
   ┌──────────────────┐                       ┌──────────────────┐
   │ X_s, edges_s,y_s │                       │ X_t, edges_t     │
   └────────┬─────────┘                       └────────┬─────────┘
            │  ppmi_matrix (Eq. 3 weights, K-step PPMI)│
            │  neighbour_input  n_i = Σ a_ij/Σa·x_j    │
            ▼                                          ▼
       (X_s, N_s)                                 (X_t, N_t)
            │                                          │
            └───────────────┬──────────────────────────┘
                            ▼
            ┌──────────────────────────────┐
            │ Shared EmbeddingModule       │
            │  FE1(x)  Eq. 1  ─┐           │
            │  FE2(n)  Eq. 2  ─┼─ concat   │
            │                 Eq. 4 →  e_i │
            └──────────────┬───────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼                             ▼
   ┌──────────────────┐       ┌─────────────────────────┐
   │ NodeClassifier   │       │ DomainDiscriminator     │
   │ Eq. 6 — softmax  │       │ GRL → MLP → 2 logits    │
   │ L_y = CE on G^s  │       │ L_d = CE (0=src,1=tgt)  │
   └────────┬─────────┘       └────────────┬────────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
        L = L_y  +  α·L_p (pairwise, Eq. 5, both sides)
              +  L_d  ← GRL flips ∂L_d/∂e on the encoder
                          (Eqs. 11–12: one SGD step
                           updates encoder, classifier,
                           AND discriminator at once)

        Schedules (paper §Implementation Details):
            μ_p = μ_0 / (1 + 10p)^0.75       lr decay
            λ_p = 2 / (1 + e^{-10p}) − 1     GRL ramp 0 → ~1
```

## Files

| File              | Role                                                                            |
|-------------------|---------------------------------------------------------------------------------|
| `data.py`         | `CrossNetwork` dataclass, `ppmi_matrix` (K-step PPMI per Levy & Goldberg 2014), `neighbour_input` (Eq. 3), `SyntheticCrossNetwork` SBM with planted domain shift |
| `layers.py`       | `FeatureExtractor` (Eqs. 1-2), `EmbeddingModule` (Eq. 4 concat fusion), `NodeClassifier` (Eq. 6), `DomainDiscriminator` (Eq. 9) |
| `train.py`        | LR / λ schedules, half-source/half-target minibatch sampler, `pairwise_loss` (Eq. 5), `train_acdne` (Algorithm 1) |
| `model.py`        | `ACDNE` orchestrator with `ACDNEConfig`, `fit` / `predict` / `predict_proba` / `embed` / `save` / `load` |
| `example.py`      | End-to-end synthetic cross-network smoke test                                   |

## Install & run

```bash
pip install -r requirements.txt
python example.py                        # default — seed 0, 1000 iters, Micro-F1 ≥ 0.65
python example.py --seed 7
python example.py --seed 42 --n-iters 1500
python example.py --embed-hidden-dim 512 # paper-faithful FE width (default 256 for speed)
```

The smoke test plants two SBMs sharing class semantics but with a
planted Gaussian domain shift on the target's attributes (analogous to
the paper's discrete bit-flip corruption protocol). It exits 0 only if
**all four** hold:

1. Source classification loss `L_y` trends down (start > end).
2. Final domain-discriminator accuracy is balanced — `|d_acc − 0.5| ≤
   0.15`. Symmetric around chance because a discriminator stuck at 0.32
   is just as "domain-fooled" as one stuck at 0.68.
3. Micro-F1 on the unlabelled target ≥ `--micro-f1-floor` (default
   0.65; the paper hits 0.66–0.83 on real cross-network transfers).
4. Micro-F1 strictly beats the source-majority-label baseline.

## Library use

```python
from sklearn.metrics import f1_score
from data import SyntheticCrossNetwork
from model import ACDNE, ACDNEConfig

net = SyntheticCrossNetwork(seed=0).generate()
model = ACDNE(ACDNEConfig(n_iters=1000)).fit(net)

y_t_pred = model.predict()                 # (n_t,) target-side predictions
print("Micro-F1:", f1_score(net.y_t, y_t_pred, average="micro"))

e_t = model.embed("target")                # learned target embeddings (Eq. 4)
e_s = model.embed("source")
probs = model.predict_proba()              # (n_t, n_classes) softmax

# Apply the trained model to a fresh target side (must share feat_dim/n_classes)
other = SyntheticCrossNetwork(seed=1).generate()
y_other = model.predict(other)

model.save("./acdne.npz")
loaded = ACDNE.load("./acdne.npz")
```

`CrossNetwork` is a frozen dataclass holding `X_s` / `X_t` (float32
features), `edges_s` / `edges_t` (int64, canonical `i < j`), `y_s`
(source labels), and an evaluation-only `y_t`. `fit` never reads
`y_t` — it's exposed purely so `example.py` can score the result.

## Hyperparameters (paper §Implementation Details)

| Symbol            | Default | Paper | Meaning                                              |
|-------------------|---------|-------|------------------------------------------------------|
| `embed_hidden_dim` (f(1)) | 512 | 512 | FE1 / FE2 hidden width                          |
| `fe_out_dim` (f(2))       | 128 | 128 | FE1 / FE2 output width                          |
| `embed_dim` (d)           | 128 | 128 | concat-layer output (final embedding)           |
| `disc_hidden_dim`         | 128 | 128 | discriminator hidden width (d(1) = d(2))        |
| `n_iters`                 | 1000 | per-dataset | iterations of Algorithm 1                |
| `batch_size`              | 100  | 100 | half source + half target per minibatch         |
| `mu_0`                    | 0.02 | 0.02 / 0.01 | initial LR (citation / Blog defaults)   |
| `p_pair` (α)              | 0.1  | 0.1 / 1e-3  | weight on pairwise constraint (Eq. 5)   |
| `weight_decay`            | 1e-3 | 1e-3 | L2 on all parameters                            |
| `ppmi_K`                  | 3    | 3    | PPMI random-walk window (Eq. 3 weights)         |
| `momentum`                | 0.9  | 0.9  | SGD momentum                                    |

The defaults are the paper's **citation-network** values. For dense
Blog-style networks the paper uses `mu_0 = 0.01`, `p_pair = 1e-3`.

`example.py` overrides `embed_hidden_dim=256` (vs paper 512) — the
synthetic SBM saturates well before 512 and the override roughly halves
CPU run time. Use `--embed-hidden-dim 512` to match the paper exactly.

## Critical implementation notes

### One SGD step trains all three modules

Adversarial training is usually written as alternating min-max steps. ACDNE
collapses it to a single SGD update over `params(encoder) ∪ params(classifier)
∪ params(discriminator)` because the gradient-reversal layer between encoder
and discriminator sign-flips `∂L_d/∂e` on the way back. The discriminator's
own parameters get the natural gradient and so are trained to *maximise*
domain-classification accuracy; the encoder simultaneously gets `−λ ·
∂L_d/∂e` and so is trained to *fool* the discriminator. This matches paper
Eq. 12 and the Ganin & Lempitsky 2015 DANN trick exactly. The MLX loop computes the equivalent encoder sign flip explicitly while
training the classifier and discriminator with their natural objectives.

### Shared encoder, not twin encoders

`EmbeddingModule` is invoked on both source and target with the *same*
parameters. The paper is explicit: "shared trainable parameters between
G^s and G^t" (§3, Fig. 1). Domain alignment is enforced by the
adversarial discriminator, not by giving each side its own encoder.
This is what makes the embedding cross-network usable at all — you
project a fresh target into the same coordinate system the classifier
was trained on.

### PPMI is precomputed once, not per minibatch

The K-step PPMI matrices `A_s_ppmi`, `A_t_ppmi` are built in `model.fit`
before the loop and indexed by minibatch. Rebuilding them each iteration
would dominate runtime (`O(n³)` matrix powers) and produce identical
weights — the random walks don't depend on training state.

### λ ramp and LR decay

Following Ganin & Lempitsky (DANN), λ ramps from 0 → ~1 as `λ_p =
2/(1+e^{−10p}) − 1`. Starting at λ = 0 lets the encoder first learn a
useful classification signal before the adversarial pressure kicks in.
LR decays as `μ_p = μ_0 / (1+10p)^0.75`. Both schedules are paper
§Implementation Details and are non-learnable — `GradientReversal.lambda_`
is a plain Python float, not a Parameter or Buffer.

### Pairwise constraint is within-batch, not full-graph

Paper Eq. 5 sums `a_ij ||e_i − e_j||²` over all pairs in a graph. We
restrict the sum to within-batch pairs (B × B sub-block of the PPMI
matrix), normalised by `B²`. This is a stochastic estimator of the
full-graph quantity and matches what Algorithm 1 line 4 actually
computes per-minibatch. The expectation over uniformly-sampled
minibatches recovers the full sum up to the normalisation constant,
which is absorbed into `p_pair`.

## Out of scope

- The DBLPv7 / Citationv1 / ACMv9 / Blog1 / Blog2 datasets used in paper §4.
- Comparative baselines (DeepWalk, node2vec, GraphSAGE, GCN, GAT, DANN,
  CDNE, ACDNE-1/2/3 ablations).
- Visualisation experiments (§4.5 t-SNE plots).
- A graph-library wrapping. The implementation uses dense (n × n) PPMI
  matrices — fine for graphs up to a few thousand nodes; rewrite with
  sparse propagation if you need to scale further.
