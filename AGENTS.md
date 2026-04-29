# AGENTS.md

> Single source of truth for AI coding agents working in this repo.
> `CLAUDE.md` and `GEMINI.md` are symlinks pointing at this file.

## What this repo is

**`supernova`** is a portfolio of from-scratch implementations of geospatial
and urban-analytics ML algorithms drawn from published research papers. Each
algorithm lives under `algorithms/<framework>/<name>/` and is independently
runnable.

## Repo layout

```
supernova/
├── AGENTS.md                       this file (also CLAUDE.md, GEMINI.md)
├── .gitignore                      excludes .research/, __pycache__, etc.
├── .research/                      paywalled paper PDFs — NEVER commit
└── algorithms/
    ├── pytorch-implementation/
    │   └── <algo-name>/
    │       ├── README.md           paper ref, theory recap, run instructions
    │       ├── requirements.txt
    │       ├── data.py             dataclasses, helpers, synthetic generator
    │       ├── <algo>_<stage>.py   one module per pipeline stage
    │       ├── model.py            top-level orchestrator class
    │       └── example.py          runnable smoke test on synthetic data
    └── mlx-implementation/
        └── <algo-name>/            (same layout as above)
```

Algorithm folders are deliberately **flat** — no `src/` or deep package
hierarchies. Aim for 5–8 source files per algorithm.

## Conventions

### Implementation
- **Framework**: PyTorch by default. Pretrained vision backbones via
  `torch.hub.load_state_dict_from_url` with an offline-fallback `RuntimeError`
  that names the expected cache path so users can side-load weights.
- **Hyperparameters**: faithful to the source paper. Cite section / equation
  in the module docstring or comment so reviewers can verify quickly.
- **Module docstrings**: top of each file references the paper section it
  implements (e.g. `"""Stage 1 — paper §3.2."""`).
- **Synthetic data**: every algorithm ships a generator inside `data.py` so
  `example.py` runs offline without proprietary datasets.
- **Smoke-test contract**: `example.py` exits non-zero unless the algorithm
  satisfies a measurable property (Welch's t-test on cluster similarity,
  classification accuracy threshold, reconstruction loss bound, etc.).

### Code style
- Minimal comments — only when the WHY is non-obvious.
- Type hints throughout; `dataclass` for records.
- No `tests/` directory — `example.py` is the smoke test.
- No backward-compat shims, no feature flags, no speculative abstractions.

### What NOT to include
- Proprietary datasets from the source paper.
- Baseline-model comparisons (out of scope — focus on the target algorithm).
- Downstream evaluation tasks (e.g. XGBoost classifiers); the algorithm's
  output embedding is what users plug into their own evaluation.
- Paper PDFs (copyright). Keep them in `.research/` (gitignored).

## Adding a new algorithm

1. Drop the paper PDF in `.research/` (gitignored).
2. Create `algorithms/pytorch-implementation/<name>/` (or `mlx-implementation/`) with the files listed in the layout above.
3. Implement to paper fidelity; cite sections in docstrings.
4. Write a synthetic generator + smoke test that exits 0 on success.
5. Verify: `cd algorithms/pytorch-implementation/<name> && python example.py`.
6. Update the **Existing algorithms** table below.
7. Tag a release: `gh release create vX.Y.0 --generate-notes`.

## Existing algorithms

| Algorithm    | Paper                                                                 | Folder                       |
|--------------|-----------------------------------------------------------------------|------------------------------|
| Geo-Tile2Vec | Luo et al., *ACM TSAS* 9(2) Article 10, 2023 (doi:10.1145/3571741)    | `algorithms/pytorch-implementation/geotile2vec/` |
| TransFlower  | Luo et al., arXiv:2402.15398v1, 2024                                  | `algorithms/pytorch-implementation/transflower/` |
| AAGNN        | Zhou et al., *CIKM '21*, 2021 (doi:10.1145/3459637.3482195)           | `algorithms/pytorch-implementation/aagnn/`       |
| MHGL         | Zhou et al., *SIAM SDM 2022* (unseen anomaly detection on networks)   | `algorithms/pytorch-implementation/mhgl/`        |
| ACDNE        | Shen et al., *AAAI 2020* (cross-network node classification)          | `algorithms/pytorch-implementation/acdne/`       |

## Cross-agent setup

This file is the single source of truth. Tool-specific entry points are
**symlinks** to this file (zero duplication, zero drift):

- `CLAUDE.md` → `AGENTS.md`     (Claude Code)
- `GEMINI.md` → `AGENTS.md`     (Gemini CLI)
- `AGENTS.md` itself is read natively by **Codex**, **OpenCode**, and any
  other agent following the [agents.md](https://agents.md) convention.

To add support for another agent, create a new symlink rather than copying
content. On macOS / Linux:

```bash
ln -s AGENTS.md <NEW_FILE>.md
git add <NEW_FILE>.md && git commit -m "Wire up <agent> to AGENTS.md"
```

On Windows, prefer a one-line stub file containing `@AGENTS.md` (if the
agent supports `@` imports) over a Windows symlink, which requires admin
privileges to create.
