# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each tagged release ships one from-scratch algorithm implementation under
`algorithms/<framework>/<name>/`. The `Unreleased` section accumulates work
landed on `main` between tag cuts.

## [Unreleased]

## [0.3.0] — 2026-04-29

### Added
- **AAGNN** (Abnormality-Aware Graph Neural Network) PyTorch implementation
  under `algorithms/pytorch-implementation/aagnn/` — Zhou et al.,
  *CIKM '21* (doi:10.1145/3459637.3482195).
  - `data.py` — graph dataclasses and synthetic stochastic-block-model
    generator with planted anomalies.
  - `layer.py` — abnormality-aware GNN layer (paper §3.1).
  - `train.py` — hypersphere objective and end-to-end training loop
    (paper §3.2–3.3, Algorithm 1).
  - `model.py` — top-level orchestrator class wrapping the layer and
    training loop.
  - `example.py` — synthetic-SBM smoke test enforcing an ROC-AUC ≥ 0.75
    contract on planted anomalies.
  - `README.md` — paper reference, theory recap, and run instructions.
- AAGNN row added to the **Existing algorithms** table in `AGENTS.md`.

## [0.2.0] — 2026-04-28

### Added
- **TransFlower** PyTorch implementation under
  `algorithms/pytorch-implementation/transflower/` — Luo et al.,
  arXiv:2402.15398v1 (2024).

### Changed
- Restructured `algorithms/` into framework-segregated subdirectories
  (`pytorch-implementation/`, `mlx-implementation/`) and moved
  `geotile2vec/` under `pytorch-implementation/`.
- Made the workspace agent-agnostic via a single `AGENTS.md` source of
  truth, with `CLAUDE.md` and `GEMINI.md` as symlinks pointing at it.

## [0.1.0] — 2026-04-27

### Added
- **Geo-Tile2Vec** PyTorch implementation under
  `algorithms/pytorch-implementation/geotile2vec/` — Luo et al.,
  *ACM TSAS* 9(2) Article 10, 2023 (doi:10.1145/3571741).
- Initial repo scaffolding: `AGENTS.md` conventions, `.gitignore`
  excluding `.research/` paywalled PDFs, and the `algorithms/` layout.

[Unreleased]: https://github.com/SpectrumTantrum/supernova/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/SpectrumTantrum/supernova/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/SpectrumTantrum/supernova/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/SpectrumTantrum/supernova/releases/tag/v0.1.0
