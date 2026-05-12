# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] — 2026-05-12

### Added
- Reproducibility manifest (`results/experiment_manifest.json`) recording
  git hash, master seed, hyperparameters, and library versions per run.
- Sensitivity sweep on `THEORY_BONUS` environment parameter
  (`experiments/sensitivity_theory_bonus.py`); summary table is
  Appendix B of the paper.
- `experiments/main.py` paper-cited entry point with `--config <name>`
  shortcut and `--seed` override.
- `LICENSE` (Apache-2.0) and `CITATION.cff` files for academic
  redistribution.

### Changed
- Reliability threshold `q_low` raised from 0.30 to 0.50 for the
  contaminated subsample so that guesser learners contribute to
  `m_robust` (see paper §VIII.G item 4).
- `THEORY_BONUS` is now read from the `L2T_THEORY_BONUS` environment
  variable at module import. Default value is unchanged (0.15); the
  override is required because joblib worker processes reimport the
  module fresh and would otherwise miss any in-process patching.
- Terminology in docstrings and report templates: "synthetic
  students/learners" → "simulated students/learners" (paper
  synchronization). The "synthetic methodology" remains the proper
  name of the random-graph control methodology.

### Fixed
- Transfer-gate over-trigger when a single early-mastered skill blocked
  further task selection. Replaced with a "≥70% of eligible skills
  mastered" criterion (paper §VIII.G item 2).
- `trains` hyperedge under-population in probability methodology
  (8 → ~50 edges, paper §VIII.G item 1).
- Manifest git_hash mismatch from out-of-band rerun (`v0.1.6.1`).

## [0.1.5] — 2026-05-12

### Changed
- Probability methodology rebalance: ≥3 training tasks per skill, mean
  1.0 required skills per task (paper §VIII.G item 3). 12 new tasks
  added for previously under-represented skills.

## [0.1.4] — 2026-05-12

### Fixed
- L2T transfer-gate now requires a majority of eligible skills mastered
  (default `transfer_phase_threshold = 0.70`) before entering the
  transfer phase, rather than a single mastered skill.

## [0.1.3] — 2026-05-12

### Fixed
- Initial `trains` hyperedge population in probability methodology
  (the v0.1.2 JSON declared metaskill edges in multi-tail form which
  collapsed propagation weight; expanded to single-tail form).

## [0.1.2] — 2026-05-12

### Added
- `methodologies/probability_basics.json` — canonical introductory
  probability methodology, replacing the Fano case study.
- `experiments/run_main.py` switched to two-methodology pooled testing
  on probability + synthetic.

### Removed
- `methodologies/fano.json` — the 12-task Fano-coding fixture had too
  little statistical power for any hypothesis test.

## [0.1.1] — 2026-05-12

### Fixed
- L2T `transfer_threshold` raised from 0.70 to 0.90 to stop the policy
  from spending 60% of actions on transfer items.
- Holdout transfer battery introduced to prevent contamination of the
  terminal `m_transfer` measurement by in-loop drilling.
- New hypothesis test `H1b` pools B3+B4 against B1 on `m_robust`.

## [0.1.0] — 2026-05-12

### Added
- Initial release of the simulation harness implementing §V.B + §VII.D
  of the L2T paper: typed knowledge hypergraph, reliability-weighted
  Bayesian belief update (Eqs. 4, 23), four policies B1–B4, 16-state
  pedagogical FSM with past-LTL invariants, and eleven operational
  metrics (`m_mastery`, `m_transfer`, `m_retention`, `m_hint`,
  `m_robust`, …).
- Two methodologies (Fano-plane case study, synthetic abstract
  generator) and the medium / mini configs.
