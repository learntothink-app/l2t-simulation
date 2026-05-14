# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-14

Methodological revisions per internal peer-review feedback on §VIII.
**Numerical results differ from v0.1.6** — paper v2 (forthcoming)
will cite v0.2.0; paper v1 remains pinned to tag v0.1.6.

### Added
- `m_inference_robust` metric: mean absolute error between final
  per-skill belief `b_T` and ground-truth `p_true_T`, restricted to
  contaminated students (guesser / copier). Isolates inference
  quality from action-selection, unlike `m_selection_robust`. Wired
  through `TrajectoryMetrics`, `aggregate_table`, `test_h3` (with
  lower-is-better direction flip), and a new
  `test_inference_robust_b4_vs_b3` ablation.
- Synthetic methodology is now a K-graph ensemble (default K=30
  in `medium.yaml`). Per-graph means feed
  `results/tables/synthetic_per_graph.csv` and
  `synthetic_aggregate.csv` (mean ± std over K). Trajectories are
  pooled across graphs for hypothesis tests.
- `PermissiveL2TPolicy` — adversarial baseline that inherits B3's
  action-selection intent but strips the two internal pedagogical
  guards (`_can_hint`, `_concept_theory_pending`).
- `experiments/adversarial_invariant_test.py` — H4-empirical test
  that compares guarded vs permissive L2T on past-LTL violation
  counts. Writes `results/tables/adversarial_invariant_test.json`.
- `results/tables/inference_robust.json` — dedicated report with
  pooled / per-methodology / per-behaviour decomposition of
  m_inference_robust, plus the isolated-stream sanity diagnostic.
- `scripts/sanity_b4_inference_multi.py` — 30-seed diagnostic
  replaying B3's `(action, observation, q_t)` stream through B4's
  belief update to isolate the q_t weight effect from action
  selection.

### Changed
- **BREAKING (metric semantics):** `m_robust` is preserved as a
  backward-compatibility alias of the new `m_selection_robust`. Old
  callers continue to compute the same selection-only quantity, but
  papers should disambiguate when citing.
- `test_h3` now includes `m_inference_robust` with lower-is-better
  direction handling. Cohen's d is reported with positive sign
  meaning "policy beats B1", so the `d > 0.2` threshold works
  uniformly across all four metrics.

### Findings
- **Original H1 resolved (B4 vs B3 isolated ablation, pooled
  N=8215 contaminated students):** cohen_d = -1.03, p = 1.0. The
  heuristic q_t of Eq. (23) provides no isolated benefit over
  rule-based L2T; in fact it actively degrades inference quality.
- **Behaviour-specific failure mode of q_t:**
  - guesser (n=5053): B4 vs B1 d = +0.77, p < 10⁻³⁰⁶ — q_t
    correctly downweights noise.
  - copier (n=3162): B4 vs B1 d = -5.54, p = 1.0 — q_t
    catastrophically over-downweights informative correctness
    signal.
- **H4 empirical (Task 2):** permissive L2T produces 2000/1612/0
  (theory-first / no-spoiler / transfer-gate) violations across
  N=2000 trajectories vs 0/0/0 for guarded B3. Guards do real work.
- **H3 m_mastery (synthetic K=30 ensemble):** B3 - B1 = +0.053 ±
  0.014; B4 - B1 = +0.056 ± 0.014; B2 (BKT) - B1 = +0.041 ± 0.009.
  Adaptive policies confirmed > Random; topology variability small.

### Deferred to v0.2.1
- Task 3 (fixed challenge battery for H1) — original H1 question
  is now answered with d=-1.03 from Task 1; the battery becomes
  cross-design triangulation rather than primary test.
- Task 5 (B5 prompt-tuned LLM baseline for Statement 1 §IX.A.6) —
  pending Anthropic API budget approval.

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
