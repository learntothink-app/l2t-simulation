# L2T Simulation

Implementation of the simulation protocol from §VII of *LearnToThink: Structuring Adaptive Tutoring via Typed Hypergraphs and Verifiable Pedagogy* (Yusupova, Popov, Smorchkov, Chuprov; 2026).

The harness produces empirical numbers for hypotheses **H1–H4** and verifies the
three past-LTL pedagogical invariants by model-checking the controller FSM
defined in §V.B.

## Install

```bash
python -m pip install -e .
```

## Quick run

```bash
# 10s smoke test
python experiments/run_main.py --config experiments/configs/mini.yaml

# Full run (N=1000, T=100, 4 policies × 2 methodologies)
python experiments/run_main.py --config experiments/configs/medium.yaml

# Model-check invariants on the controller FSM (no simulation)
python experiments/run_model_checking.py

# Sensitivity to hyperedge type weights w_kappa (±20%)
python experiments/run_sensitivity.py
```

Results land in `results/` (tables, figures, `report.html`, raw trajectories)
and a LaTeX fragment for §VIII is written to `docs/paper_section_VIII.tex`.

The exact seeds, hyperparameters, and library versions used for any given run
are dumped to `experiment_manifest.json` next to the results.

## Tests

```bash
pytest
```

The integration test (`tests/test_integration.py`) drives all four policies on
both methodologies with `N=10, T=20` end-to-end in a few seconds and is the
single most useful smoke test.

## Mapping from paper formulas to code

See `docs/formulas_to_code_map.md`.
