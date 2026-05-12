# Contributing to L2T Simulation

Thank you for your interest in contributing to L2T Simulation, the
reproducible harness for the LearnToThink framework.

## Scope

This repository implements the simulation protocol described in §VII of
the L2T paper (Yusupova, Popov, Smorchkov, Chuprov; 2026). Contributions
are welcome in three categories:

1. **Reproducibility fixes** — bug reports and patches that improve the
   reproducibility of paper results.
2. **New methodologies** — L2T-conformant methodologies in additional
   domains (linear algebra, Newtonian mechanics, etc.) that extend the
   §VIII.D structural argument.
3. **Trained-classifier `q_t` replacement** — implementation of the
   supervised reliability classifier sketched in §III.F and
   Appendix E.

## Code style

- Python 3.11+; type hints required on public APIs.
- `ruff` for linting, `black` for formatting (line length 100).
- Tests via `pytest`; new features require at least one test.

## How to submit changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Run tests: `pytest`.
4. Run linting (if installed locally): `ruff check . && black --check .`.
5. Open a pull request with a clear description and reference to any
   paper section the change touches.

## Reporting issues

Use GitHub Issues with the appropriate label:

- `bug` — unexpected behaviour.
- `reproducibility` — paper-result regression (please include the
  config, seed, and observed-vs-expected numbers).
- `enhancement` — new feature.
- `documentation` — README, docstrings, formulas-to-code mapping.

## License

By contributing, you agree that your contributions will be licensed
under the Apache License 2.0 (see `LICENSE`).
