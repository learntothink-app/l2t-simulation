"""Sensitivity analysis: perturb the hyperedge type weights w_κ by ±20%
and observe the effect on m_transfer / m_robust under B4.

Usage::

    python experiments/run_sensitivity.py --n-students 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from l2t_sim.analysis import bootstrap_ci
from l2t_sim.hypergraph import DEFAULT_TYPE_WEIGHTS
from l2t_sim.methodology import generate_synthetic_methodology
from l2t_sim.simulation import run_simulation
from l2t_sim.utils import setup_logging


def _run_with_weights(md, weights_scale: float, n_students: int, t_steps: int, seed: int):
    # Replace methodology.hypergraph.type_weights in place.
    md.hypergraph.type_weights = {
        k: v * weights_scale for k, v in DEFAULT_TYPE_WEIGHTS.items()
    }
    return run_simulation(
        policy_name="B4_full_l2t",
        methodology=md,
        n_students=n_students,
        t_steps=t_steps,
        master_seed=seed,
        n_jobs=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-students", type=int, default=200)
    parser.add_argument("--t-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("INFO")
    log = logging.getLogger("sensitivity")

    md = generate_synthetic_methodology(seed=1)
    summary = {}
    for label, scale in (("baseline", 1.0), ("minus20", 0.8), ("plus20", 1.2)):
        log.info("running B4 with w_κ × %.2f", scale)
        t0 = time.perf_counter()
        # rebuild methodology each round so weight changes propagate
        md = generate_synthetic_methodology(seed=1)
        recs = _run_with_weights(md, scale, args.n_students, args.t_steps, args.seed)
        secs = time.perf_counter() - t0
        m_transfer = [r.m_transfer for r in recs]
        m_robust = [r.m_robust for r in recs if r.m_robust == r.m_robust]
        summary[label] = {
            "seconds": secs,
            "n": len(recs),
            "m_transfer_mean_ci": bootstrap_ci(m_transfer, n_resamples=2000, seed=args.seed),
            "m_robust_mean_ci": bootstrap_ci(m_robust, n_resamples=2000, seed=args.seed) if m_robust else None,
        }
        log.info("  done in %.1fs", secs)

    # Compare relative deltas vs. baseline.
    base_t = summary["baseline"]["m_transfer_mean_ci"][0]
    deltas = {}
    for label, payload in summary.items():
        if label == "baseline":
            continue
        new_t = payload["m_transfer_mean_ci"][0]
        deltas[label] = (new_t - base_t) / max(abs(base_t), 1e-9)
    summary["m_transfer_relative_delta"] = deltas
    print(json.dumps(summary, indent=2, default=str))
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
