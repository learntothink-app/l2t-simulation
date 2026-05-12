"""Sensitivity analysis: sweep environment.THEORY_BONUS.

The headline statistical results (H1b, H3, H4) of v0.1.5 are computed
with ``environment.THEORY_BONUS = 0.15`` — the value we calibrated on
the v0.1.1 ``small`` config. A natural reviewer concern is that this
specific value "builds in" L2T's advantage. This script sweeps
THEORY_BONUS ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25} and re-evaluates
the pooled hypotheses at each value.

Usage::

    python experiments/sensitivity_theory_bonus.py

Output: ``results/sensitivity/theory_bonus_sweep.json`` plus a summary
table printed to stdout. The summary table is reproduced as Appendix B
of the paper.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2t_sim import environment  # noqa: E402
from l2t_sim.analysis import test_h1b, test_h2, test_h3, test_h4  # noqa: E402
from l2t_sim.methodology import (  # noqa: E402
    generate_synthetic_methodology,
    load_probability_methodology,
)
from l2t_sim.simulation import run_simulation  # noqa: E402


THEORY_BONUS_VALUES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
N_STUDENTS = 1000
T_STEPS = 100
MASTER_SEED = 42
METHOD_SEED = 1
N_JOBS = -1

POLICY_NAMES = ("B1_random", "B2_bkt", "B3_l2t_no_q", "B4_full_l2t")


def main() -> int:
    out_dir = ROOT / "results" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the original value so the sweep is reversible.
    original_theory_bonus = environment.THEORY_BONUS

    results: list[dict] = []
    try:
        for tb in THEORY_BONUS_VALUES:
            print(f"\n=== THEORY_BONUS = {tb:.2f} ===", flush=True)
            environment.THEORY_BONUS = tb

            # Reload methodologies for each value so any cached state is
            # rebuilt fresh (the methodology itself does not depend on
            # THEORY_BONUS, but reload is cheap and removes any doubt).
            prob = load_probability_methodology(
                ROOT / "methodologies" / "probability_basics.json"
            )
            synth = generate_synthetic_methodology(seed=METHOD_SEED)

            pooled: dict[str, list] = {p: [] for p in POLICY_NAMES}
            t0 = time.perf_counter()
            for meth_name, meth in (("probability", prob), ("synthetic", synth)):
                for pol_name in POLICY_NAMES:
                    recs = run_simulation(
                        policy_name=pol_name,
                        methodology=meth,
                        n_students=N_STUDENTS,
                        t_steps=T_STEPS,
                        master_seed=MASTER_SEED,
                        n_jobs=N_JOBS,
                    )
                    pooled[pol_name].extend(recs)
            elapsed = time.perf_counter() - t0
            print(f"  simulated in {elapsed:.1f}s", flush=True)

            h1b = test_h1b(pooled["B1_random"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
            h2 = test_h2(pooled["B2_bkt"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
            h3 = test_h3(
                pooled["B1_random"],
                pooled["B2_bkt"],
                pooled["B3_l2t_no_q"],
                pooled["B4_full_l2t"],
            )
            h4 = test_h4(pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])

            def _get(cell: dict, key: str, default=None):
                if cell is None:
                    return default
                v = cell.get(key)
                return v if v is not None else default

            row = {
                "theory_bonus": tb,
                "H1b_cohen_d": h1b.get("cohen_d"),
                "H1b_p_value": h1b.get("p_value"),
                "H1b_confirmed": h1b["confirmed"],
                "H3_mastery_b2_d": _get(h3.get("m_mastery_b2_vs_b1"), "cohen_d"),
                "H3_mastery_b3_d": _get(h3.get("m_mastery_b3_vs_b1"), "cohen_d"),
                "H3_mastery_b4_d": _get(h3.get("m_mastery_b4_vs_b1"), "cohen_d"),
                "H3_transfer_b3_d": _get(h3.get("m_transfer_b3_vs_b1"), "cohen_d"),
                "H3_transfer_b4_d": _get(h3.get("m_transfer_b4_vs_b1"), "cohen_d"),
                "H3_mastery_b3_confirmed": _get(h3.get("m_mastery_b3_vs_b1"), "confirmed"),
                "H3_mastery_b4_confirmed": _get(h3.get("m_mastery_b4_vs_b1"), "confirmed"),
                "H3_transfer_b3_confirmed": _get(h3.get("m_transfer_b3_vs_b1"), "confirmed"),
                "H3_transfer_b4_confirmed": _get(h3.get("m_transfer_b4_vs_b1"), "confirmed"),
                "H4_violations_b3": h4["violations_b3"],
                "H4_violations_b4": h4["violations_b4"],
                "H4_confirmed": h4["h4_confirmed"],
            }
            results.append(row)
            print(
                f"  H1b d={row['H1b_cohen_d']:+.3f} | "
                f"H3 mast B3 d={row['H3_mastery_b3_d']:+.3f} | "
                f"H3 trans B3 d={row['H3_transfer_b3_d']:+.3f} | "
                f"H4 viol B3={row['H4_violations_b3']}",
                flush=True,
            )
    finally:
        environment.THEORY_BONUS = original_theory_bonus

    out_path = out_dir / "theory_bonus_sweep.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(
        f"{'theory_bonus':>14s} {'H1b d':>8s} {'mast B3 d':>11s} "
        f"{'mast B4 d':>11s} {'trans B3 d':>11s} {'trans B4 d':>11s} "
        f"{'H4 viol B3':>11s} {'H4 viol B4':>11s}",
        flush=True,
    )
    for r in results:
        h1b_d = f"{r['H1b_cohen_d']:+.3f}" if r["H1b_cohen_d"] is not None else "  -    "
        print(
            f"{r['theory_bonus']:>14.2f} {h1b_d:>8s} "
            f"{r['H3_mastery_b3_d']:>+11.3f} {r['H3_mastery_b4_d']:>+11.3f} "
            f"{r['H3_transfer_b3_d']:>+11.3f} {r['H3_transfer_b4_d']:>+11.3f} "
            f"{r['H4_violations_b3']:>11d} {r['H4_violations_b4']:>11d}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
