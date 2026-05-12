"""Sensitivity analysis: sweep environment.THEORY_BONUS.

The headline statistical results (H1b, H3, H4) of v0.1.5 are computed
with ``environment.THEORY_BONUS = 0.15`` — the value we calibrated on
the v0.1.1 ``small`` config. A natural reviewer concern is that this
specific value "builds in" L2T's advantage. This script sweeps
THEORY_BONUS ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25} and re-evaluates
the pooled hypotheses at each value.

Implementation note: environment.THEORY_BONUS is read from the
L2T_THEORY_BONUS env var at module import. joblib workers spawned by
run_simulation inherit env vars from the parent process, so we set the
env var *before* invoking each subprocess. In-process monkey-patching
would NOT propagate to workers (each worker reimports the module with
the original default).

Usage::

    python experiments/sensitivity_theory_bonus.py

Output: ``results/sensitivity/theory_bonus_sweep.json`` plus a summary
table printed to stdout. The summary table is reproduced as Appendix B
of the paper.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
THEORY_BONUS_VALUES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)


# ---------------------------------------------------------------------------
# Child mode: runs one sweep cell and prints a single JSON object on stdout.


def _run_one_cell() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from l2t_sim import environment  # noqa: E402
    from l2t_sim.analysis import test_h1b, test_h2, test_h3, test_h4  # noqa: E402
    from l2t_sim.methodology import (  # noqa: E402
        generate_synthetic_methodology,
        load_probability_methodology,
    )
    from l2t_sim.simulation import run_simulation  # noqa: E402

    tb = environment.THEORY_BONUS
    assert tb == float(os.environ["L2T_THEORY_BONUS"]), (
        "module-level THEORY_BONUS did not pick up env var override"
    )

    prob = load_probability_methodology(
        ROOT / "methodologies" / "probability_basics.json"
    )
    synth = generate_synthetic_methodology(seed=1)

    pooled: dict[str, list] = {p: [] for p in ("B1_random", "B2_bkt", "B3_l2t_no_q", "B4_full_l2t")}
    for meth in (prob, synth):
        for pol in pooled:
            recs = run_simulation(
                policy_name=pol,
                methodology=meth,
                n_students=1000,
                t_steps=100,
                master_seed=42,
                n_jobs=-1,
            )
            pooled[pol].extend(recs)

    h1b = test_h1b(pooled["B1_random"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
    h2 = test_h2(pooled["B2_bkt"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
    h3 = test_h3(
        pooled["B1_random"],
        pooled["B2_bkt"],
        pooled["B3_l2t_no_q"],
        pooled["B4_full_l2t"],
    )
    h4 = test_h4(pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])

    def _get(cell: dict | None, key: str):
        if cell is None:
            return None
        v = cell.get(key)
        return v

    return {
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


# ---------------------------------------------------------------------------
# Parent mode: orchestrates sweep across THEORY_BONUS values.


def _parent() -> int:
    out_dir = ROOT / "results" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for tb in THEORY_BONUS_VALUES:
        print(f"\n=== THEORY_BONUS = {tb:.2f} ===", flush=True)
        env = dict(os.environ)
        env["L2T_THEORY_BONUS"] = f"{tb:.4f}"
        env["L2T_SENSITIVITY_CHILD"] = "1"
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(SELF)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(proc.stdout, flush=True)
            print(proc.stderr, file=sys.stderr, flush=True)
            raise SystemExit(f"child for THEORY_BONUS={tb} failed with {proc.returncode}")

        # Child emits one JSON line as last line of stdout.
        line = proc.stdout.strip().splitlines()[-1]
        row = json.loads(line)
        results.append(row)
        h1b_d = row["H1b_cohen_d"]
        h1b_str = f"{h1b_d:+.3f}" if h1b_d is not None else "  -    "
        print(
            f"  simulated in {elapsed:.1f}s | "
            f"H1b d={h1b_str} | "
            f"mast B3 d={row['H3_mastery_b3_d']:+.3f} | "
            f"trans B3 d={row['H3_transfer_b3_d']:+.3f} | "
            f"H4 viol B3={row['H4_violations_b3']}",
            flush=True,
        )

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


def main() -> int:
    if os.environ.get("L2T_SENSITIVITY_CHILD") == "1":
        # Child path: emit one JSON line.
        row = _run_one_cell()
        print(json.dumps(row), flush=True)
        return 0
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
