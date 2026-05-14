"""H4-empirical adversarial test.

Run two policies on identical conditions:

* ``B3_l2t_no_q``  — full L2T policy with its internal pedagogical
  guards (``_can_hint`` honours ``hint_requested``;
  ``_concept_theory_pending`` schedules microtheory before tasks).

* ``B3p_l2t_permissive`` — identical action-selection intent but with
  the two guards stripped (see :class:`l2t_sim.policies.PermissiveL2TPolicy`).

We then count past-LTL invariant violations on each policy. If the
permissive variant produces non-zero counts and B3 produces zero,
the guards are empirically necessary — Statement 1's invariant
guarantee is a property of policy design, not of the test setup.

Output: ``results/tables/adversarial_invariant_test.json``.

A note on ``transfer_gate``: the simulation event vocabulary does not
emit ``block_done`` in the T=100 horizon, so the third invariant is
structurally untriggered in both policies (always True). We report
this honestly. Theory-first and no-spoiler are the two invariants
that this test actually exercises.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2t_sim.methodology import (  # noqa: E402
    generate_synthetic_methodology,
    load_probability_methodology,
)
from l2t_sim.simulation import run_simulation  # noqa: E402


N_STUDENTS = 1000
T_STEPS = 100
MASTER_SEED = 42
METHOD_SEED = 1


def _count_violations(records: list) -> dict[str, int]:
    """Tally per-invariant violations across a population of trajectories."""

    counts: Counter[str] = Counter()
    for r in records:
        for inv, held in r.invariants_held.items():
            if not held:
                counts[inv] += 1
    return {
        "theory_first": int(counts.get("theory_first", 0)),
        "no_spoiler": int(counts.get("no_spoiler", 0)),
        "transfer_gate": int(counts.get("transfer_gate", 0)),
    }


def _run_one(policy_name: str, methodology, n_students: int = N_STUDENTS) -> tuple[list, float]:
    t0 = time.perf_counter()
    recs = run_simulation(
        policy_name=policy_name,
        methodology=methodology,
        n_students=n_students,
        t_steps=T_STEPS,
        master_seed=MASTER_SEED,
        n_jobs=-1,
    )
    return recs, time.perf_counter() - t0


def main() -> int:
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    prob = load_probability_methodology(
        ROOT / "methodologies" / "probability_basics.json"
    )
    synth = generate_synthetic_methodology(seed=METHOD_SEED)

    print(
        f"Adversarial invariant test  N={N_STUDENTS}  T={T_STEPS}  seed={MASTER_SEED}",
        flush=True,
    )
    print(
        f"  guarded baseline:   B3_l2t_no_q",
        flush=True,
    )
    print(
        f"  permissive variant: B3p_l2t_permissive (no _can_hint gate, no theory-pending gate)",
        flush=True,
    )

    output: dict = {
        "config": {
            "n_students": N_STUDENTS,
            "t_steps": T_STEPS,
            "master_seed": MASTER_SEED,
            "method_seed": METHOD_SEED,
        },
        "per_methodology": {},
    }

    for meth_name, meth in (("probability", prob), ("synthetic", synth)):
        print(f"\n=== Methodology: {meth_name} ===", flush=True)
        guarded_recs, t_g = _run_one("B3_l2t_no_q", meth)
        guarded = _count_violations(guarded_recs)
        print(
            f"  B3 (guarded):    theory_first={guarded['theory_first']:>5d}  "
            f"no_spoiler={guarded['no_spoiler']:>5d}  "
            f"transfer_gate={guarded['transfer_gate']:>5d}   ({t_g:.1f}s)",
            flush=True,
        )

        permissive_recs, t_p = _run_one("B3p_l2t_permissive", meth)
        permissive = _count_violations(permissive_recs)
        print(
            f"  B3p (permissive):theory_first={permissive['theory_first']:>5d}  "
            f"no_spoiler={permissive['no_spoiler']:>5d}  "
            f"transfer_gate={permissive['transfer_gate']:>5d}   ({t_p:.1f}s)",
            flush=True,
        )

        output["per_methodology"][meth_name] = {
            "B3_l2t_no_q": {**guarded, "n": len(guarded_recs)},
            "B3p_l2t_permissive": {**permissive, "n": len(permissive_recs)},
        }

    # Pooled across methodologies.
    pooled_guarded: Counter[str] = Counter()
    pooled_permissive: Counter[str] = Counter()
    pooled_n = 0
    for cell in output["per_methodology"].values():
        for k in ("theory_first", "no_spoiler", "transfer_gate"):
            pooled_guarded[k] += cell["B3_l2t_no_q"][k]
            pooled_permissive[k] += cell["B3p_l2t_permissive"][k]
        pooled_n += cell["B3_l2t_no_q"]["n"]
    output["pooled"] = {
        "n_per_policy": pooled_n,
        "B3_l2t_no_q": dict(pooled_guarded),
        "B3p_l2t_permissive": dict(pooled_permissive),
    }

    print(
        f"\nPooled (n={pooled_n} per policy):"
        f"\n  B3 :  TF={pooled_guarded['theory_first']}  NS={pooled_guarded['no_spoiler']}  TG={pooled_guarded['transfer_gate']}"
        f"\n  B3p: TF={pooled_permissive['theory_first']}  NS={pooled_permissive['no_spoiler']}  TG={pooled_permissive['transfer_gate']}",
        flush=True,
    )

    out_path = out_dir / "adversarial_invariant_test.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
