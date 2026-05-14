"""Replicates sanity_b4_inference.py over 30 random student seeds.

For each guesser, replays B3's (action, observation, q_t) stream into
B4's belief update. If q_t weight is structurally hurting inference,
B4 final MAE should be > B3 final MAE in the majority of seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2t_sim.environment import SyntheticStudent
from l2t_sim.methodology import generate_synthetic_methodology
from l2t_sim.policies import Context, L2TPolicy
from l2t_sim.reliability import compute_reliability
from l2t_sim.simulation import _baseline_time_for
from l2t_sim.student_model import StudentModel


def run_one(methodology, student_seed: int, behaviour: str) -> tuple[float, float]:
    rng_env = np.random.default_rng(student_seed)
    rng_pol = np.random.default_rng(student_seed + 1)
    student = SyntheticStudent.sample(methodology=methodology, behaviour=behaviour, rng=rng_env)
    policy_b3 = L2TPolicy(methodology.hypergraph, methodology, rng_pol, use_reliability=False)
    model_b3 = StudentModel(methodology.hypergraph, methodology, use_reliability=False)
    belief_b3 = model_b3.initial_belief()
    ctx = Context(t=0, methodology=methodology)
    stream = []
    for t in range(100):
        ctx.t = t
        a = policy_b3.select_action(belief_b3, ctx)
        o = student.observe(a)
        q = compute_reliability(a, o, _baseline_time_for(a, methodology))
        belief_b3 = model_b3.bayes_update(belief_b3, a, o, q)
        student.update_p_true(a, o)
        stream.append((a, o, q))
        ctx.last_observation = o
        if a.type in ("task", "drill", "probe", "transfer"):
            ctx.current_item = a.target_id
        if a.type == "microtheory":
            mt = methodology.microtheories.get(a.target_id)
            if mt is not None:
                for c in mt.concepts:
                    ctx.theory_checked.add(c)
    p_true_final = student.p_true.copy()
    mae_b3 = float(np.mean(np.abs(belief_b3.p_hat - p_true_final)))

    model_b4 = StudentModel(methodology.hypergraph, methodology, use_reliability=True)
    belief_b4 = model_b4.initial_belief()
    for a, o, q in stream:
        belief_b4 = model_b4.bayes_update(belief_b4, a, o, q)
    mae_b4 = float(np.mean(np.abs(belief_b4.p_hat - p_true_final)))
    return mae_b3, mae_b4


def main() -> None:
    import json

    methodology = generate_synthetic_methodology(seed=1)
    out: dict = {
        "description": (
            "Isolated test of q_t: B3's (action, observation, q_t) stream "
            "is replayed through B4's belief update on the same student. "
            "Δ(B4-B3) > 0 ⇒ q_t weight degrades inference on this stream."
        ),
        "seeds": list(range(1000, 1030)),
        "methodology_seed": 1,
        "per_behaviour": {},
    }
    for behaviour in ("guesser", "copier"):
        diffs = []
        b3s = []
        b4s = []
        for seed in out["seeds"]:
            mae_b3, mae_b4 = run_one(methodology, seed, behaviour)
            diffs.append(mae_b4 - mae_b3)
            b3s.append(mae_b3)
            b4s.append(mae_b4)
        diffs_arr = np.array(diffs)
        n_pos = int(np.sum(diffs_arr > 0))
        print(f"\n=== {behaviour} ({len(diffs)} seeds, isolated q_t test on identical stream) ===")
        print(f"  mean MAE B3: {np.mean(b3s):.4f}")
        print(f"  mean MAE B4: {np.mean(b4s):.4f}")
        print(f"  Δ(B4-B3) mean: {diffs_arr.mean():+.4f}")
        print(f"  Δ(B4-B3) std:  {diffs_arr.std(ddof=1):+.4f}")
        print(f"  seeds where B4 worse: {n_pos}/{len(diffs)}")
        if diffs_arr.mean() > 0.02:
            interp = (
                "q_t over-downweights informative correctness signal: "
                "behaviour flag triggers downweight even when the observation "
                "itself carries valid mastery-correlated information."
            )
        elif diffs_arr.mean() < -0.02:
            interp = (
                "q_t correctly downweights noise: observations under this "
                "behaviour carry little mastery information; downweighting "
                "stabilises the belief."
            )
        else:
            interp = "q_t makes essentially no difference on this stream."
        out["per_behaviour"][behaviour] = {
            "B3_MAE_mean": float(np.mean(b3s)),
            "B4_MAE_mean": float(np.mean(b4s)),
            "delta_B4_minus_B3_mean": float(diffs_arr.mean()),
            "delta_B4_minus_B3_std": float(diffs_arr.std(ddof=1)),
            "seeds_B4_worse": f"{n_pos}/{len(diffs)}",
            "interpretation": interp,
        }

    out_path = ROOT / "scripts" / "sanity_b4_inference_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
