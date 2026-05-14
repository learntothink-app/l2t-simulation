"""Sanity diagnostic: why does B4 (full q_t) flip direction on synthetic
m_inference_robust vs B3 (q_t = 1)?

Two complementary traces on the SAME guesser student, same synthetic
methodology (seed=1), same RNG streams for the environment:

  trace_B3: run L2TPolicy(use_reliability=False) — q_t=1 in the
            Bayesian update.
  trace_B4: replay the exact (action, observation, q_t) stream from B3
            *through B4's belief update*, so the only difference is the
            q_t weight applied at update time.

This isolates the q_t effect from any difference in action selection.
If B4's belief diverges from p_true_t more than B3's, the heuristic
q_t under-weights informative observations on contaminated streams.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2t_sim.environment import SyntheticStudent  # noqa: E402
from l2t_sim.methodology import generate_synthetic_methodology  # noqa: E402
from l2t_sim.policies import Context, L2TPolicy  # noqa: E402
from l2t_sim.reliability import compute_reliability  # noqa: E402
from l2t_sim.student_model import StudentModel  # noqa: E402
from l2t_sim.simulation import _baseline_time_for  # noqa: E402


def main() -> None:
    methodology = generate_synthetic_methodology(seed=1)
    student_seed = 12345

    # --- Run B3 (use_reliability=False, q_t baked to 1.0 in update) -------
    rng_env_b3 = np.random.default_rng(student_seed)
    rng_pol_b3 = np.random.default_rng(student_seed + 1)
    student = SyntheticStudent.sample(
        methodology=methodology, behaviour="guesser", rng=rng_env_b3
    )
    policy_b3 = L2TPolicy(
        methodology.hypergraph, methodology, rng_pol_b3, use_reliability=False
    )
    model_b3 = StudentModel(methodology.hypergraph, methodology, use_reliability=False)
    belief_b3 = model_b3.initial_belief()
    ctx_b3 = Context(t=0, methodology=methodology)

    # We will record the exact (action, observation, q_t) stream for replay.
    stream: list = []
    abs_err_b3: list[float] = []
    abs_err_b3.append(float(np.mean(np.abs(belief_b3.p_hat - student.p_true))))

    T = 100
    for t in range(T):
        ctx_b3.t = t
        action = policy_b3.select_action(belief_b3, ctx_b3)
        observation = student.observe(action)
        q_t = compute_reliability(action, observation, _baseline_time_for(action, methodology))
        new_belief = model_b3.bayes_update(belief_b3, action, observation, q_t)
        student.update_p_true(action, observation)
        stream.append((action, observation, q_t))
        ctx_b3.last_observation = observation
        if action.type in ("task", "drill", "probe", "transfer"):
            ctx_b3.current_item = action.target_id
        # bookkeeping for L2TPolicy's internal context
        if action.type == "microtheory":
            mt = methodology.microtheories.get(action.target_id)
            if mt is not None:
                for c in mt.concepts:
                    ctx_b3.theory_checked.add(c)
        belief_b3 = new_belief
        abs_err_b3.append(float(np.mean(np.abs(belief_b3.p_hat - student.p_true))))

    p_true_final = student.p_true.copy()

    # --- Replay the same stream through B4 (use_reliability=True) ---------
    model_b4 = StudentModel(methodology.hypergraph, methodology, use_reliability=True)
    belief_b4 = model_b4.initial_belief()
    # Use the SAME ground truth as B3 reached, so the inference target is
    # identical. We do NOT regenerate the student — we just replay updates.
    abs_err_b4: list[float] = []
    abs_err_b4.append(float(np.mean(np.abs(belief_b4.p_hat - p_true_final))))
    for action, observation, q_t in stream:
        belief_b4 = model_b4.bayes_update(belief_b4, action, observation, q_t)
        abs_err_b4.append(float(np.mean(np.abs(belief_b4.p_hat - p_true_final))))

    # --- Summary ---------------------------------------------------------
    print(f"Student behaviour: guesser (seed {student_seed})")
    print(f"Methodology: synthetic (seed 1)")
    print(f"T = {T} steps; replayed identical (action, observation, q_t) stream into B4.")
    print()
    print(f"Mean q_t across the {T}-step stream: "
          f"{float(np.mean([q for _, _, q in stream])):.3f}")
    print(f"Stream q_t distribution (deciles):")
    qs = sorted(q for _, _, q in stream)
    for k in (0, 10, 25, 50, 75, 90, 100):
        idx = min(len(qs) - 1, int(k / 100.0 * len(qs)))
        print(f"  p{k:>3d}: {qs[idx]:.3f}")
    print()
    print("Step  abs_err_B3   abs_err_B4   Δ(B4-B3)")
    for t in (0, 1, 5, 10, 20, 30, 50, 75, 100):
        if t > T:
            continue
        print(f"  {t:>3d}    {abs_err_b3[t]:.4f}       {abs_err_b4[t]:.4f}       {abs_err_b4[t] - abs_err_b3[t]:+.4f}")

    print()
    print(f"Final abs_err: B3 = {abs_err_b3[-1]:.4f},  B4 = {abs_err_b4[-1]:.4f}")
    print(f"Per-skill final |b_T - p_true_T|:")
    print(f"  B3: {np.round(np.abs(belief_b3.p_hat - p_true_final), 3).tolist()}")
    print(f"  B4: {np.round(np.abs(belief_b4.p_hat - p_true_final), 3).tolist()}")
    print()
    print(f"Final beliefs:")
    print(f"  p_true:  {np.round(p_true_final, 3).tolist()}")
    print(f"  b_T B3:  {np.round(belief_b3.p_hat, 3).tolist()}")
    print(f"  b_T B4:  {np.round(belief_b4.p_hat, 3).tolist()}")


if __name__ == "__main__":
    main()
