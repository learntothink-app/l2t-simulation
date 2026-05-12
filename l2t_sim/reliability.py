"""Reliability score q_t (Eq. 23 of the paper).

The four indicators encode the four well-documented forms of inflated
correctness in digital tutoring: gaming-the-system (``too_fast``), absent
reasoning (``no_explanation``), template copying (``copying_pattern``), and
its positive counterpart (``self_explanation``).
"""

from __future__ import annotations

from .core_types import Action, Observation


# Coefficients (§VII; expert-set per the §App.E refinement plan).
ALPHA: float = 0.40  # too_fast
BETA: float = 0.20   # no_explanation
GAMMA: float = 0.50  # copying_pattern
DELTA: float = 0.30  # self_explanation
Q_MIN: float = 0.05  # floor for SNIPS variance bound (Lemma 3)


def compute_reliability(
    action: Action,
    observation: Observation,
    action_baseline_time: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    q_min: float = Q_MIN,
) -> float:
    """Heuristic plug-in for q_t* (Eq. 23).

        q_t = clip_{[q_min, 1]}( 1
                    - α·1[too_fast]
                    - β·1[no_explanation]
                    - γ·1[copying_pattern]
                    + δ·1[self_explanation] ).
    """

    # Non-evaluative actions have no informativeness gradient -> mid value.
    if action.type in ("microtheory", "retention", "hint"):
        return 1.0  # these contribute no model update anyway

    too_fast = observation.time_seconds < 0.3 * max(action_baseline_time, 1e-3)
    no_explanation = not observation.self_explanation
    copying = observation.copying_pattern
    self_expl = observation.self_explanation

    q = 1.0
    q -= alpha * (1.0 if too_fast else 0.0)
    q -= beta * (1.0 if no_explanation else 0.0)
    q -= gamma * (1.0 if copying else 0.0)
    q += delta * (1.0 if self_expl else 0.0)
    return float(max(q_min, min(1.0, q)))
