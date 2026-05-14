"""Tests for the operational metrics (Eqs. 30–34)."""

from __future__ import annotations

import math

import numpy as np

from l2t_sim.core_types import Action, Observation, StepRecord
from l2t_sim.metrics import (
    m_hint,
    m_inference_robust,
    m_mastery,
    m_robust,
    m_selection_robust,
    m_transfer,
)


def _step(target: str, correct: bool, action_type: str = "task", q_t: float = 1.0,
          hint_level: int = 0) -> StepRecord:
    return StepRecord(
        t=0,
        action=Action(action_type, target, hint_level=hint_level),
        observation=Observation(
            correct=correct,
            answer_format="open_text",
            time_seconds=60.0,
            hint_requested=False,
            n_hints_used=hint_level,
            hint_level_reached=hint_level,
            self_explanation=True,
            copying_pattern=False,
        ),
        q_t=q_t,
        p_hat=[0.5],
        p_true=[0.5],
        fsm_state="",
    )


def test_m_mastery_zero_for_no_change() -> None:
    p = np.array([0.5, 0.5, 0.5])
    assert m_mastery(p, p) == 0.0


def test_m_transfer_all_correct() -> None:
    transfer = {"TR_0", "TR_1"}
    rs = [_step("TR_0", True, "transfer"), _step("TR_1", True, "transfer")]
    assert m_transfer(rs, transfer) == 1.0


def test_m_robust_nan_when_no_low_q() -> None:
    transfer = {"TR_0"}
    rs = [_step("TR_0", True, "transfer", q_t=0.95)]
    v = m_robust(rs, transfer, q_low=0.3)
    assert math.isnan(v)


def test_m_hint_nonpositive() -> None:
    rs = [_step("T_0", correct=False, action_type="hint", hint_level=3),
          _step("T_0", correct=False, action_type="hint", hint_level=5)]
    v = m_hint(rs, hint_level_max=6, j_star=4)
    assert v <= 0


def test_m_selection_robust_is_alias_of_m_robust() -> None:
    """Paper v0.1.6 cites ``m_robust``; v0.2.0 renames to
    ``m_selection_robust``. The alias must compute the same value."""
    transfer = {"TR_0"}
    rs = [_step("TR_0", True, "transfer", q_t=0.30)]
    assert m_robust(rs, transfer, q_low=0.50) == m_selection_robust(
        rs, transfer, q_low=0.50
    )


def test_m_inference_robust_perfect_belief_zero() -> None:
    """Identical b_T and p_true_T → 0 error on a contaminated student."""
    p = np.array([0.2, 0.5, 0.9])
    assert m_inference_robust(p, p, "guesser") == 0.0


def test_m_inference_robust_anti_correlated_high() -> None:
    """Anti-correlated belief (b = 1 - p) gives error ≈ |1 - 2p|."""
    p_true = np.array([0.0, 0.5, 1.0])
    p_hat = 1.0 - p_true
    v = m_inference_robust(p_hat, p_true, "copier")
    # |1-0| + |0-0.5| + |-1+1|? Actually |b-p| = [1, 0, 1] -> mean 2/3
    assert abs(v - 2.0 / 3.0) < 1e-9


def test_m_inference_robust_nan_for_honest() -> None:
    """The metric is only defined for contaminated behaviours."""
    p = np.array([0.5, 0.5])
    assert math.isnan(m_inference_robust(p, p, "honest"))
    assert math.isnan(m_inference_robust(p, p, "help_seeker"))
