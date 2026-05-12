"""Tests for the operational metrics (Eqs. 30–34)."""

from __future__ import annotations

import math

import numpy as np

from l2t_sim.core_types import Action, Observation, StepRecord
from l2t_sim.metrics import m_hint, m_mastery, m_robust, m_transfer


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
