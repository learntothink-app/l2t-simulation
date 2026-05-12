"""Tests for the reliability heuristic q_t (Eq. 23)."""

from __future__ import annotations

from l2t_sim.core_types import Action, Observation
from l2t_sim.reliability import Q_MIN, compute_reliability


def _obs(**kw) -> Observation:
    base = dict(
        correct=True,
        answer_format="open_text",
        time_seconds=60.0,
        hint_requested=False,
        n_hints_used=0,
        hint_level_reached=0,
        self_explanation=True,
        copying_pattern=False,
        error_type=None,
    )
    base.update(kw)
    return Observation(**base)


def test_q_in_unit_interval() -> None:
    a = Action("task", "T_0")
    for kw in (
        {},
        {"self_explanation": False},
        {"copying_pattern": True},
        {"time_seconds": 1.0},
        {"self_explanation": False, "copying_pattern": True, "time_seconds": 1.0},
    ):
        q = compute_reliability(a, _obs(**kw), action_baseline_time=60.0)
        assert Q_MIN <= q <= 1.0, kw


def test_copier_low_q() -> None:
    a = Action("task", "T_0")
    q = compute_reliability(
        a,
        _obs(copying_pattern=True, self_explanation=False),
        action_baseline_time=60.0,
    )
    assert q < 0.5


def test_honest_high_q() -> None:
    a = Action("task", "T_0")
    q = compute_reliability(
        a,
        _obs(self_explanation=True, copying_pattern=False, time_seconds=60.0),
        action_baseline_time=60.0,
    )
    assert q > 0.8
