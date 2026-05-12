"""Tests for the Bayesian student model (Eq. 4)."""

from __future__ import annotations

import numpy as np

from l2t_sim.core_types import Action, Observation
from l2t_sim.methodology import generate_synthetic_methodology
from l2t_sim.student_model import StudentModel


def _mk():
    md = generate_synthetic_methodology(seed=1)
    return md, StudentModel(md.hypergraph, md, use_reliability=True)


def _obs(correct: bool, **kw) -> Observation:
    return Observation(
        correct=correct,
        answer_format="open_text",
        time_seconds=60.0,
        hint_requested=False,
        n_hints_used=0,
        hint_level_reached=0,
        self_explanation=True,
        copying_pattern=False,
        error_type=kw.get("error_type"),
    )


def test_bayes_update_stays_in_interior() -> None:
    md, model = _mk()
    belief = model.initial_belief()
    tid = next(iter(md.tasks))
    a = Action("task", tid)
    for _ in range(100):
        o = _obs(correct=True)
        belief = model.bayes_update(belief, a, o, q_t=1.0)
        assert np.all(belief.p_hat > 0) and np.all(belief.p_hat < 1)
        assert not np.isnan(belief.p_hat).any()


def test_q_zero_means_no_update() -> None:
    md, model = _mk()
    belief = model.initial_belief()
    tid = next(iter(md.tasks))
    a = Action("task", tid)
    new = model.bayes_update(belief, a, _obs(True), q_t=0.0)
    assert np.allclose(belief.p_hat, new.p_hat)


def test_q_one_matches_no_reliability() -> None:
    md = generate_synthetic_methodology(seed=1)
    m_q = StudentModel(md.hypergraph, md, use_reliability=True)
    m_nq = StudentModel(md.hypergraph, md, use_reliability=False)
    tid = next(iter(md.tasks))
    a = Action("task", tid)
    b1 = m_q.initial_belief()
    b2 = m_nq.initial_belief()
    new1 = m_q.bayes_update(b1, a, _obs(True), q_t=1.0)
    new2 = m_nq.bayes_update(b2, a, _obs(True), q_t=1.0)
    assert np.allclose(new1.p_hat, new2.p_hat)


def test_correct_answer_increases_required_skills() -> None:
    md, model = _mk()
    belief = model.initial_belief()
    tid = next(iter(md.tasks))
    task = md.tasks[tid]
    a = Action("task", tid)
    new = model.bayes_update(belief, a, _obs(True), q_t=1.0)
    for s in task.required_skills:
        j = model.skill_index[s]
        assert new.p_hat[j] > belief.p_hat[j] - 1e-9
