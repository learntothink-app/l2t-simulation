"""Tests for the synthetic environment (behaviour types, dynamics)."""

from __future__ import annotations

import numpy as np

from l2t_sim.core_types import Action
from l2t_sim.environment import SyntheticStudent
from l2t_sim.methodology import generate_synthetic_methodology


def _student(behaviour: str, seed: int = 0) -> SyntheticStudent:
    md = generate_synthetic_methodology(seed=1)
    rng = np.random.default_rng(seed)
    return SyntheticStudent.sample(md, behaviour, rng)


def test_p_true_stays_in_interior() -> None:
    s = _student("honest", seed=0)
    md = s.methodology
    actions = [Action("task", t) for t in list(md.tasks.keys())[:8]]
    for _ in range(50):
        for a in actions:
            o = s.observe(a)
            s.update_p_true(a, o)
    assert np.all(s.p_true > 0.0)
    assert np.all(s.p_true < 1.0)
    assert not np.any(np.isnan(s.p_true))


def test_guesser_independent_of_p_true() -> None:
    rng = np.random.default_rng(123)
    md = generate_synthetic_methodology(seed=1)
    # Pick a MC task.
    mc_id = next(t for t, info in md.tasks.items() if info.answer_format == "multiple_choice")
    info = md.tasks[mc_id]

    high = SyntheticStudent.sample(md, "guesser", np.random.default_rng(0))
    high.p_true[:] = 0.99
    low = SyntheticStudent.sample(md, "guesser", np.random.default_rng(1))
    low.p_true[:] = 0.01

    n_high = sum(1 for _ in range(2000) if high.observe(Action("task", mc_id)).correct)
    n_low = sum(1 for _ in range(2000) if low.observe(Action("task", mc_id)).correct)
    expected = 2000.0 * (1.0 / info.n_options)
    # Both should be within a wide tolerance of 1/n_options independent of mastery.
    assert abs(n_high - expected) < 200
    assert abs(n_low - expected) < 200


def test_copier_high_accuracy() -> None:
    s = _student("copier", seed=2)
    md = s.methodology
    tid = next(iter(md.tasks))
    a = Action("task", tid)
    correct = sum(1 for _ in range(500) if s.observe(a).correct)
    assert correct / 500.0 > 0.7


def test_help_seeker_improves_with_hints() -> None:
    md = generate_synthetic_methodology(seed=1)
    tid = next(iter(md.tasks))
    a = Action("task", tid)

    no_hint = SyntheticStudent.sample(md, "help_seeker", np.random.default_rng(0))
    with_hint = SyntheticStudent.sample(md, "help_seeker", np.random.default_rng(1))
    # mimic 3 hints already received
    with_hint._hints_used[tid] = 3

    n_no = sum(1 for _ in range(800) if no_hint.observe(a).correct)
    n_h = sum(1 for _ in range(800) if with_hint.observe(a).correct)
    assert n_h > n_no  # not a hugely tight test, but strictly monotone


def test_behaviour_proportions() -> None:
    from l2t_sim.simulation import sample_behaviour

    rng = np.random.default_rng(0)
    props = {"honest": 0.60, "guesser": 0.15, "copier": 0.10, "help_seeker": 0.15}
    out = [sample_behaviour(rng, props) for _ in range(2000)]
    for name, target in props.items():
        share = out.count(name) / 2000.0
        assert abs(share - target) < 0.03, (name, share, target)
