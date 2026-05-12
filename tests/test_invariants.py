"""Tests for the past-LTL invariant checkers and FSM model-checking."""

from __future__ import annotations

from l2t_sim.invariants import (
    check_no_spoiler,
    check_theory_first,
    check_transfer_gate,
    verify_invariants_on_fsm,
)


def test_theory_first_positive() -> None:
    trace = [
        {"event": "theory_checked", "concept": "C0"},
        {"event": "task_presented", "task": "T0"},
    ]
    assert check_theory_first(trace, {"T0": ["C0"]})


def test_theory_first_negative() -> None:
    trace = [{"event": "task_presented", "task": "T0"}]
    assert not check_theory_first(trace, {"T0": ["C0"]})


def test_no_spoiler_below_threshold_ok() -> None:
    trace = [{"event": "hint_delivered", "task": "T0", "hint_level": 2}]
    assert check_no_spoiler(trace, j_star=4)


def test_no_spoiler_above_threshold_requires_request() -> None:
    bad = [{"event": "hint_delivered", "task": "T0", "hint_level": 5}]
    good = [
        {"event": "hint_requested", "task": "T0"},
        {"event": "hint_delivered", "task": "T0", "hint_level": 5},
    ]
    assert not check_no_spoiler(bad, j_star=4)
    assert check_no_spoiler(good, j_star=4)


def test_transfer_gate() -> None:
    bad = [{"event": "block_done"}]
    good = [{"event": "transfer_passed"}, {"event": "block_done"}]
    assert not check_transfer_gate(bad)
    assert check_transfer_gate(good)


def test_fsm_model_checking_passes() -> None:
    rep = verify_invariants_on_fsm(max_depth=8)
    assert rep.theory_first
    assert rep.no_spoiler
    assert rep.transfer_gate
    assert rep.n_states == 16
    assert rep.n_transitions > 0
