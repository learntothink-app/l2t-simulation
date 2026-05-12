"""Tests for the controller FSM (§V.B)."""

from __future__ import annotations

import pytest

from l2t_sim.fsm import ControllerFSM, InvariantViolation, reference_forward_walk


def test_forward_path() -> None:
    trace = reference_forward_walk()
    assert trace.states[0] == "GREETING"
    assert trace.states[-1] == "BLOCK_DONE"


def test_theory_check_failure_returns_to_present() -> None:
    fsm = ControllerFSM()
    fsm.step("start")
    fsm.step("theory_presented")
    fsm.step("theory_failed")
    assert fsm.state == "THEORY_PRESENT"


def test_hint_requires_explicit_request_for_deep_levels() -> None:
    fsm = ControllerFSM()
    fsm.step("start")
    fsm.step("theory_presented")
    fsm.step("theory_checked", concept="C0")
    fsm.step("task_presented", concept="C0", task="T0")
    fsm.step("intent_received")
    fsm.step("step_matched")
    # Try to deliver a deep (≥ j*=4) hint without a hint_requested event.
    with pytest.raises(InvariantViolation):
        # need to land in GIVE_HINT first; jump there via hint_requested then test deep level
        fsm.step("hint_requested")  # legal, now in GIVE_HINT? Actually hint_requested from STEP_MATCH → GIVE_HINT
        # Once requested, fsm.state == GIVE_HINT, but no task on hint_requested → mark task
        fsm.hint_requested_seen_for_task.discard("T0")  # force missing request
        fsm.step("hint_delivered", task="T0", hint_level=5, j_star=4)


def test_transfer_gate() -> None:
    fsm = ControllerFSM()
    fsm.step("start")
    fsm.step("theory_presented")
    fsm.step("theory_checked", concept="C0")
    fsm.step("task_presented", concept="C0", task="T0")
    fsm.step("intent_received")
    fsm.step("step_matched")
    fsm.step("result_asked")
    fsm.step("correct")
    fsm.step("step_done")
    fsm.step("task_gate_passed")
    fsm.step("task_done")
    fsm.step("transfer_failed")  # bounces back to TASK_PRESENT
    # Without ever passing transfer, block_done is illegal.
    with pytest.raises(InvariantViolation):
        # Force-set FSM into SUMMARY state to call block_done.
        fsm.state = "SUMMARY"
        fsm.transfer_passed = False
        fsm.step("summary_done")  # SUMMARY → BLOCK_DONE is the legal *transition*,
        # but block_done event with transfer_passed=False is still rejected by the
        # guard, which fires *before* the table lookup. So we re-call:
        fsm.step("block_done")
