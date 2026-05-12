"""Deterministic pedagogical controller (§V.B).

The controller is a Kripke structure ``M_c = (Q, q_0, δ, L)`` with 16 states.
States are labelled with atomic propositions over which the past-LTL
invariants in :mod:`l2t_sim.invariants` are checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal


STATES: tuple[str, ...] = (
    "GREETING",
    "THEORY_PRESENT",
    "THEORY_CHECK",
    "TASK_PRESENT",
    "WAIT_INTENT",
    "STEP_MATCH",
    "ASK_RESULT",
    "EVAL_RESULT",
    "STEP_DONE",
    "GIVE_HINT",
    "CHECK_TASK_GATE",
    "TASK_DONE",
    "TRANSFER",
    "RETENTION",
    "SUMMARY",
    "BLOCK_DONE",
)

# Events ``Ev`` the transition function reads.
EVENTS: tuple[str, ...] = (
    "start",
    "theory_presented",
    "theory_checked",
    "theory_failed",
    "task_presented",
    "intent_received",
    "step_matched",
    "result_asked",
    "correct",
    "partial",
    "incorrect",
    "hint_requested",
    "hint_delivered",
    "step_done",
    "task_gate_passed",
    "task_done",
    "transfer_passed",
    "transfer_failed",
    "retention_passed",
    "retention_failed",
    "summary_done",
    "block_done",
)


class InvariantViolation(Exception):
    """Raised when an event triggers a transition forbidden by ``δ``."""


@dataclass
class FsmTrace:
    """One linear pass through the FSM, with the labels seen at each step."""

    states: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class ControllerFSM:
    """The 16-state controller of §V.B with side-condition guards."""

    def __init__(self) -> None:
        self.state: str = "GREETING"
        self.trace: FsmTrace = FsmTrace(states=["GREETING"], events=[])

        # past-trace flags used by side conditions
        self.theory_checked: set[str] = set()
        self.hint_requested_seen_for_task: set[str] = set()
        self.transfer_passed: bool = False

        # Counters
        self.theory_attempts: int = 0

    # -- transition tables --------------------------------------------------

    _FORWARD: dict[str, dict[str, str]] = {
        "GREETING": {"start": "THEORY_PRESENT"},
        "THEORY_PRESENT": {"theory_presented": "THEORY_CHECK"},
        "THEORY_CHECK": {
            "theory_checked": "TASK_PRESENT",
            "theory_failed": "THEORY_PRESENT",
        },
        "TASK_PRESENT": {"task_presented": "WAIT_INTENT"},
        "WAIT_INTENT": {
            "intent_received": "STEP_MATCH",
            "hint_requested": "GIVE_HINT",
        },
        "STEP_MATCH": {
            "step_matched": "ASK_RESULT",
            "hint_requested": "GIVE_HINT",
        },
        "ASK_RESULT": {
            "result_asked": "EVAL_RESULT",
            "hint_requested": "GIVE_HINT",
        },
        "EVAL_RESULT": {
            "correct": "STEP_DONE",
            "partial": "STEP_MATCH",
            "incorrect": "STEP_MATCH",
        },
        "STEP_DONE": {"step_done": "CHECK_TASK_GATE"},
        "GIVE_HINT": {"hint_delivered": "STEP_MATCH"},
        "CHECK_TASK_GATE": {"task_gate_passed": "TASK_DONE"},
        "TASK_DONE": {"task_done": "TRANSFER"},
        "TRANSFER": {
            "transfer_passed": "RETENTION",
            "transfer_failed": "TASK_PRESENT",
        },
        "RETENTION": {
            "retention_passed": "SUMMARY",
            "retention_failed": "THEORY_PRESENT",
        },
        "SUMMARY": {"summary_done": "BLOCK_DONE"},
        # BLOCK_DONE is terminal; we model the labelling proposition
        # ``block_done`` as a self-loop so it can be emitted as an event after
        # entering the state. The transfer_passed guard still fires.
        "BLOCK_DONE": {"block_done": "BLOCK_DONE"},
    }

    # -- step ---------------------------------------------------------------

    def step(self, event: str, *, concept: str | None = None, task: str | None = None,
             hint_level: int = 0, j_star: int = 4) -> str:
        """Take a single event; return the new state.

        Side conditions enforced here implement Statement 1 (App. C):
          * ``task_present(T)`` is admissible only if every concept of ``T``
            has been theory_checked.
          * ``hint_delivered_j`` with j ≥ j* is admissible only after
            ``hint_requested``.
          * ``block_done`` is admissible only after ``transfer_passed``.
        """

        if event not in EVENTS:
            raise InvariantViolation(f"unknown event: {event!r}")

        # Side conditions (a), (b), (c) of Statement 1.
        if event == "task_presented":
            if concept and concept not in self.theory_checked:
                raise InvariantViolation(
                    f"task_present requires theory_checked({concept}); aborted"
                )
        if event == "hint_delivered":
            if hint_level >= j_star and (not task or task not in self.hint_requested_seen_for_task):
                raise InvariantViolation(
                    f"hint_delivered at level {hint_level} ≥ j*={j_star} without hint_requested"
                )
        if event == "block_done" and not self.transfer_passed:
            raise InvariantViolation("block_done requires prior transfer_passed")

        # Bookkeeping (atomic propositions)
        if event == "theory_checked" and concept is not None:
            self.theory_checked.add(concept)
        if event == "hint_requested" and task is not None:
            self.hint_requested_seen_for_task.add(task)
        if event == "transfer_passed":
            self.transfer_passed = True
        if event == "theory_failed":
            self.theory_attempts += 1
        if event == "theory_checked":
            self.theory_attempts = 0  # reset counter

        # transition lookup
        table = self._FORWARD.get(self.state, {})
        nxt = table.get(event)
        if nxt is None:
            raise InvariantViolation(
                f"illegal transition: {self.state} --[{event}]-->"
            )
        self.state = nxt
        self.trace.states.append(nxt)
        self.trace.events.append(event)
        return nxt

    # -- enumeration --------------------------------------------------------

    def admissible_events(self) -> list[str]:
        """Events that don't immediately violate a guard from the current state.

        Used by the model-checker (BFS over reachable states / event prefixes).
        """
        return list(self._FORWARD.get(self.state, {}).keys())


# ---------------------------------------------------------------------------
# Linear (deterministic) reference walk through the FSM, useful for tests.


def reference_forward_walk() -> FsmTrace:
    """A minimal canonical trace GREETING → BLOCK_DONE."""
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
    fsm.step("transfer_passed")
    fsm.step("retention_passed")
    fsm.step("summary_done")
    fsm.step("block_done")
    return fsm.trace
