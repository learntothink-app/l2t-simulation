"""Past-LTL pedagogical invariants (Eq. C1, Def. 1) and FSM model-checking.

Three formulas, each ``□(p → ◇^{-1} q)``-shaped (past-safety):

* ``φ_theory-first``: every ``task_present(T)`` is preceded by
  ``theory_checked(c)`` for *every* ``c ∈ Concepts(T)``.
* ``φ_no-spoiler``: every ``hint_delivered_j`` with ``j ≥ j*`` is preceded by
  a matching ``hint_requested``.
* ``φ_transfer-gate``: every ``block_done`` is preceded by ``transfer_passed``.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .fsm import ControllerFSM, EVENTS, InvariantViolation, STATES


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event-trace representation.
#
# A "trajectory" for the invariant-checker is a list of event records:
#   {"event": str, "concept": str|None, "task": str|None, "hint_level": int}


def _events_with_task_concepts(record: dict) -> tuple[str, str | None, str | None, int]:
    return (
        record["event"],
        record.get("concept"),
        record.get("task"),
        int(record.get("hint_level", 0)),
    )


def check_theory_first(trajectory: Iterable[dict], task_concepts: dict[str, list[str]]) -> bool:
    """φ_theory-first.

    For every ``task_presented`` event in ``trajectory`` referring to task T,
    *all* concepts in ``task_concepts[T]`` must have been ``theory_checked``
    earlier in the trajectory.
    """
    checked: set[str] = set()
    for rec in trajectory:
        ev, concept, task, _ = _events_with_task_concepts(rec)
        if ev == "theory_checked" and concept is not None:
            checked.add(concept)
        elif ev == "task_presented" and task is not None:
            needed = task_concepts.get(task, [])
            for c in needed:
                if c not in checked:
                    return False
    return True


def check_no_spoiler(trajectory: Iterable[dict], j_star: int = 4) -> bool:
    """φ_no-spoiler.

    For every ``hint_delivered`` at level ``j ≥ j*``, there must be a prior
    ``hint_requested`` for the same task.
    """
    requested: set[str] = set()
    for rec in trajectory:
        ev, _, task, level = _events_with_task_concepts(rec)
        if ev == "hint_requested" and task is not None:
            requested.add(task)
        elif ev == "hint_delivered" and level >= j_star:
            if task is None or task not in requested:
                return False
    return True


def check_transfer_gate(trajectory: Iterable[dict]) -> bool:
    """φ_transfer-gate: ``block_done`` requires prior ``transfer_passed``."""
    passed = False
    for rec in trajectory:
        ev, _, _, _ = _events_with_task_concepts(rec)
        if ev == "transfer_passed":
            passed = True
        elif ev == "block_done":
            if not passed:
                return False
    return True


# ---------------------------------------------------------------------------
# BFS-based model-checking over the FSM.
#
# We exhaustively enumerate event-prefixes of bounded length and verify that
# (a) the FSM never raises InvariantViolation on an admitted prefix, and
# (b) each generated trajectory satisfies the three invariants.
#
# Because the FSM disallows illegal transitions in :meth:`ControllerFSM.step`,
# any path the BFS *reaches* is necessarily compliant — that is exactly the
# point. The BFS confirms that the *guarded* automaton ``M_c`` admits only
# φ-compliant traces.


@dataclass
class InvariantReport:
    theory_first: bool
    no_spoiler: bool
    transfer_gate: bool
    n_states: int
    n_transitions: int
    n_traces_checked: int
    seconds: float

    @property
    def all_held(self) -> bool:
        return self.theory_first and self.no_spoiler and self.transfer_gate


def _kwargs_for_event(ev: str, j_star: int = 4) -> dict:
    """Default event payloads used during BFS enumeration."""

    if ev == "theory_checked":
        return {"concept": "C0", "j_star": j_star}
    if ev == "task_presented":
        return {"task": "T0", "concept": "C0", "j_star": j_star}
    if ev == "hint_requested":
        return {"task": "T0", "j_star": j_star}
    if ev == "hint_delivered":
        return {"task": "T0", "hint_level": 2, "j_star": j_star}
    return {"j_star": j_star}


def verify_invariants_on_fsm(
    max_depth: int = 30,
    j_star: int = 4,
    max_traces: int = 5000,
    fsm_factory: Callable[[], ControllerFSM] = ControllerFSM,
    task_concepts: dict[str, list[str]] | None = None,
) -> InvariantReport:
    """Bounded DFS over reachable event-trajectories of the controller.

    Each frontier element is a ``(fsm_clone, trace)`` pair. Branching at
    ``EVAL_RESULT`` (correct/partial/incorrect), ``THEORY_CHECK``
    (passed/failed), etc. spawns distinct successors. The controller refuses
    illegal transitions inside :meth:`ControllerFSM.step`, so any trace that
    survives to completion is by-construction compliant. We then verify
    each completed trace explicitly against the three invariants.

    ``fsm_factory`` lets callers inject a *broken* FSM subclass to assert
    that the model-checker would catch a violated invariant — see the
    negative test in :file:`tests/test_invariants.py`.
    """

    start = time.perf_counter()
    n_transitions = sum(len(out) for out in ControllerFSM._FORWARD.values())
    if task_concepts is None:
        task_concepts = {"T0": ["C0"], "T1": ["C0", "C1"]}

    frontier: list[tuple[ControllerFSM, list[dict]]] = [(fsm_factory(), [])]
    completed: list[list[dict]] = []

    while frontier and len(completed) < max_traces:
        fsm, trace = frontier.pop()
        if len(trace) >= max_depth or fsm.state == "BLOCK_DONE":
            completed.append(trace)
            continue
        admissible = fsm.admissible_events()
        if not admissible:
            completed.append(trace)
            continue
        for ev in admissible:
            kwargs = _kwargs_for_event(ev, j_star)
            child = copy.deepcopy(fsm)
            try:
                child.step(ev, **kwargs)
            except InvariantViolation:
                continue
            rec: dict = {"event": ev}
            for k, v in kwargs.items():
                if k != "j_star":
                    rec[k] = v
            frontier.append((child, trace + [rec]))

    invariants_ok = {"theory_first": True, "no_spoiler": True, "transfer_gate": True}
    for trace in completed:
        invariants_ok["theory_first"] &= check_theory_first(trace, task_concepts)
        invariants_ok["no_spoiler"] &= check_no_spoiler(trace, j_star=j_star)
        invariants_ok["transfer_gate"] &= check_transfer_gate(trace)

    seconds = time.perf_counter() - start
    return InvariantReport(
        theory_first=invariants_ok["theory_first"],
        no_spoiler=invariants_ok["no_spoiler"],
        transfer_gate=invariants_ok["transfer_gate"],
        n_states=len(STATES),
        n_transitions=n_transitions,
        n_traces_checked=len(completed),
        seconds=seconds,
    )


class UnguardedControllerFSM(ControllerFSM):
    """A controller with the ``theory_first`` invariant deliberately removed.

    Two changes from :class:`ControllerFSM`:
    (a) the transition table allows a ``task_presented`` shortcut directly
        from ``THEORY_PRESENT`` (skipping ``THEORY_CHECK``);
    (b) the corresponding side-condition guard in :meth:`step` is dropped.

    Used exclusively for the negative test of
    :func:`verify_invariants_on_fsm` — the model-checker must produce
    ``theory_first=False`` when this FSM is injected, demonstrating that
    the verification machinery actually catches reachable violations
    rather than being vacuously true.
    """

    _FORWARD: dict[str, dict[str, str]] = {
        **ControllerFSM._FORWARD,
        "THEORY_PRESENT": {
            "theory_presented": "THEORY_CHECK",
            "task_presented": "WAIT_INTENT",  # broken shortcut — skips theory
        },
    }

    def step(
        self,
        event: str,
        *,
        concept: str | None = None,
        task: str | None = None,
        hint_level: int = 0,
        j_star: int = 4,
    ) -> str:
        if event not in EVENTS:
            raise InvariantViolation(f"unknown event: {event!r}")
        # Theory-first guard DELIBERATELY OMITTED here.
        # No-spoiler and transfer-gate guards (kept).
        if event == "hint_delivered" and hint_level >= j_star:
            if not task or task not in self.hint_requested_seen_for_task:
                raise InvariantViolation(
                    f"hint_delivered at level {hint_level} ≥ j*={j_star} without hint_requested"
                )
        if event == "block_done" and not self.transfer_passed:
            raise InvariantViolation("block_done requires prior transfer_passed")
        # Bookkeeping (same as parent).
        if event == "theory_checked" and concept is not None:
            self.theory_checked.add(concept)
        if event == "hint_requested" and task is not None:
            self.hint_requested_seen_for_task.add(task)
        if event == "transfer_passed":
            self.transfer_passed = True
        if event == "theory_failed":
            self.theory_attempts += 1
        if event == "theory_checked":
            self.theory_attempts = 0
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


# ---------------------------------------------------------------------------
# Helpers used by the simulation to keep an event log per trajectory.


def trajectory_satisfies_all(
    events: list[dict],
    task_concepts: dict[str, list[str]],
    j_star: int = 4,
) -> dict[str, bool]:
    return {
        "theory_first": check_theory_first(events, task_concepts),
        "no_spoiler": check_no_spoiler(events, j_star=j_star),
        "transfer_gate": check_transfer_gate(events),
    }
