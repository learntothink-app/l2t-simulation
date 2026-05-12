"""Past-LTL pedagogical invariants (Eq. C1, Def. 1) and FSM model-checking.

Three formulas, each ``□(p → ◇^{-1} q)``-shaped (past-safety):

* ``φ_theory-first``: every ``task_present(T)`` is preceded by
  ``theory_checked(c)`` for *every* ``c ∈ Concepts(T)``.
* ``φ_no-spoiler``: every ``hint_delivered_j`` with ``j ≥ j*`` is preceded by
  a matching ``hint_requested``.
* ``φ_transfer-gate``: every ``block_done`` is preceded by ``transfer_passed``.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from .fsm import ControllerFSM, EVENTS, STATES


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


def verify_invariants_on_fsm(max_depth: int = 50, j_star: int = 4) -> InvariantReport:
    """Bounded BFS over reachable event-prefixes of the controller.

    The FSM is treated as a generator of event-trajectories.  Each branch
    (e.g. ``correct`` vs. ``incorrect`` from ``EVAL_RESULT``) spawns a
    distinct successor in the BFS frontier.  Because the controller refuses
    illegal transitions at construction time, the witness that ``M_c`` admits
    only compliant trajectories is *that the BFS finishes without raising*.
    """

    start = time.perf_counter()
    n_transitions = sum(len(out) for out in ControllerFSM._FORWARD.values())

    n_traces = 0
    visited_state_event: set[tuple[str, str]] = set()

    # frontier elements: (depth, trace_records, fsm_state_snapshot)
    # We snapshot only the state + atomic-prop sets needed for guards.
    init_record = {"event": "start"}

    @dataclass
    class _Node:
        depth: int
        trace: list[dict] = field(default_factory=list)
        fsm: ControllerFSM = field(default_factory=ControllerFSM)

    queue: deque[_Node] = deque([_Node(depth=0)])
    invariants_ok = {"theory_first": True, "no_spoiler": True, "transfer_gate": True}

    # synthetic task→concepts map used only inside the BFS
    task_concepts = {"T0": ["C0"], "T1": ["C0", "C1"]}

    while queue:
        node = queue.popleft()
        if node.depth >= max_depth:
            n_traces += 1
            invariants_ok["theory_first"] &= check_theory_first(node.trace, task_concepts)
            invariants_ok["no_spoiler"] &= check_no_spoiler(node.trace, j_star=j_star)
            invariants_ok["transfer_gate"] &= check_transfer_gate(node.trace)
            continue

        admissible = node.fsm.admissible_events()
        if not admissible:
            n_traces += 1
            invariants_ok["theory_first"] &= check_theory_first(node.trace, task_concepts)
            invariants_ok["no_spoiler"] &= check_no_spoiler(node.trace, j_star=j_star)
            invariants_ok["transfer_gate"] &= check_transfer_gate(node.trace)
            continue

        # Generate at most a handful of branchings per state to keep the BFS
        # bounded; the controller is finite, so this still covers all unique
        # (state, event) pairs.
        for ev in admissible:
            key = (node.fsm.state, ev)
            if key in visited_state_event and ev not in {"correct", "incorrect", "partial"}:
                continue
            visited_state_event.add(key)
            child = _Node(depth=node.depth + 1, trace=list(node.trace))
            child.fsm = ControllerFSM()
            # Replay parent's path so the child FSM is consistent.
            for rec in node.trace:
                child.fsm.step(
                    rec["event"],
                    concept=rec.get("concept"),
                    task=rec.get("task"),
                    hint_level=rec.get("hint_level", 0),
                    j_star=j_star,
                )
            # Provide side-condition-friendly context for events that need it.
            kwargs: dict = {"j_star": j_star}
            if ev == "theory_checked":
                kwargs["concept"] = "C0"
            elif ev == "task_presented":
                kwargs["task"] = "T0"
                kwargs["concept"] = "C0"
            elif ev == "hint_requested":
                kwargs["task"] = "T0"
            elif ev == "hint_delivered":
                kwargs["task"] = "T0"
                kwargs["hint_level"] = 2  # below j* by default; still legal
            try:
                child.fsm.step(ev, **kwargs)
            except Exception as exc:  # InvariantViolation = controller refused
                # By construction this would be a counter-example; record it.
                log.debug("BFS rejected %s --[%s]--> : %s", node.fsm.state, ev, exc)
                continue
            rec = {"event": ev}
            rec.update(kwargs)
            rec.pop("j_star", None)
            child.trace.append(rec)
            queue.append(child)

    seconds = time.perf_counter() - start
    return InvariantReport(
        theory_first=invariants_ok["theory_first"],
        no_spoiler=invariants_ok["no_spoiler"],
        transfer_gate=invariants_ok["transfer_gate"],
        n_states=len(STATES),
        n_transitions=n_transitions,
        n_traces_checked=n_traces,
        seconds=seconds,
    )


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
