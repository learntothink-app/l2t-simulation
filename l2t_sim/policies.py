"""Action-selection policies B1–B4 (§VII.B, §V.D).

* B1 — Random curriculum, uniform over ``A_safe``.
* B2 — Classical BKT-adaptive (no hypergraph; per-skill scalar mastery).
* B3 — Full L2T architecture *without* reliability (``q_t ≡ 1``).
* B4 — Full L2T with the reliability-weighted update (``q_t`` from Eq. 23).

All L2T policies (B3, B4) respect the past-LTL pedagogical invariants
``φ_theory-first``, ``φ_no-spoiler``, ``φ_transfer-gate`` *by construction*.
The Random policy may violate them — that's part of the experiment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .core_types import Action, Observation
from .methodology import MethodologyData, Task
from .student_model import BeliefState


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context passed to every select_action.


@dataclass
class Context:
    t: int
    methodology: MethodologyData
    # Concepts whose theory has been checked (past-LTL trace).
    theory_checked: set[str] = field(default_factory=set)
    # Last observation, used for hint_requested gating.
    last_observation: Observation | None = None
    # Target_id of the most recent evaluative action — used to route hints to
    # the right item (NOT just "last task in history", which can be stale).
    current_item: str | None = None
    # Tasks already completed (correct) and transfer tasks passed.
    completed_tasks: set[str] = field(default_factory=set)
    transfer_passed: bool = False
    # Hints already used per-task (for hint_level ladder).
    hints_used_per_task: dict[str, int] = field(default_factory=dict)
    # Steps without any model movement — used to break loops.
    stale_streak: int = 0
    # Recent attempts per task — used to avoid spamming the same item.
    recent_attempts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class


class Policy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def select_action(self, belief: BeliefState, context: Context) -> Action: ...

    def reset(self) -> None:
        """Hook for stateful policies (BKT)."""


# ---------------------------------------------------------------------------
# B1 — Random


class RandomPolicy(Policy):
    """Uniform over ``A_safe`` — no respect for belief or invariants."""

    name = "B1_random"

    def __init__(self, hypergraph, methodology: MethodologyData, rng: np.random.Generator) -> None:
        self.methodology = methodology
        self.rng = rng

    def select_action(self, belief: BeliefState, context: Context) -> Action:
        md = self.methodology
        # build a pool of (action_type, target_id) pairs
        choices: list[tuple[str, str]] = []
        choices.extend(("task", t) for t in md.tasks)
        choices.extend(("drill", d) for d in md.drills)
        choices.extend(("probe", p) for p in md.probes)
        choices.extend(("transfer", t) for t in md.transfer_tasks)
        choices.extend(("microtheory", mt) for mt in md.microtheories)
        # random policy can also issue a proactive hint occasionally
        for tid in list(md.tasks.keys())[:5]:
            choices.append(("hint", tid))
        if not choices:
            return Action(type="task", target_id="")
        idx = int(self.rng.integers(0, len(choices)))
        atype, tid = choices[idx]
        hint_level = int(self.rng.integers(1, 7)) if atype == "hint" else 0
        return Action(type=atype, target_id=tid, hint_level=hint_level)


# ---------------------------------------------------------------------------
# B2 — BKT


class BKTPolicy(Policy):
    """Classical Bayesian Knowledge Tracing baseline.

    Maintains per-skill ``p_known`` with the standard 4-parameter update.
    Picks the task whose required skill has the smallest ``p_known``.
    """

    name = "B2_bkt"

    def __init__(
        self,
        hypergraph,
        methodology: MethodologyData,
        rng: np.random.Generator,
        p_init: float = 0.30,
        p_learn: float = 0.10,
        p_guess: float = 0.20,
        p_slip: float = 0.10,
    ) -> None:
        self.methodology = methodology
        self.rng = rng
        self.skill_ids = list(methodology.hypergraph.skill_ids)
        self.p_known: dict[str, float] = {s: p_init for s in self.skill_ids}
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip
        self._last_action: Action | None = None

    def reset(self) -> None:
        self.p_known = {s: 0.30 for s in self.skill_ids}
        self._last_action = None

    def observe_outcome(self, action: Action, observation: Observation) -> None:
        """Update BKT scalars using the standard recursion."""

        if action.type not in ("task", "drill", "probe", "transfer"):
            return
        task = self.methodology.all_tasks.get(action.target_id) or self.methodology.drills.get(
            action.target_id
        ) or self.methodology.probes.get(action.target_id)
        if task is None:
            return
        for sk in task.required_skills:
            if sk not in self.p_known:
                continue
            p_k = self.p_known[sk]
            if observation.correct:
                num = p_k * (1 - self.p_slip)
                den = num + (1 - p_k) * self.p_guess
            else:
                num = p_k * self.p_slip
                den = num + (1 - p_k) * (1 - self.p_guess)
            posterior = num / max(den, 1e-9)
            # Forward transition: chance of *learning* this step.
            self.p_known[sk] = posterior + (1 - posterior) * self.p_learn
            self.p_known[sk] = float(np.clip(self.p_known[sk], 1e-6, 1 - 1e-6))

    def select_action(self, belief: BeliefState, context: Context) -> Action:
        # Pick the lowest-known skill and a task that trains it.
        if not self.p_known:
            return Action(type="task", target_id="")
        lowest = min(self.p_known.items(), key=lambda kv: kv[1])[0]
        candidates = self.methodology.task_ids_for_skill(lowest)
        if not candidates:
            candidates = list(self.methodology.tasks.keys())
        if not candidates:
            return Action(type="probe", target_id=next(iter(self.methodology.probes), ""))
        tid = candidates[int(self.rng.integers(0, len(candidates)))]
        return Action(type="task", target_id=tid)


# ---------------------------------------------------------------------------
# B3 / B4 — Full L2T (with vs without reliability weighting)


class L2TPolicy(Policy):
    """Rule-based L2T policy (Stage 1 of §V.D).

    The selection rules mirror §VII.D and §V.B. The class is parameterised
    by ``use_reliability``; that flag is *not* read by the policy itself —
    it affects the student model — but it is exposed so that the runner
    pipeline can dispatch on it cleanly.
    """

    def __init__(
        self,
        hypergraph,
        methodology: MethodologyData,
        rng: np.random.Generator,
        use_reliability: bool = True,
        low_mastery_threshold: float = 0.30,
        prereq_threshold: float = 0.70,
        retention_period: int = 7,
        transfer_threshold: float = 0.70,
    ) -> None:
        self.methodology = methodology
        self.rng = rng
        self.use_reliability = use_reliability
        self.low_mastery_threshold = low_mastery_threshold
        self.prereq_threshold = prereq_threshold
        self.retention_period = retention_period
        self.transfer_threshold = transfer_threshold

        self.skill_ids = list(methodology.hypergraph.skill_ids)
        self.skill_index = {s: i for i, s in enumerate(self.skill_ids)}
        self.prereq_dag = methodology.prereq_dag  # list[(from, to)]
        self.prereqs_of: dict[str, list[str]] = {s: [] for s in self.skill_ids}
        for a, b in self.prereq_dag:
            if b in self.prereqs_of:
                self.prereqs_of[b].append(a)

    @property
    def name(self) -> str:
        return "B4_full_l2t" if self.use_reliability else "B3_l2t_no_q"

    # -- helpers ------------------------------------------------------------

    def _prereqs_mastered(self, skill_id: str, p_hat: np.ndarray) -> bool:
        for pre in self.prereqs_of.get(skill_id, ()):
            j = self.skill_index.get(pre)
            if j is None:
                continue
            if p_hat[j] < self.prereq_threshold:
                return False
        return True

    def _concept_theory_pending(self, task: Task, context: Context) -> str | None:
        """Return a concept that must be theory_checked before this task, or None."""
        for c in task.required_concepts:
            if c not in context.theory_checked:
                return c
        return None

    def _find_microtheory(self, concept_id: str) -> str | None:
        for mt in self.methodology.microtheories.values():
            if concept_id in mt.concepts:
                return mt.id
        return None

    def _can_hint(self, context: Context) -> bool:
        return context.last_observation is not None and context.last_observation.hint_requested

    # -- core action selection ---------------------------------------------

    def select_action(self, belief: BeliefState, context: Context) -> Action:
        md = self.methodology
        p_hat = belief.p_hat

        # 1) honour an outstanding hint request (φ_no-spoiler is enforced —
        #    we only emit a hint *after* a hint_requested event for the same
        #    item, tracked via ``context.current_item``).
        if self._can_hint(context) and context.current_item is not None:
            target = context.current_item
            cur = context.hints_used_per_task.get(target, 0)
            next_level = min(cur + 1, md.hint_ladder_max)
            return Action(type="hint", target_id=target, hint_level=next_level)

        # 2) periodic retention check
        if context.t > 0 and context.t % self.retention_period == 0:
            mastered = [i for i, p in enumerate(p_hat) if p > self.transfer_threshold]
            if mastered:
                idx = int(self.rng.choice(mastered))
                sid = self.skill_ids[idx]
                # use a drill/transfer linked to this skill for retention probe
                cands = md.transfer_ids_for_skill(sid) or md.drill_ids_for_skill(sid)
                if cands:
                    return Action(type="retention", target_id=cands[0])

        # 3) skills that look low-mastery → probe for diagnosis.
        low_skills = [
            self.skill_ids[i]
            for i, p in enumerate(p_hat)
            if p < self.low_mastery_threshold
        ]
        if low_skills:
            # Among low_skills, pick the one with the least recent attempts.
            low_skills.sort(key=lambda s: context.recent_attempts.get(s, 0))
            for sid in low_skills:
                # If no probe at all yet for this skill, run one (diagnostic).
                probe_cands = md.probe_ids_for_skill(sid)
                if probe_cands and self.rng.random() < 0.25:
                    pid = probe_cands[int(self.rng.integers(0, len(probe_cands)))]
                    return Action(type="probe", target_id=pid)

        # 4) prerequisite-respecting "drill the weakest mastered-prereqs skill".
        eligible: list[tuple[float, str]] = []
        for i, sid in enumerate(self.skill_ids):
            if self._prereqs_mastered(sid, p_hat):
                eligible.append((float(p_hat[i]), sid))
        eligible.sort(key=lambda kv: kv[0])

        # 5) transfer phase: if eligible top is well-mastered, try a transfer task.
        if eligible and eligible[-1][0] > self.transfer_threshold:
            best_skill = eligible[-1][1]
            t_cands = md.transfer_ids_for_skill(best_skill)
            if t_cands:
                tid = t_cands[int(self.rng.integers(0, len(t_cands)))]
                return Action(type="transfer", target_id=tid)

        # 6) work on the weakest eligible skill: theory → task → drill.
        if eligible:
            target_skill = eligible[0][1]
            task_cands = md.task_ids_for_skill(target_skill)
            if task_cands:
                tid = task_cands[int(self.rng.integers(0, len(task_cands)))]
                task = md.tasks[tid]
                pending_c = self._concept_theory_pending(task, context)
                if pending_c is not None:
                    mt = self._find_microtheory(pending_c)
                    if mt is not None:
                        return Action(type="microtheory", target_id=mt)
                return Action(type="task", target_id=tid)
            drill_cands = md.drill_ids_for_skill(target_skill)
            if drill_cands:
                did = drill_cands[int(self.rng.integers(0, len(drill_cands)))]
                return Action(type="drill", target_id=did)

        # 7) fallback: any task.
        if md.tasks:
            tid = next(iter(md.tasks))
            task = md.tasks[tid]
            pending_c = self._concept_theory_pending(task, context)
            if pending_c is not None:
                mt = self._find_microtheory(pending_c)
                if mt is not None:
                    return Action(type="microtheory", target_id=mt)
            return Action(type="task", target_id=tid)
        if md.probes:
            return Action(type="probe", target_id=next(iter(md.probes)))
        return Action(type="task", target_id="")
