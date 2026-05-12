"""Synthetic student environment.

Implements the four behaviour types from §VII (honest, guesser, copier,
help_seeker) and the logistic-normal mastery dynamics of Eq. (A3). The
expected mastery increment is given by Eq. (7); we sample around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.special import expit, logit

from .core_types import Action, Observation, BehaviourType
from .methodology import MethodologyData, Task


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamics parameters (frozen in config; see §VII.A and §App.A).

ETA0_BY_ACTION: dict[str, float] = {
    "drill": 0.30,
    "task": 0.50,
    "transfer": 0.70,
    "probe": 0.10,
    "microtheory": 0.05,
    "hint": 0.00,
    "retention": 0.00,
}
ETA1_META: float = 0.20  # per meta_helps_learn coupling
ETA_MINUS_DEFAULT: float = 0.10  # error without signature
ETA_MINUS_SIGNATURE: float = 0.30  # error matching an error_signature
SIGMA_K: float = 0.10  # stochasticity of mastery dynamics (Eq. A3)
SIGMA_J: float = 0.10  # stochasticity of meta-skill dynamics

# Retention probe: forgetting rate λ in p_true ← p_true·exp(-λ·Δt).
FORGETTING_LAMBDA_PER_DAY: float = 0.05


# ---------------------------------------------------------------------------


def _logit_safe(x: np.ndarray, lo: float = 1e-6, hi: float = 1 - 1e-6) -> np.ndarray:
    return logit(np.clip(x, lo, hi))


def _expit_safe(x: np.ndarray, lo: float = 1e-6, hi: float = 1 - 1e-6) -> np.ndarray:
    return np.clip(expit(x), lo, hi)


@dataclass
class SyntheticStudent:
    """Synthetic learner with hidden state ``s_t = (p, c, m, e, ψ)``.

    We model the two pedagogically-load-bearing components: ``p_true`` and
    ``m_true``. The behaviour type governs the observation kernel.
    """

    p_true: np.ndarray  # (K,) skill mastery, in (0,1)
    m_true: np.ndarray  # (M,) meta-skill levels, in [0,1]
    behaviour: BehaviourType
    methodology: MethodologyData
    rng: np.random.Generator
    skill_index: dict[str, int] = field(default_factory=dict)
    metaskill_index: dict[str, int] = field(default_factory=dict)
    prereq_perturbation: bool = False

    # internal: last (action_id -> hints_consumed) per task, so that
    # the help_seeker accumulates a P(correct) bonus.
    _hints_used: dict[str, int] = field(default_factory=dict)
    _retention_offset_days: float = 0.0  # how much forgetting has accumulated

    # -- factory ------------------------------------------------------------

    @classmethod
    def sample(
        cls,
        methodology: MethodologyData,
        behaviour: BehaviourType,
        rng: np.random.Generator,
        prereq_perturbation: bool = False,
        alpha: float = 2.0,
        beta: float = 2.0,
    ) -> "SyntheticStudent":
        """Sample initial state.

        ``p_true[k] ~ Beta(α, β)`` with α, β > 1 to guarantee p ∈ (0,1)
        (Lemma 2 needs strict interior; see §VII.A bullet 1 of the paper).
        """
        K = len(methodology.hypergraph.skill_ids)
        M = len(methodology.hypergraph.metaskill_ids)
        p = rng.beta(alpha, beta, size=K)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        m = rng.beta(alpha, beta, size=max(M, 1))
        m = np.clip(m, 1e-6, 1 - 1e-6)
        if M == 0:
            m = np.zeros(0)
        skill_index = {sid: i for i, sid in enumerate(methodology.hypergraph.skill_ids)}
        meta_index = {mid: i for i, mid in enumerate(methodology.hypergraph.metaskill_ids)}
        return cls(
            p_true=p,
            m_true=m,
            behaviour=behaviour,
            methodology=methodology,
            rng=rng,
            skill_index=skill_index,
            metaskill_index=meta_index,
            prereq_perturbation=prereq_perturbation,
        )

    # -- internal helpers ---------------------------------------------------

    def _task_lookup(self, target_id: str) -> Task | None:
        md = self.methodology
        if target_id in md.tasks:
            return md.tasks[target_id]
        if target_id in md.transfer_tasks:
            return md.transfer_tasks[target_id]
        if target_id in md.drills:
            return md.drills[target_id]
        if target_id in md.probes:
            return md.probes[target_id]
        return None

    def _skill_mastery(self, task: Task) -> float:
        """σ(mean logit(p_true[required])) — used by honest/help_seeker."""

        if not task.required_skills:
            return 0.5
        idx = [self.skill_index[s] for s in task.required_skills if s in self.skill_index]
        if not idx:
            return 0.5
        logits = _logit_safe(self.p_true[idx])
        return float(expit(np.mean(logits)))

    def _error_signature_match(self, task: Task) -> str | None:
        """Pick an error type for a wrong answer, possibly matching an
        error_signature edge in the hypergraph. With probability 0.4 the error
        is a "characteristic" one (links to an actual error vertex), otherwise
        the error is generic."""

        if self.rng.random() < 0.4 and task.required_skills:
            # find error vertices linked (via error_signature) to one of the required skills
            H = self.methodology.hypergraph
            cands: list[str] = []
            for eid, e in H.edges.items():
                if e.kind != "error_signature":
                    continue
                if set(e.head) & set(task.required_skills):
                    cands.extend(e.tail)
            if cands:
                return str(self.rng.choice(cands))
        return None

    # -- observation kernel -------------------------------------------------

    def observe(self, action: Action) -> Observation:
        """Generate ``o_t`` for ``a_t``. Branches on self.behaviour."""

        task = self._task_lookup(action.target_id)

        # Non-content actions: theory / hint / retention return cheap defaults.
        if action.type in ("microtheory", "retention"):
            return Observation(
                correct=True,
                answer_format="open_text",
                time_seconds=10.0,
                hint_requested=False,
                n_hints_used=0,
                hint_level_reached=0,
                self_explanation=True,
                copying_pattern=False,
            )
        if action.type == "hint":
            # A hint event just increments the counter; correctness undefined.
            # hint_requested is *False* on the hint observation itself — the
            # request has been honoured, so the next step should not loop back
            # into another hint without a fresh request from the student.
            level = max(0, int(action.hint_level))
            self._hints_used[action.target_id] = max(
                self._hints_used.get(action.target_id, 0), level
            )
            return Observation(
                correct=False,
                answer_format="open_text",
                time_seconds=5.0,
                hint_requested=False,
                n_hints_used=level,
                hint_level_reached=level,
                self_explanation=False,
                copying_pattern=False,
            )

        if task is None:
            # unknown target → degenerate observation
            return Observation(
                correct=False,
                answer_format="open_text",
                time_seconds=10.0,
                hint_requested=False,
                n_hints_used=0,
                hint_level_reached=0,
                self_explanation=False,
                copying_pattern=False,
            )

        baseline_time = task.time_estimate_seconds
        rng = self.rng

        # Behaviour branches -----------------------------------------------
        if self.behaviour == "guesser":
            p_correct = 1.0 / task.n_options if task.answer_format == "multiple_choice" else 0.10
            time_seconds = float(rng.gamma(shape=2.0, scale=baseline_time / 8.0))
            hint_requested = bool(rng.random() < 0.05)
            self_explanation = False
            copying_pattern = False

        elif self.behaviour == "copier":
            p_correct = 0.85
            if rng.random() < 0.5:
                time_seconds = float(rng.gamma(shape=2.0, scale=baseline_time / 8.0))
            else:
                time_seconds = float(rng.gamma(shape=8.0, scale=baseline_time / 4.0))
            hint_requested = False
            self_explanation = False
            copying_pattern = True

        elif self.behaviour == "help_seeker":
            base = self._skill_mastery(task)
            bonus_per_hint = 0.20
            n_hints = self._hints_used.get(action.target_id, 0)
            p_correct = min(0.95, base + bonus_per_hint * n_hints)
            time_seconds = float(rng.gamma(shape=4.0, scale=baseline_time / 4.0))
            hint_requested = bool(rng.random() < 0.60)
            self_explanation = bool(rng.random() < 0.30)
            copying_pattern = False

        else:  # honest
            base = self._skill_mastery(task)
            p_correct = base
            time_seconds = float(rng.gamma(shape=4.0, scale=baseline_time / 4.0))
            hint_requested = bool(base < 0.5 and rng.random() < 0.10)
            self_explanation = bool(rng.random() < 0.60)
            copying_pattern = False

        # Apply retention forgetting if simulated time has advanced.
        if self._retention_offset_days > 0:
            decay = float(np.exp(-FORGETTING_LAMBDA_PER_DAY * self._retention_offset_days))
            p_correct *= decay

        p_correct = float(np.clip(p_correct, 0.0, 1.0))
        correct = bool(rng.random() < p_correct)
        error_type = self._error_signature_match(task) if not correct else None

        return Observation(
            correct=correct,
            answer_format=task.answer_format,
            time_seconds=time_seconds,
            hint_requested=hint_requested,
            n_hints_used=self._hints_used.get(action.target_id, 0),
            hint_level_reached=self._hints_used.get(action.target_id, 0),
            self_explanation=self_explanation,
            copying_pattern=copying_pattern,
            error_type=error_type,
        )

    # -- dynamics -----------------------------------------------------------

    def update_p_true(self, action: Action, obs: Observation) -> None:
        """Eq. (7) + Eq. (A3): logistic-normal mastery update.

            μ_{t,k} = 1[correct]·(η_0(a) + Σ_{j∈MetaFor(k)} η_{1,k,j}·m_j)
                      - 1[error]·η_{-1}(a, errtype)
            p_{t+1,k} = σ(logit(p_{t,k}) + ξ_{t,k}),   ξ ~ N(μ, σ²).
        """

        task = self._task_lookup(action.target_id)
        if task is None or action.type in ("hint", "microtheory", "retention"):
            return  # no skill movement on these actions

        H = self.methodology.hypergraph
        required_idx = [
            self.skill_index[s]
            for s in task.required_skills
            if s in self.skill_index
        ]
        if not required_idx:
            return

        eta0 = ETA0_BY_ACTION.get(action.type, 0.0)
        eta_minus = ETA_MINUS_SIGNATURE if obs.error_type else ETA_MINUS_DEFAULT

        # Meta-skill coupling: for each required skill k, sum η_{1,k,j}·m_j over
        # j ∈ MetaFor(k), where MetaFor(k) = { j : ∃ edge meta_helps_learn (M_j) → ... ∋ k }.
        meta_term = np.zeros(len(required_idx), dtype=float)
        for j_local, idx in enumerate(required_idx):
            skill_id = self.methodology.hypergraph.skill_ids[idx]
            for e in H.edges.values():
                if e.kind != "meta_helps_learn":
                    continue
                if skill_id not in e.head:
                    continue
                for mid in e.tail:
                    if mid in self.metaskill_index:
                        m_j = float(self.m_true[self.metaskill_index[mid]])
                        meta_term[j_local] += ETA1_META * m_j

        if obs.correct:
            mu = eta0 + meta_term
        else:
            mu = -eta_minus * np.ones_like(meta_term)

        xi = self.rng.normal(loc=mu, scale=SIGMA_K)
        new_logits = _logit_safe(self.p_true[required_idx]) + xi
        self.p_true[required_idx] = _expit_safe(new_logits)

        # Error redistribution: if there is an error_signature pointing to OTHER
        # skills not in required_idx, those skills also decrement (Eq. A3 final
        # remark; §II.B on error redistribution).
        if not obs.correct and obs.error_type:
            extra_skills: set[int] = set()
            for e in H.edges.values():
                if e.kind != "error_signature":
                    continue
                if obs.error_type in e.tail:
                    for sk in e.head:
                        if sk in self.skill_index:
                            extra_skills.add(self.skill_index[sk])
            extra_skills.difference_update(required_idx)
            if extra_skills:
                extra_idx = sorted(extra_skills)
                xi_extra = self.rng.normal(loc=-eta_minus * 0.5, scale=SIGMA_K, size=len(extra_idx))
                logits_extra = _logit_safe(self.p_true[extra_idx]) + xi_extra
                self.p_true[extra_idx] = _expit_safe(logits_extra)

        # Meta-skill update (small bump on correct answer using a trains_metaskill
        # edge if any; otherwise just slight regression to the mean).
        if obs.correct and self.m_true.size > 0:
            for e in H.edges.values():
                if e.kind != "trains_metaskill":
                    continue
                if action.target_id not in e.tail:
                    continue
                for mid in e.head:
                    if mid in self.metaskill_index:
                        j = self.metaskill_index[mid]
                        bump = self.rng.normal(loc=0.05, scale=SIGMA_J)
                        new_mlogit = _logit_safe(self.m_true[j : j + 1]) + bump
                        self.m_true[j] = float(_expit_safe(new_mlogit)[0])

    # -- retention ----------------------------------------------------------

    def advance_time(self, delta_days: float) -> None:
        """Bookkeeping for retention probe — accumulates forgetting interval."""
        self._retention_offset_days = float(delta_days)
