"""Bayesian student model with reliability-weighted updates (Eq. 4).

The propagation operator ``G(H)`` is cached once per methodology — the
hypergraph never changes during a run, so reusing the cached matrix turns
the per-step update into a tiny matmul.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import expit, logit

from .core_types import Action, Observation
from .hypergraph import Hypergraph, build_propagation_operator
from .methodology import MethodologyData


CLIP_LO, CLIP_HI = 1e-6, 1 - 1e-6


@dataclass
class BeliefState:
    """``b_t`` distilled to the marginals over (skills, meta-skills)."""

    p_hat: np.ndarray           # shape (K,), in (0,1)
    m_hat: np.ndarray           # shape (M,), in (0,1)
    history: List[tuple] = field(default_factory=list)  # for policy context


class StudentModel:
    """Bayesian learner-state estimator. Implements Eq. (4)."""

    def __init__(
        self,
        hypergraph: Hypergraph,
        methodology: MethodologyData,
        use_reliability: bool = True,
        propagation_op: np.ndarray | None = None,
    ) -> None:
        self.H = hypergraph
        self.methodology = methodology
        self.use_reliability = use_reliability

        self.skill_ids = list(hypergraph.skill_ids)
        self.metaskill_ids = list(hypergraph.metaskill_ids)
        self.skill_index = {sid: i for i, sid in enumerate(self.skill_ids)}
        self.metaskill_index = {mid: i for i, mid in enumerate(self.metaskill_ids)}

        self.G_H = (
            propagation_op
            if propagation_op is not None
            else build_propagation_operator(hypergraph)
        )
        self.edge_ids = list(hypergraph.edge_ids)
        self.edge_index = {eid: i for i, eid in enumerate(self.edge_ids)}

        # Pre-compute which edges are activated by an action targeting a given
        # task / drill / probe id. An edge is activated if the target id appears
        # in the edge's support.
        self._edges_by_target: dict[str, list[str]] = {}
        for eid, e in hypergraph.edges.items():
            for vid in e.support():
                self._edges_by_target.setdefault(vid, []).append(eid)

    # -- belief --------------------------------------------------------------

    def initial_belief(self) -> BeliefState:
        K = len(self.skill_ids)
        M = max(len(self.metaskill_ids), 0)
        return BeliefState(
            p_hat=0.5 * np.ones(K, dtype=float),
            m_hat=0.5 * np.ones(M, dtype=float) if M > 0 else np.zeros(0),
        )

    # -- update --------------------------------------------------------------

    def _build_signal(self, action: Action, observation: Observation) -> np.ndarray:
        """Build u_θ(a_t, o_t) ∈ R^|E|.

        Component-wise:
          * For each hyperedge whose support contains ``action.target_id``,
            ``u[e] = +1`` if the observation is correct, ``-1`` otherwise.
          * Independently, for every ``error_signature`` edge whose tail
            contains the reported ``error_type``, ``u[e] = -1`` (overwriting).
        """

        u = np.zeros(len(self.edge_ids), dtype=float)

        # Action does not produce a model-update for non-evaluative types.
        if action.type in ("hint", "microtheory", "retention"):
            return u

        sign = 1.0 if observation.correct else -1.0
        for eid in self._edges_by_target.get(action.target_id, []):
            u[self.edge_index[eid]] = sign

        if observation.error_type is not None and not observation.correct:
            for eid, e in self.H.edges.items():
                if e.kind != "error_signature":
                    continue
                if observation.error_type in e.tail:
                    u[self.edge_index[eid]] = -1.0
        return u

    def bayes_update(
        self,
        belief: BeliefState,
        action: Action,
        observation: Observation,
        q_t: float,
    ) -> BeliefState:
        """Eq. (4): ``p_{t+1} = σ( logit(p_t) + q_t · G(H) · u )``."""

        u = self._build_signal(action, observation)
        if not np.any(u):
            new_history = belief.history + [(action.type, action.target_id, observation.correct, q_t)]
            return BeliefState(
                p_hat=belief.p_hat.copy(),
                m_hat=belief.m_hat.copy(),
                history=new_history,
            )

        w = q_t if self.use_reliability else 1.0
        delta_logits = w * (self.G_H @ u)
        if len(delta_logits) != len(belief.p_hat):
            # Defensive: should match by construction.
            raise RuntimeError(
                f"shape mismatch: G(H)·u={delta_logits.shape}, p_hat={belief.p_hat.shape}"
            )

        new_logit = logit(np.clip(belief.p_hat, CLIP_LO, CLIP_HI)) + delta_logits
        p_new = np.clip(expit(new_logit), CLIP_LO, CLIP_HI)

        # Meta-skills: bump on trains_metaskill edges activated by the action.
        m_new = belief.m_hat.copy()
        if m_new.size > 0:
            for eid, e in self.H.edges.items():
                if e.kind != "trains_metaskill":
                    continue
                if action.target_id not in e.tail:
                    continue
                bump = 0.10 * (1.0 if observation.correct else -1.0) * w
                for mid in e.head:
                    if mid in self.metaskill_index:
                        j = self.metaskill_index[mid]
                        new_m_logit = logit(np.clip(m_new[j], CLIP_LO, CLIP_HI)) + bump
                        m_new[j] = float(np.clip(expit(new_m_logit), CLIP_LO, CLIP_HI))

        new_history = belief.history + [(action.type, action.target_id, observation.correct, q_t)]
        return BeliefState(p_hat=p_new, m_hat=m_new, history=new_history)
