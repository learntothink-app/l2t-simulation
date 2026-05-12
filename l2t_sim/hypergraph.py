"""Typed knowledge hypergraph and propagation operator G(H).

Implements §II.A and Eq. (5) / Appendix A of the L2T paper. The hypergraph
carries pedagogical semantics through *typed* hyperedges; the operator
``G(H) ∈ R^{K × |E|}`` redistributes evidence from edges back to skill
vertices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


VertexType = Literal[
    "concept",
    "skill",
    "metaskill",
    "error",
    "task",
    "drill",
    "probe",
    "topic",
]
VERTEX_TYPES: tuple[str, ...] = (
    "concept",
    "skill",
    "metaskill",
    "error",
    "task",
    "drill",
    "probe",
    "topic",
)

EdgeKind = Literal[
    "requires",
    "trains",
    "diagnoses",
    "error_signature",
    "meta_helps_learn",
    "transfer_variant",
    "trains_metaskill",
]
EDGE_KINDS: tuple[str, ...] = (
    "requires",
    "trains",
    "diagnoses",
    "error_signature",
    "meta_helps_learn",
    "transfer_variant",
    "trains_metaskill",
)

# §App.A and Fig. 2 of the paper.
DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "requires": 1.0,
    "trains": 1.2,
    "diagnoses": 1.5,
    "error_signature": 1.8,
    "meta_helps_learn": 0.8,
    "transfer_variant": 1.0,
    "trains_metaskill": 0.8,
}


@dataclass(frozen=True)
class Vertex:
    """Vertex in the typed hypergraph. Cf. Eq. (1)."""

    id: str
    vtype: VertexType


@dataclass(frozen=True)
class Hyperedge:
    """Hyperedge e = (tail, head, kind). Cf. Eq. (2).

    ``tail`` generalises the source vertex of a directed graph edge to an
    *input* subset; ``head`` is the *output* subset.
    """

    id: str
    tail: tuple[str, ...]
    head: tuple[str, ...]
    kind: EdgeKind
    weight: float = 1.0  # per-edge weight w_{κ,e}

    def support(self) -> set[str]:
        return set(self.tail) | set(self.head)


class Hypergraph:
    """Mutable typed hypergraph H = (V, E)."""

    def __init__(self, type_weights: dict[str, float] | None = None) -> None:
        self.vertices: dict[str, Vertex] = {}
        self.edges: dict[str, Hyperedge] = {}
        self.type_weights: dict[str, float] = dict(type_weights or DEFAULT_TYPE_WEIGHTS)

    # -- structure -----------------------------------------------------------

    def add_vertex(self, vid: str, vtype: VertexType) -> Vertex:
        if vid in self.vertices:
            raise ValueError(f"duplicate vertex id: {vid}")
        if vtype not in VERTEX_TYPES:
            raise ValueError(f"unknown vertex type: {vtype}")
        v = Vertex(vid, vtype)
        self.vertices[vid] = v
        return v

    def add_edge(
        self,
        eid: str,
        tail: tuple[str, ...] | list[str],
        head: tuple[str, ...] | list[str],
        kind: EdgeKind,
        weight: float = 1.0,
    ) -> Hyperedge:
        if eid in self.edges:
            raise ValueError(f"duplicate edge id: {eid}")
        if kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {kind}")
        tail_t, head_t = tuple(tail), tuple(head)
        for vid in (*tail_t, *head_t):
            if vid not in self.vertices:
                raise ValueError(f"edge {eid} references unknown vertex {vid}")
        e = Hyperedge(eid, tail_t, head_t, kind, float(weight))
        self.edges[eid] = e
        return e

    # -- selectors -----------------------------------------------------------

    def _ids_of_type(self, vtype: VertexType) -> list[str]:
        return [v.id for v in self.vertices.values() if v.vtype == vtype]

    @property
    def skill_ids(self) -> list[str]:
        return self._ids_of_type("skill")

    @property
    def metaskill_ids(self) -> list[str]:
        return self._ids_of_type("metaskill")

    @property
    def concept_ids(self) -> list[str]:
        return self._ids_of_type("concept")

    @property
    def edge_ids(self) -> list[str]:
        return list(self.edges.keys())


def build_propagation_operator(
    H: Hypergraph,
    type_weights: dict[str, float] | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Build G(H) per Eq. (5):

        G(H) = D_v^{-1/2} · (Σ_κ w_κ · H_κ · W_κ) · D_e^{-1/2}.

    Rows of the returned matrix are indexed by *skill* vertices (so the
    output of ``G(H) · u`` lives in skill-logit space, matching Eq. (4)).
    Columns are indexed by *all* hyperedges in ``H.edge_ids`` order. The
    sum over types is exactly the construction in App. A; isolated skills
    receive an ``eps`` floor on their degree to avoid division by zero
    (the row stays zero in M_inner, so G is zero in that row).
    """

    tw = dict(type_weights or H.type_weights)
    skills = H.skill_ids
    if not skills:
        raise ValueError("hypergraph has no skill vertices")

    skill_index: dict[str, int] = {sid: i for i, sid in enumerate(skills)}
    K = len(skills)

    edge_ids = H.edge_ids
    E = len(edge_ids)
    edge_index: dict[str, int] = {eid: i for i, eid in enumerate(edge_ids)}

    M_inner = np.zeros((K, E), dtype=float)
    Dv_diag = np.zeros(K, dtype=float)
    De_diag = np.zeros(E, dtype=float)

    for eid, e in H.edges.items():
        col = edge_index[eid]
        w_kappa = float(tw.get(e.kind, 1.0))
        w_edge = float(e.weight)
        support = e.support()
        De_diag[col] = float(len(support))
        # skills touched by this edge (intersection of edge support with skill set)
        for vid in support:
            if vid not in skill_index:
                continue
            row = skill_index[vid]
            # Inner sum Σ_κ w_κ · H_κ · W_κ; only the column for this edge gets
            # contribution w_κ · w_edge.
            M_inner[row, col] += w_kappa * w_edge
            # Weighted degree (Eq. A1):
            #   [D_v]_kk = Σ_κ w_κ · Σ_{e ∋ k, e ∈ E_κ} w_{κ,e}.
            Dv_diag[row] += w_kappa * w_edge

    # Stabilise zero degrees (isolated skill / orphan edge support).
    Dv_safe = np.where(Dv_diag > 0, Dv_diag, eps)
    De_safe = np.where(De_diag > 0, De_diag, eps)

    Dv_inv_sqrt = 1.0 / np.sqrt(Dv_safe)
    De_inv_sqrt = 1.0 / np.sqrt(De_safe)

    # G = diag(Dv^{-1/2}) @ M_inner @ diag(De^{-1/2})
    G = (M_inner * De_inv_sqrt[np.newaxis, :]) * Dv_inv_sqrt[:, np.newaxis]

    # Zero out rows for skills that received no edge mass (cosmetic).
    G[Dv_diag <= 0, :] = 0.0
    G[:, De_diag <= 0] = 0.0
    return G


@dataclass
class PropagationOperator:
    """Cached G(H) and its index maps."""

    G: np.ndarray
    skill_index: dict[str, int]
    edge_index: dict[str, int]

    @classmethod
    def build(cls, H: Hypergraph) -> "PropagationOperator":
        G = build_propagation_operator(H)
        skills = H.skill_ids
        skill_index = {sid: i for i, sid in enumerate(skills)}
        edge_index = {eid: i for i, eid in enumerate(H.edge_ids)}
        return cls(G=G, skill_index=skill_index, edge_index=edge_index)


def lemma1_bound(H: Hypergraph, type_weights: dict[str, float] | None = None) -> float:
    """Numerical evaluation of the RHS of Lemma 1 (App. A, Eq. A2):

        ||G(H)||_2 ≤ Σ_κ sqrt( w_κ · max_{e ∈ E_κ} w_{κ,e} ).

    Returned as a positive float; an empty hypergraph yields 0.
    """

    tw = dict(type_weights or H.type_weights)
    by_kind: dict[str, list[float]] = {}
    for e in H.edges.values():
        by_kind.setdefault(e.kind, []).append(e.weight)
    total = 0.0
    for kind, weights in by_kind.items():
        w_kappa = float(tw.get(kind, 1.0))
        total += float(np.sqrt(w_kappa * max(weights)))
    return total
