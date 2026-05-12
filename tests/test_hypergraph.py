"""Tests for the typed hypergraph and the propagation operator G(H)."""

from __future__ import annotations

import numpy as np

from l2t_sim.hypergraph import (
    DEFAULT_TYPE_WEIGHTS,
    Hypergraph,
    build_propagation_operator,
    lemma1_bound,
)


def _simple_graph() -> Hypergraph:
    H = Hypergraph()
    for s in ("S1", "S2", "S3"):
        H.add_vertex(s, "skill")
    H.add_vertex("T1", "task")
    H.add_edge("e1", ("T1",), ("S1", "S2"), "requires", weight=1.0)
    H.add_edge("e2", ("T1",), ("S2", "S3"), "trains", weight=1.0)
    return H


def test_simple_hypergraph_dimensions() -> None:
    H = _simple_graph()
    G = build_propagation_operator(H)
    assert G.shape == (3, 2), G.shape
    assert np.isfinite(G).all()


def test_boundedness_lemma1() -> None:
    rng = np.random.default_rng(0)
    H = Hypergraph()
    K = 50
    for i in range(K):
        H.add_vertex(f"S{i}", "skill")
    # Add a single task vertex and 200 hyperedges of mixed types.
    H.add_vertex("T", "task")
    kinds = list(DEFAULT_TYPE_WEIGHTS.keys())
    for j in range(200):
        kind = kinds[j % len(kinds)]
        n = int(rng.integers(2, 5))
        skills = rng.choice(K, size=n, replace=False)
        head = tuple(f"S{i}" for i in skills)
        weight = float(rng.uniform(0.5, 2.0))
        H.add_edge(f"e{j}", ("T",), head, kind, weight=weight)
    G = build_propagation_operator(H)
    s = float(np.linalg.norm(G, ord=2))  # spectral norm
    bound = lemma1_bound(H)
    assert s <= bound + 1e-6, f"spectral={s}, bound={bound}"


def test_isolated_skill_does_not_break() -> None:
    H = Hypergraph()
    H.add_vertex("S1", "skill")
    H.add_vertex("S_iso", "skill")
    H.add_vertex("T", "task")
    H.add_edge("e1", ("T",), ("S1",), "requires")
    G = build_propagation_operator(H)
    assert G.shape == (2, 1)
    # The row of the isolated skill is identically zero.
    assert np.allclose(G[1], 0.0)
    assert not np.isnan(G).any()


def test_reproducibility() -> None:
    H1 = _simple_graph()
    H2 = _simple_graph()
    G1 = build_propagation_operator(H1)
    G2 = build_propagation_operator(H2)
    assert np.allclose(G1, G2)


def test_symmetric_normalization_bounded_row_sum() -> None:
    H = _simple_graph()
    G = build_propagation_operator(H)
    # No row should dominate; absolute row sum is bounded above by a constant
    # proportional to the entry magnitudes — finite & reasonable.
    assert np.all(np.isfinite(G))
    assert np.max(np.abs(G).sum(axis=1)) < 10.0
