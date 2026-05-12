"""Tests for the methodology loader / generator."""

from __future__ import annotations

from pathlib import Path

from l2t_sim.methodology import generate_synthetic_methodology

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_is_dag() -> None:
    md = generate_synthetic_methodology(seed=7)
    seen = set()
    for a, b in md.prereq_dag:
        seen.add((a, b))
    # Sufficient & easy: skills are S_0..S_K-1 and edges go (i, j) with i < j.
    for a, b in md.prereq_dag:
        ai = int(a.split("_")[1])
        bi = int(b.split("_")[1])
        assert ai < bi, (a, b)


def test_synthetic_reproducible() -> None:
    md1 = generate_synthetic_methodology(seed=11)
    md2 = generate_synthetic_methodology(seed=11)
    assert set(md1.hypergraph.vertices) == set(md2.hypergraph.vertices)
    assert set(md1.hypergraph.edges) == set(md2.hypergraph.edges)
    assert md1.prereq_dag == md2.prereq_dag


def test_no_dangling_references() -> None:
    md = generate_synthetic_methodology(seed=3)
    vids = set(md.hypergraph.vertices)
    for e in md.hypergraph.edges.values():
        for v in (*e.tail, *e.head):
            assert v in vids
