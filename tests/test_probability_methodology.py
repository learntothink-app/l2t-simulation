"""Smoke tests for the probability methodology loader."""

from __future__ import annotations

from pathlib import Path

from l2t_sim.methodology import load_probability_methodology


ROOT = Path(__file__).resolve().parent.parent
PROB_PATH = ROOT / "methodologies" / "probability_basics.json"


def test_loads_without_errors() -> None:
    md = load_probability_methodology(PROB_PATH)
    assert md.name == "probability"


def test_expected_structure() -> None:
    md = load_probability_methodology(PROB_PATH)
    assert len(md.hypergraph.skill_ids) >= 10, "expected at least 10 skills"
    assert len(md.hypergraph.concept_ids) >= 8, "expected at least 8 concepts"
    assert len(md.tasks) >= 20, "expected at least 20 training tasks"
    assert len(md.transfer_tasks) >= 15, "expected at least 15 transfer tasks"
    assert len(md.microtheories) >= 8, "expected at least 8 microtheories"


def test_holdout_split() -> None:
    md = load_probability_methodology(PROB_PATH)
    # 18 transfer tasks → 8 are reserved as holdout.
    assert len(md.holdout_transfer_ids) == 8
    # Main-loop pool is the rest.
    assert len(md.main_loop_transfer_ids()) == len(md.transfer_tasks) - 8


def test_all_task_skills_exist_in_hypergraph() -> None:
    md = load_probability_methodology(PROB_PATH)
    skill_ids = set(md.hypergraph.skill_ids)
    for task in list(md.tasks.values()) + list(md.transfer_tasks.values()):
        for s in task.required_skills:
            assert s in skill_ids, f"task {task.id} references missing skill {s}"


def test_all_task_concepts_exist() -> None:
    md = load_probability_methodology(PROB_PATH)
    concept_ids = set(md.hypergraph.concept_ids)
    for task in md.tasks.values():
        for c in task.required_concepts:
            assert c in concept_ids, f"task {task.id} references missing concept {c}"


def test_error_signature_edges_exist() -> None:
    md = load_probability_methodology(PROB_PATH)
    count = sum(1 for e in md.hypergraph.edges.values() if e.kind == "error_signature")
    assert count >= 8, f"expected at least 8 error_signature edges, got {count}"


def test_metaskill_edges_exist() -> None:
    md = load_probability_methodology(PROB_PATH)
    count = sum(
        1
        for e in md.hypergraph.edges.values()
        if e.kind in ("meta_helps_learn", "trains_metaskill")
    )
    assert count >= 6, f"expected at least 6 metaskill-related edges, got {count}"


def test_prereq_dag_present() -> None:
    md = load_probability_methodology(PROB_PATH)
    assert len(md.prereq_dag) >= 5, "expected at least 5 prerequisite edges"


def test_microtheory_concepts_resolve() -> None:
    md = load_probability_methodology(PROB_PATH)
    concept_ids = set(md.hypergraph.concept_ids)
    for mt in md.microtheories.values():
        for c in mt.concepts:
            assert c in concept_ids, f"microtheory {mt.id} references missing concept {c}"
