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


def test_all_required_skills_have_trains_edges() -> None:
    """Regression: every skill referenced by a task must be the head of at
    least one ``trains`` edge. Without this, the typed-hypergraph
    propagation operator has no direct task→skill channel for that skill."""

    md = load_probability_methodology(PROB_PATH)
    trained: set[str] = set()
    for e in md.hypergraph.edges.values():
        if e.kind == "trains":
            trained.update(e.head)
    required: set[str] = set()
    for t in md.tasks.values():
        required.update(t.required_skills)
    missing = required - trained
    assert not missing, f"skills with no trains edge: {sorted(missing)}"


def test_minimum_edge_counts() -> None:
    """Regression: catch silent regressions in the metaskill / trains graph
    if a future edit drops to the multi-tail / multi-head form."""

    from collections import Counter

    md = load_probability_methodology(PROB_PATH)
    kinds = Counter(e.kind for e in md.hypergraph.edges.values())
    assert kinds["trains"] >= 40, f"trains edges: {kinds['trains']} (expected >= 40)"
    assert (
        kinds["trains_metaskill"] >= 30
    ), f"trains_metaskill edges: {kinds['trains_metaskill']} (expected >= 30)"
    assert (
        kinds["meta_helps_learn"] >= 15
    ), f"meta_helps_learn edges: {kinds['meta_helps_learn']} (expected >= 15)"


def test_every_skill_has_min_3_tasks() -> None:
    """Each skill should be trained by at least 3 tasks — content quality
    invariant introduced in v0.1.5 to prevent the v0.1.4 over-overlap
    pattern from re-emerging (three basics in 88% of tasks)."""

    from collections import Counter

    md = load_probability_methodology(PROB_PATH)
    counts: Counter[str] = Counter()
    for t in md.tasks.values():
        for s in t.required_skills:
            counts[s] += 1
    for skill in md.hypergraph.skill_ids:
        n = counts.get(skill, 0)
        assert n >= 3, f"skill {skill} has only {n} tasks (expected >= 3)"


def test_mean_required_skills_per_task_low() -> None:
    """Mean required_skills per task should be ~1 — each task targets a
    primary skill, prerequisites are expressed via concept → skill
    `requires` edges, not by inflating required_skills lists."""

    md = load_probability_methodology(PROB_PATH)
    sizes = [len(t.required_skills) for t in md.tasks.values()]
    mean = sum(sizes) / len(sizes)
    assert mean < 1.5, f"mean required_skills/task = {mean:.2f} (expected < 1.5)"
