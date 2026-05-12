"""Methodologies: JSON-driven loader and synthetic abstract generator.

A *methodology* is the union of:
  * a typed hypergraph ``H`` (knowledge structure), and
  * the L2T-protocol artefacts: tasks, drills, probes, transfer-sets,
    micro-theories, retention schedule, hint ladders.

Both halves are carried in :class:`MethodologyData`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import hypergraph as hg
from .hypergraph import Hyperedge, Hypergraph, Vertex
from .utils import load_json, save_json


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data records


@dataclass(frozen=True)
class Task:
    """L2T task / drill / probe record (Sec. IV.A).

    Required-skill / required-concept references are id-strings into the
    hypergraph; missing references are dropped silently so JSON fixtures
    with stray ids load robustly.
    """

    id: str
    kind: str  # "task" | "drill" | "probe" | "scaffolding" | "transfer"
    required_skills: tuple[str, ...]
    required_concepts: tuple[str, ...] = ()
    diagnoses_skills: tuple[str, ...] = ()
    trains_skills: tuple[str, ...] = ()
    hint_ladder_length: int = 6
    answer_format: str = "open_text"
    n_options: int = 4
    time_estimate_seconds: float = 120.0


@dataclass(frozen=True)
class MicroTheory:
    id: str
    concepts: tuple[str, ...]
    title: str = ""


@dataclass
class MethodologyData:
    """Everything a policy / simulation needs beyond the hypergraph itself."""

    name: str
    hypergraph: Hypergraph
    tasks: dict[str, Task] = field(default_factory=dict)
    drills: dict[str, Task] = field(default_factory=dict)
    probes: dict[str, Task] = field(default_factory=dict)
    transfer_tasks: dict[str, Task] = field(default_factory=dict)
    microtheories: dict[str, MicroTheory] = field(default_factory=dict)
    prereq_dag: list[tuple[str, str]] = field(default_factory=list)
    retention_schedule_days: tuple[int, ...] = (1, 3, 7, 14)
    hint_ladder_max: int = 6
    j_star_spoiler_level: int = 4  # "partial solution" threshold for m_hint
    # Holdout subset of transfer_tasks reserved for the terminal battery.
    # Policies must not pick these during the main loop — otherwise the
    # terminal m_transfer measurement degenerates into "repeat known tasks"
    # rather than "transfer to novel ones". Empty set = no split (e.g. when
    # the methodology has too few transfer tasks to support a holdout).
    holdout_transfer_ids: frozenset[str] = field(default_factory=frozenset)

    # -- selectors ----------------------------------------------------------

    @property
    def all_tasks(self) -> dict[str, Task]:
        out: dict[str, Task] = {}
        out.update(self.tasks)
        out.update(self.transfer_tasks)
        return out

    def task_ids_for_skill(self, skill_id: str) -> list[str]:
        return [tid for tid, t in self.tasks.items() if skill_id in t.required_skills]

    def drill_ids_for_skill(self, skill_id: str) -> list[str]:
        return [did for did, d in self.drills.items() if skill_id in d.trains_skills]

    def probe_ids_for_skill(self, skill_id: str) -> list[str]:
        return [pid for pid, p in self.probes.items() if skill_id in p.diagnoses_skills]

    def transfer_ids_for_skill(self, skill_id: str) -> list[str]:
        return [
            tid
            for tid, t in self.transfer_tasks.items()
            if skill_id in t.required_skills and tid not in self.holdout_transfer_ids
        ]

    def main_loop_transfer_ids(self) -> list[str]:
        """Transfer task ids visible to policies during the main loop."""
        return [tid for tid in self.transfer_tasks if tid not in self.holdout_transfer_ids]


# ---------------------------------------------------------------------------
# JSON-driven loader (used for probability_basics.json and any other
# similarly-structured fixture).


def _load_json_methodology(path: str | Path, name: str) -> MethodologyData:
    """Build a :class:`MethodologyData` from a JSON fixture matching the
    probability_basics schema (also compatible with the old Fano shape).

    The last 8 transfer tasks are reserved as holdout battery when the
    methodology has >= 16 transfer tasks; >8 but <16 yields a proportional
    split; otherwise the holdout is empty.
    """

    raw = load_json(path)
    H = Hypergraph()

    kind_map = {
        "concept": "concept",
        "skill": "skill",
        "metaskill": "metaskill",
        "error": "error",
        "topic": "topic",
    }
    for node in raw["hypergraph"]["nodes"]:
        vkind = kind_map.get(node["kind"])
        if vkind is None:
            continue
        if node["id"] in H.vertices:
            continue
        H.add_vertex(node["id"], vkind)

    tasks: dict[str, Task] = {}
    drills: dict[str, Task] = {}
    probes: dict[str, Task] = {}
    transfer_tasks: dict[str, Task] = {}
    microtheories: dict[str, MicroTheory] = {}

    def _safe_skill_refs(seq: Sequence[str] | None) -> tuple[str, ...]:
        if not seq:
            return ()
        return tuple(s for s in seq if s in H.vertices and H.vertices[s].vtype == "skill")

    def _safe_concept_refs(seq: Sequence[str] | None) -> tuple[str, ...]:
        if not seq:
            return ()
        return tuple(c for c in seq if c in H.vertices and H.vertices[c].vtype == "concept")

    for raw_t in raw.get("scaffolding_tasks", []) + raw.get("target_tasks", []):
        tid = raw_t["id"]
        if tid not in H.vertices:
            H.add_vertex(tid, "task")
        skills = _safe_skill_refs(raw_t.get("required_skills"))
        concepts = _safe_concept_refs(raw_t.get("required_concepts"))
        answer_format = str(raw_t.get("answer_format", "open_text"))
        n_options = int(raw_t.get("n_options", 4))
        tasks[tid] = Task(
            id=tid,
            kind="task",
            required_skills=skills,
            required_concepts=concepts,
            answer_format=answer_format,
            n_options=n_options,
            time_estimate_seconds=60.0 * float(raw_t.get("time_estimate_minutes", 2)),
        )
        for sk in skills:
            H.add_edge(f"R_{tid}_{sk}", (tid,), (sk,), "requires", weight=1.0)

    for raw_d in raw.get("drills", []):
        did = raw_d["id"]
        if did not in H.vertices:
            H.add_vertex(did, "drill")
        train_skills = _safe_skill_refs(raw_d.get("trains_skills"))
        drills[did] = Task(
            id=did,
            kind="drill",
            required_skills=train_skills,
            trains_skills=train_skills,
            time_estimate_seconds=60.0 * float(raw_d.get("estimated_time_minutes", 3)),
        )
        for sk in train_skills:
            eid = f"T_{did}_{sk}"
            if eid in H.edges:
                continue
            H.add_edge(eid, (did,), (sk,), "trains", weight=1.0)

    for raw_p in raw.get("probes", []):
        pid = raw_p["id"]
        if pid not in H.vertices:
            H.add_vertex(pid, "probe")
        diag_skills = _safe_skill_refs(raw_p.get("diagnoses_skills"))
        diag_concepts = _safe_concept_refs(raw_p.get("diagnoses_concepts"))
        probes[pid] = Task(
            id=pid,
            kind="probe",
            required_skills=diag_skills,
            required_concepts=diag_concepts,
            diagnoses_skills=diag_skills,
            answer_format="multiple_choice",
            n_options=4,
            time_estimate_seconds=60.0,
        )
        for sk in diag_skills:
            eid = f"DG_{pid}_{sk}"
            if eid in H.edges:
                continue
            H.add_edge(eid, (pid,), (sk,), "diagnoses", weight=1.0)

    for ts in raw.get("transfer_sets", []):
        target_skills = _safe_skill_refs(ts.get("target_skills"))
        target_concepts = _safe_concept_refs(ts.get("target_concepts"))
        for raw_t in ts.get("tasks", []):
            tid = raw_t["id"]
            if tid not in H.vertices:
                H.add_vertex(tid, "task")
            # Per-task required_skills override the set-level target_skills
            # when present.
            per_task_skills = _safe_skill_refs(raw_t.get("required_skills"))
            per_task_concepts = _safe_concept_refs(raw_t.get("required_concepts"))
            transfer_tasks[tid] = Task(
                id=tid,
                kind="transfer",
                required_skills=per_task_skills or target_skills,
                required_concepts=per_task_concepts or target_concepts,
                answer_format=str(raw_t.get("answer_format", "open_text")),
                n_options=int(raw_t.get("n_options", 4)),
                time_estimate_seconds=60.0 * float(raw_t.get("time_estimate_minutes", 3)),
            )
            for sk in per_task_skills or target_skills:
                eid = f"TV_{tid}_{sk}"
                if eid in H.edges:
                    continue
                H.add_edge(eid, (tid,), (sk,), "transfer_variant", weight=1.0)

    for mt in raw.get("micro_theories", []):
        concepts = _safe_concept_refs(mt.get("concept_ids"))
        microtheories[mt["id"]] = MicroTheory(
            id=mt["id"], concepts=concepts, title=mt.get("title", "")
        )

    for raw_e in raw["hypergraph"]["hyperedges"]:
        eid = raw_e["id"]
        if eid in H.edges:
            continue
        tail = tuple(v for v in raw_e["tail"] if v in H.vertices)
        head = tuple(v for v in raw_e["head"] if v in H.vertices)
        if not tail or not head:
            log.warning("dropping edge %s: tail/head missing from vertex set", eid)
            continue
        weight = hg.DEFAULT_TYPE_WEIGHTS.get(raw_e["kind"], 1.0)
        H.add_edge(eid, tail, head, raw_e["kind"], weight=weight)

    prereq = [(d["from"], d["to"]) for d in raw["hypergraph"].get("prereq_dag", [])]

    transfer_ids_ordered = list(transfer_tasks.keys())
    if len(transfer_ids_ordered) >= 16:
        holdout_ids = frozenset(transfer_ids_ordered[-8:])
    elif len(transfer_ids_ordered) > 8:
        n_holdout = len(transfer_ids_ordered) - max(1, len(transfer_ids_ordered) // 2)
        holdout_ids = frozenset(transfer_ids_ordered[-n_holdout:])
    else:
        holdout_ids = frozenset()

    return MethodologyData(
        name=name,
        hypergraph=H,
        tasks=tasks,
        drills=drills,
        probes=probes,
        transfer_tasks=transfer_tasks,
        microtheories=microtheories,
        prereq_dag=prereq,
        retention_schedule_days=tuple(raw.get("retention_schedule", [1, 3, 7, 14])),
        hint_ladder_max=6,
        holdout_transfer_ids=holdout_ids,
    )


def load_probability_methodology(path: str | Path) -> MethodologyData:
    """Build the canonical probability_basics methodology."""
    return _load_json_methodology(path, name="probability")


# ---------------------------------------------------------------------------
# Synthetic abstract methodology


def generate_synthetic_methodology(
    seed: int = 1,
    K_skills: int = 10,
    n_concepts: int = 8,
    n_metaskills: int = 3,
    n_errors: int = 12,
    n_tasks: int = 40,
    n_transfer: int = 15,
    n_drills: int = 20,
    n_probes: int = 8,
    prereq_prob: float = 0.15,
    hint_ladder_max: int = 6,
) -> MethodologyData:
    """Generate a synthetic abstract methodology with fixed (seeded) topology.

    Parameters match §VII.B "medium" of the paper. The resulting hypergraph is
    saved as JSON next to the methodologies package so the exact draw can be
    re-used by external scripts.
    """

    rng = np.random.default_rng(seed)
    H = Hypergraph()

    skills = [f"S_{i}" for i in range(K_skills)]
    for s in skills:
        H.add_vertex(s, "skill")
    concepts = [f"C_{i}" for i in range(n_concepts)]
    for c in concepts:
        H.add_vertex(c, "concept")
    metaskills = [f"M_{i}" for i in range(n_metaskills)]
    for m in metaskills:
        H.add_vertex(m, "metaskill")
    errors = [f"E_{i}" for i in range(n_errors)]
    for e in errors:
        H.add_vertex(e, "error")

    # Prerequisite DAG over skills (forward edges i<j only ⇒ acyclic).
    prereq: list[tuple[str, str]] = []
    for i in range(K_skills):
        for j in range(i + 1, K_skills):
            if rng.random() < prereq_prob:
                prereq.append((skills[i], skills[j]))

    # Concepts: each `requires` 1..3 skills.
    for c in concepts:
        k = int(rng.integers(1, 4))
        target = rng.choice(skills, size=k, replace=False)
        for sk in target:
            H.add_edge(f"RC_{c}_{sk}", (c,), (sk,), "requires", weight=1.0)

    # Metaskills: each `meta_helps_learn` 2..4 skills.
    for m in metaskills:
        k = int(rng.integers(2, 5))
        target = rng.choice(skills, size=k, replace=False)
        for sk in target:
            H.add_edge(f"MH_{m}_{sk}", (m,), (sk,), "meta_helps_learn", weight=1.0)

    # Errors: each `error_signature` 1..2 skills.
    for e in errors:
        k = int(rng.integers(1, 3))
        target = rng.choice(skills, size=k, replace=False)
        for sk in target:
            H.add_edge(f"ES_{e}_{sk}", (e,), (sk,), "error_signature", weight=1.0)

    # Tasks.
    tasks: dict[str, Task] = {}
    for i in range(n_tasks):
        tid = f"T_{i}"
        H.add_vertex(tid, "task")
        k = int(rng.integers(1, 4))
        target = list(rng.choice(skills, size=k, replace=False))
        for sk in target:
            H.add_edge(f"R_{tid}_{sk}", (tid,), (sk,), "requires", weight=1.0)
        if rng.random() < 0.4:
            for sk in target:
                eid = f"TR_{tid}_{sk}"
                if eid not in H.edges:
                    H.add_edge(eid, (tid,), (sk,), "trains", weight=1.0)
        # 1..2 required concepts per task — drives the theory-first invariant
        # and the theory-bonus mechanic in the environment.
        n_concepts_req = int(rng.integers(1, 3))
        req_concepts = tuple(rng.choice(concepts, size=n_concepts_req, replace=False))
        # half of tasks are multiple choice
        if rng.random() < 0.5:
            answer_format = "multiple_choice"
            n_options = int(rng.choice([3, 4, 5]))
        else:
            answer_format = "open_text"
            n_options = 4
        tasks[tid] = Task(
            id=tid,
            kind="task",
            required_skills=tuple(target),
            required_concepts=req_concepts,
            answer_format=answer_format,
            n_options=n_options,
            hint_ladder_length=hint_ladder_max,
            time_estimate_seconds=float(rng.uniform(60.0, 300.0)),
        )

    # Transfer tasks (variants of the first n_transfer regular tasks).
    transfer_tasks: dict[str, Task] = {}
    base_task_ids = list(tasks.keys())[:n_transfer]
    for i, base_tid in enumerate(base_task_ids):
        tid = f"TR_{i}"
        H.add_vertex(tid, "task")
        required = tasks[base_tid].required_skills
        base_concepts = tasks[base_tid].required_concepts
        H.add_edge(f"TV_{tid}_{base_tid}", (tid,), (base_tid,), "transfer_variant", weight=1.0)
        for sk in required:
            H.add_edge(f"R_{tid}_{sk}", (tid,), (sk,), "requires", weight=1.0)
        transfer_tasks[tid] = Task(
            id=tid,
            kind="transfer",
            required_skills=required,
            required_concepts=base_concepts,
            answer_format="open_text",
            time_estimate_seconds=float(rng.uniform(120.0, 360.0)),
        )

    # Drills.
    drills: dict[str, Task] = {}
    for i in range(n_drills):
        did = f"D_{i}"
        H.add_vertex(did, "drill")
        k = int(rng.integers(1, 3))
        target = list(rng.choice(skills, size=k, replace=False))
        for sk in target:
            H.add_edge(f"T_{did}_{sk}", (did,), (sk,), "trains", weight=1.0)
        drills[did] = Task(
            id=did,
            kind="drill",
            required_skills=tuple(target),
            trains_skills=tuple(target),
            answer_format="open_text",
            time_estimate_seconds=float(rng.uniform(30.0, 90.0)),
        )

    # trains_metaskill: each metaskill is trained by 3..6 tasks/drills.
    # Without these edges, m_meta is identically zero — see Eq. (A3) final
    # remark and the m_meta column of the result table.
    pool = list(tasks.keys()) + list(drills.keys())
    for m in metaskills:
        n_train = int(rng.integers(3, 7))
        if not pool:
            continue
        targets = rng.choice(pool, size=min(n_train, len(pool)), replace=False)
        for tid in targets:
            eid = f"TM_{m}_{tid}"
            if eid not in H.edges:
                H.add_edge(eid, (tid,), (m,), "trains_metaskill", weight=1.0)

    # Probes.
    probes: dict[str, Task] = {}
    for i in range(n_probes):
        pid = f"P_{i}"
        H.add_vertex(pid, "probe")
        k = int(rng.integers(1, 3))
        target = list(rng.choice(skills, size=k, replace=False))
        for sk in target:
            H.add_edge(f"DG_{pid}_{sk}", (pid,), (sk,), "diagnoses", weight=1.0)
        probes[pid] = Task(
            id=pid,
            kind="probe",
            required_skills=tuple(target),
            diagnoses_skills=tuple(target),
            answer_format="multiple_choice",
            n_options=int(rng.choice([3, 4, 5])),
            time_estimate_seconds=float(rng.uniform(20.0, 60.0)),
        )

    # Microtheories: 1 per concept, with a 1-1 concept reference.
    microtheories = {
        f"MT_{c}": MicroTheory(id=f"MT_{c}", concepts=(c,), title=f"Microtheory for {c}")
        for c in concepts
    }

    # Holdout battery: reserve the last 8 transfer tasks for the terminal
    # probe. Policies see only the first n_transfer-8 during the main loop.
    # Skip the split if we don't have enough tasks to leave a non-empty
    # main-loop pool.
    transfer_ids_ordered = list(transfer_tasks.keys())
    if len(transfer_ids_ordered) >= 16:
        holdout_ids = frozenset(transfer_ids_ordered[-8:])
    elif len(transfer_ids_ordered) > 8:
        # keep at least 1 transfer in main loop, rest are holdout
        n_holdout = len(transfer_ids_ordered) - max(1, len(transfer_ids_ordered) // 2)
        holdout_ids = frozenset(transfer_ids_ordered[-n_holdout:])
    else:
        holdout_ids = frozenset()

    md = MethodologyData(
        name="synthetic",
        hypergraph=H,
        tasks=tasks,
        drills=drills,
        probes=probes,
        transfer_tasks=transfer_tasks,
        microtheories=microtheories,
        prereq_dag=prereq,
        hint_ladder_max=hint_ladder_max,
        holdout_transfer_ids=holdout_ids,
    )
    return md


def save_synthetic_to_json(md: MethodologyData, path: str | Path) -> None:
    """Serialise a synthetic methodology to a Fano-shaped JSON for inspection."""

    payload = {
        "protocol_version": "0.1.0",
        "manifest": {"methodology_id": md.name, "title": "Synthetic abstract"},
        "hypergraph": {
            "nodes": [{"id": v.id, "kind": v.vtype} for v in md.hypergraph.vertices.values()],
            "hyperedges": [
                {
                    "id": e.id,
                    "kind": e.kind,
                    "tail": list(e.tail),
                    "head": list(e.head),
                    "weight": e.weight,
                }
                for e in md.hypergraph.edges.values()
            ],
            "prereq_dag": [{"from": a, "to": b} for a, b in md.prereq_dag],
        },
        "tasks": [task.__dict__ for task in md.tasks.values()],
        "transfer_tasks": [task.__dict__ for task in md.transfer_tasks.values()],
        "drills": [task.__dict__ for task in md.drills.values()],
        "probes": [task.__dict__ for task in md.probes.values()],
        "microtheories": [
            {"id": mt.id, "concepts": list(mt.concepts), "title": mt.title}
            for mt in md.microtheories.values()
        ],
    }
    save_json(path, payload)
