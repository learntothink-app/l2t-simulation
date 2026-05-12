"""One-shot generator: append missing trains / trains_metaskill /
meta_helps_learn / error_signature edges to methodologies/probability_basics.json.

The v0.1.2 JSON declared trains_metaskill and meta_helps_learn as a few
multi-tail / multi-head edges. Those collapse the propagation weight per
(task, metaskill) pair to one share of a single edge, which under the
typed-hypergraph G(H) (Eq. 5) gives much smaller effective propagation
than expanded single-edge form. v0.1.3 expands them.

Run once:

    python scripts/fix_probability_edges.py

It is idempotent w.r.t. (kind, tail, head): edges that already exist with
the same triple are skipped. Run again after manual edits and only the
missing entries will be added.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "methodologies" / "probability_basics.json"


# ---------------------------------------------------------------------------
# Source-of-truth lists. Hand-curated per v0.1.3 spec §2.2 / §2.3.

ALL_SKILLS = [
    "S_enumerate_outcomes",
    "S_count_favorable",
    "S_compute_simple_probability",
    "S_apply_complement_rule",
    "S_recognize_independence",
    "S_compute_joint_independent",
    "S_apply_addition_mutually_exclusive",
    "S_apply_addition_general",
    "S_compute_conditional",
    "S_apply_bayes_simple",
    "S_compute_expected_value_discrete",
    "S_distinguish_probability_frequency",
]

ENUMERATION_TASKS = [
    # Group 1 — basic enumeration
    "T_basic_2coins_at_least_one_H",
    "T_basic_die_even",
    "T_basic_2dice_sum7",
    "T_basic_bag_3red_2blue",
    "T_basic_card_heart",
    # Group 5 — general addition (also exercises careful enumeration)
    "T_general_card_red_or_face",
    "T_general_die_even_or_high",
    "T_general_2dice_first1_or_second6",
]

ANDOR_TASKS = [
    # Group 4 — mutually exclusive ("or")
    "T_mutex_die_1_or_2",
    "T_mutex_bag_red_or_blue",
    "T_mutex_card_ace_or_king",
    # Group 5 — general addition ("or" with overlap)
    "T_general_card_red_or_face",
    "T_general_die_even_or_high",
    "T_general_2dice_first1_or_second6",
]

# Per spec §2.4 — extra error_signature edges.
ERROR_SIGNATURE_EXTRA = [
    ("E_equiprobability_bias", "S_count_favorable"),
    ("E_conjunction_fallacy", "MS_check_range_validity"),
    ("E_independence_assumption_error", "S_compute_joint_independent"),
    ("E_addition_without_subtraction", "S_apply_addition_mutually_exclusive"),
]


def _existing_triples(edges: list[dict]) -> set[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    out: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for e in edges:
        out.add((e["kind"], tuple(e["tail"]), tuple(e["head"])))
    return out


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    edges: list[dict] = data["hypergraph"]["hyperedges"]
    existing = _existing_triples(edges)
    added = 0

    def _push(kind: str, tail: list[str], head: list[str], weight: float, eid: str) -> None:
        nonlocal added
        key = (kind, tuple(tail), tuple(head))
        if key in existing:
            return
        edges.append({"id": eid, "kind": kind, "tail": tail, "head": head, "weight": weight})
        existing.add(key)
        added += 1

    # Collect all regular task entries with their required_skills.
    tasks_with_skills: list[tuple[str, list[str]]] = []
    for raw_t in data.get("scaffolding_tasks", []) + data.get("target_tasks", []):
        tasks_with_skills.append((raw_t["id"], list(raw_t.get("required_skills", []))))
    all_task_ids = [tid for tid, _ in tasks_with_skills]

    # 1) trains: each (task, required_skill) → trains edge, weight 1.0.
    for tid, skills in tasks_with_skills:
        for sk in skills:
            _push(
                "trains",
                tail=[tid],
                head=[sk],
                weight=1.0,
                eid=f"TRN_{tid}__{sk}",
            )

    # 2) trains_metaskill: every task trains MS_check_range_validity.
    for tid in all_task_ids:
        _push(
            "trains_metaskill",
            tail=[tid],
            head=["MS_check_range_validity"],
            weight=0.5,
            eid=f"TRM_{tid}__MS_check_range_validity",
        )

    # 3) trains_metaskill: enumeration tasks train MS_enumerate_carefully.
    for tid in ENUMERATION_TASKS:
        _push(
            "trains_metaskill",
            tail=[tid],
            head=["MS_enumerate_carefully"],
            weight=0.5,
            eid=f"TRM_{tid}__MS_enumerate_carefully",
        )

    # 4) trains_metaskill: and/or tasks train MS_distinguish_and_or.
    for tid in ANDOR_TASKS:
        _push(
            "trains_metaskill",
            tail=[tid],
            head=["MS_distinguish_and_or"],
            weight=0.5,
            eid=f"TRM_{tid}__MS_distinguish_and_or",
        )

    # 5) meta_helps_learn: MS_check_range_validity → all 12 skills (weight 0.10).
    for sk in ALL_SKILLS:
        _push(
            "meta_helps_learn",
            tail=["MS_check_range_validity"],
            head=[sk],
            weight=0.10,
            eid=f"MHL_MS_check_range_validity__{sk}",
        )

    # 6) meta_helps_learn: MS_enumerate_carefully → 3 skills, weight 0.15.
    for sk in ("S_enumerate_outcomes", "S_count_favorable", "S_apply_addition_general"):
        _push(
            "meta_helps_learn",
            tail=["MS_enumerate_carefully"],
            head=[sk],
            weight=0.15,
            eid=f"MHL_MS_enumerate_carefully__{sk}",
        )

    # 7) meta_helps_learn: MS_distinguish_and_or → 3 skills, weight 0.15.
    for sk in (
        "S_compute_joint_independent",
        "S_apply_addition_mutually_exclusive",
        "S_apply_addition_general",
    ):
        _push(
            "meta_helps_learn",
            tail=["MS_distinguish_and_or"],
            head=[sk],
            weight=0.15,
            eid=f"MHL_MS_distinguish_and_or__{sk}",
        )

    # 8) error_signature: 4 extra edges per §2.4.
    for err, target in ERROR_SIGNATURE_EXTRA:
        _push(
            "error_signature",
            tail=[err],
            head=[target],
            weight=1.0,
            eid=f"ES_{err}__{target}",
        )

    JSON_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Added {added} new hyperedges. Total now: {len(edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
