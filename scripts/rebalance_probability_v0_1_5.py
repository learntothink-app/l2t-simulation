"""One-shot rebalance of probability_basics.json per v0.1.5 spec.

Two coupled content changes:

(1) Trim required_skills on the 25 existing tasks down to one *primary*
    skill (or sometimes two for genuinely-equal pairs). The previous
    multi-skill annotations conflated "what the solver must know" with
    "what the task primarily trains", which gave a handful of basic
    skills 88% task coverage and broke L2T's adaptive scheduling.

(2) Add 12 new tasks targeting the under-represented skills
    (count_favorable, compute_simple_probability, recognize_independence,
    apply_bayes_simple, compute_expected_value_discrete,
    distinguish_probability_frequency) so every skill is trained by at
    least 3 tasks.

Hyperedges for the new tasks (trains + trains_metaskill) and 3 new
probes are also added. Idempotent: re-runs only append missing edges.

Run once:

    python scripts/rebalance_probability_v0_1_5.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "methodologies" / "probability_basics.json"


# ---------------------------------------------------------------------------
# §2.2 — required_skills trims (existing 25 tasks).

TRIMS: dict[str, list[str]] = {
    # Group 1 — basic enumeration
    "T_basic_2coins_at_least_one_H": ["S_enumerate_outcomes"],
    "T_basic_die_even": ["S_enumerate_outcomes"],
    "T_basic_2dice_sum7": ["S_enumerate_outcomes"],
    "T_basic_bag_3red_2blue": ["S_count_favorable"],
    "T_basic_card_heart": ["S_compute_simple_probability"],
    # Group 2 — complement
    "T_complement_die_not_one": ["S_apply_complement_rule"],
    "T_complement_2coins_not_both_T": ["S_apply_complement_rule"],
    "T_complement_3dice_at_least_one_six": ["S_apply_complement_rule"],
    # Group 3 — independence and joint
    "T_indep_2coins_HH": ["S_compute_joint_independent"],
    "T_indep_die_then_coin": ["S_recognize_independence"],
    "T_indep_2bags_both_red": ["S_compute_joint_independent"],
    "T_indep_3coins_all_H": ["S_compute_joint_independent"],
    # Group 4 — mutually exclusive
    "T_mutex_die_1_or_2": ["S_apply_addition_mutually_exclusive"],
    "T_mutex_bag_red_or_blue": ["S_apply_addition_mutually_exclusive"],
    "T_mutex_card_ace_or_king": ["S_apply_addition_mutually_exclusive"],
    # Group 5 — general addition
    "T_general_card_red_or_face": ["S_apply_addition_general"],
    "T_general_die_even_or_high": ["S_apply_addition_general"],
    "T_general_2dice_first1_or_second6": ["S_apply_addition_general"],
    # Group 6 — conditional
    "T_cond_card_heart_given_red": ["S_compute_conditional"],
    "T_cond_die_six_given_even": ["S_compute_conditional"],
    "T_cond_2dice_sum_given_first": ["S_compute_conditional"],
    "T_cond_bag_no_replacement": ["S_compute_conditional"],
    # Group 7 — bayes
    "T_bayes_medical_test_basic": ["S_apply_bayes_simple"],
    "T_bayes_two_coins_one_biased": ["S_apply_bayes_simple"],
    # Group 8 — expected value
    "T_ev_die_payout": ["S_compute_expected_value_discrete"],
}


# ---------------------------------------------------------------------------
# §2.3 / §2.4 — new tasks (12 entries). Each gets a single primary skill.

# Whether a task goes to scaffolding_tasks (groups 1-2) or target_tasks
# (groups 3-8). count_* and simple_* are scaffolding; the rest are target.

NEW_SCAFFOLDING_TASKS: list[dict] = [
    {
        "id": "T_count_card_face",
        "title": "Counting face cards",
        "statement": "From a standard 52-card deck, how many cards are face cards (Jack, Queen, King)?",
        "required_skills": ["S_count_favorable"],
        "required_concepts": ["C_event"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["8", "12", "16", "20"],
        "correct_index": 1,
        "time_estimate_minutes": 2,
    },
    {
        "id": "T_count_committee",
        "title": "Counting in a committee",
        "statement": "A class of 30 students has 18 girls and 12 boys. The committee chair must be a girl. How many girls could be chosen?",
        "required_skills": ["S_count_favorable"],
        "required_concepts": ["C_event"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["12", "18", "30", "48"],
        "correct_index": 1,
        "time_estimate_minutes": 2,
    },
    {
        "id": "T_simple_dice_one_or_six",
        "title": "Die: probability of 1 or 6",
        "statement": "You roll a fair six-sided die. What is the probability of rolling a 1 or a 6?",
        "required_skills": ["S_compute_simple_probability"],
        "required_concepts": ["C_classical_probability"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["1/6", "1/3", "1/2", "2/3"],
        "correct_index": 1,
        "time_estimate_minutes": 2,
    },
    {
        "id": "T_simple_lottery_pick",
        "title": "Small lottery: probability of winning",
        "statement": "A lottery sells 100 tickets, 5 of which are winners. You buy one ticket. What is the probability you win?",
        "required_skills": ["S_compute_simple_probability"],
        "required_concepts": ["C_classical_probability"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["1/100", "1/20", "1/10", "5/95"],
        "correct_index": 1,
        "time_estimate_minutes": 2,
    },
]

NEW_TARGET_TASKS: list[dict] = [
    {
        "id": "T_indep_card_then_coin",
        "title": "Independence: card draw then coin flip",
        "statement": "You draw a card from a deck (then return it), then flip a fair coin. Are the two outcomes independent events?",
        "required_skills": ["S_recognize_independence"],
        "required_concepts": ["C_independence"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["Yes — they are separate experiments", "No — they share the same outcome space", "Only if the card is red", "Cannot determine without more info"],
        "correct_index": 0,
        "time_estimate_minutes": 2,
    },
    {
        "id": "T_indep_with_replacement_vs_not",
        "title": "Independence: replacement matters",
        "statement": "You draw two cards from a 52-card deck. Which scenario makes the two draws independent?",
        "required_skills": ["S_recognize_independence"],
        "required_concepts": ["C_independence"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["With replacement", "Without replacement", "Both equally", "Neither — drawing cards is always dependent"],
        "correct_index": 0,
        "time_estimate_minutes": 2,
    },
    {
        "id": "T_bayes_spam_filter_basic",
        "title": "Spam filter: P(spam | flagged)",
        "statement": "A spam filter flags 90% of true spam and 5% of legitimate emails. 30% of incoming emails are spam. If an email is flagged, what is the approximate probability it is actually spam?",
        "required_skills": ["S_apply_bayes_simple"],
        "required_concepts": ["C_conditional_probability"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["0.30", "0.50", "0.89", "0.95"],
        "correct_index": 2,
        "time_estimate_minutes": 4,
    },
    {
        "id": "T_ev_coin_game",
        "title": "Three coin flips: expected number of heads",
        "statement": "You flip a fair coin three times. What is the expected number of heads?",
        "required_skills": ["S_compute_expected_value_discrete"],
        "required_concepts": ["C_expected_value"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["0", "1", "1.5", "3"],
        "correct_index": 2,
        "time_estimate_minutes": 3,
    },
    {
        "id": "T_ev_lottery_value",
        "title": "Lottery: expected payoff",
        "statement": "A lottery ticket: with probability 1/100 you win $1000, with probability 99/100 you win nothing. What is the expected payoff per ticket?",
        "required_skills": ["S_compute_expected_value_discrete"],
        "required_concepts": ["C_expected_value"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["$1", "$10", "$100", "$1000"],
        "correct_index": 1,
        "time_estimate_minutes": 3,
    },
    {
        "id": "T_distinguish_coin_10_trials",
        "title": "Coin observed frequency vs probability",
        "statement": "After 10 coin flips you observed 7 heads. What is the true theoretical probability of heads on the next flip (assuming the coin is fair)?",
        "required_skills": ["S_distinguish_probability_frequency"],
        "required_concepts": ["C_classical_probability"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["0.5", "0.7", "Cannot say — depends on history", "0.3"],
        "correct_index": 0,
        "time_estimate_minutes": 3,
    },
    {
        "id": "T_distinguish_dice_small_sample",
        "title": "Small sample of die rolls",
        "statement": "You roll a fair six-sided die 5 times and observe a 6 twice. Does this prove the die is unfair?",
        "required_skills": ["S_distinguish_probability_frequency"],
        "required_concepts": ["C_classical_probability"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["Yes — 2/5 ≠ 1/6", "Yes — the expected count is below 1", "No — the sample is too small to conclude", "Yes — P(6) is now 2/5"],
        "correct_index": 2,
        "time_estimate_minutes": 3,
    },
    {
        "id": "T_distinguish_LLN",
        "title": "Law of large numbers",
        "statement": "You flip a fair coin 1000 times vs 10 times and compute the observed frequency of heads. Which sequence is more likely to give a frequency closer to 0.5?",
        "required_skills": ["S_distinguish_probability_frequency"],
        "required_concepts": ["C_classical_probability"],
        "answer_format": "multiple_choice",
        "n_options": 4,
        "options": ["10 flips", "1000 flips", "Both equally", "Neither — coin flips are random"],
        "correct_index": 1,
        "time_estimate_minutes": 3,
    },
]


# ---------------------------------------------------------------------------
# §2.5 — hyperedges for new tasks. The L2T-friendly side conditions:
#   - every new task gets a "trains" edge to its primary skill (already
#     auto-created by the loader via per-task required_skills, but we also
#     add explicit "trains" entries so the propagation operator picks up
#     the trains-typed weight on a separate channel from "requires").
#   - every new task gets a trains_metaskill edge to MS_check_range_validity.
#   - enumeration-flavoured tasks → MS_enumerate_carefully.
#   - and/or-flavoured tasks → MS_distinguish_and_or.

NEW_ENUM_TASKS = {
    "T_count_card_face",
    "T_count_committee",
    "T_simple_lottery_pick",
}
NEW_ANDOR_TASKS = {"T_simple_dice_one_or_six"}


# ---------------------------------------------------------------------------
# §2.6 — probes for the under-covered skills.

NEW_PROBES: list[dict] = [
    {
        "id": "P_distinguish_freq",
        "title": "Frequency vs probability probe",
        "diagnoses_skills": ["S_distinguish_probability_frequency"],
        "diagnoses_concepts": ["C_classical_probability"],
    },
    {
        "id": "P_ev",
        "title": "Expected-value probe",
        "diagnoses_skills": ["S_compute_expected_value_discrete"],
        "diagnoses_concepts": ["C_expected_value"],
    },
]


# ---------------------------------------------------------------------------


def _existing_triples(edges: list[dict]) -> set[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    out = set()
    for e in edges:
        out.add((e["kind"], tuple(e["tail"]), tuple(e["head"])))
    return out


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    # --- step 1: trim required_skills on existing tasks ------------------
    trimmed = 0
    for section in ("scaffolding_tasks", "target_tasks"):
        for t in data.get(section, []):
            if t["id"] in TRIMS and t.get("required_skills") != TRIMS[t["id"]]:
                t["required_skills"] = TRIMS[t["id"]]
                trimmed += 1

    # --- step 2: append new tasks ---------------------------------------
    existing_task_ids = {t["id"] for t in data.get("scaffolding_tasks", [])} | {
        t["id"] for t in data.get("target_tasks", [])
    }
    appended = 0
    for t in NEW_SCAFFOLDING_TASKS:
        if t["id"] not in existing_task_ids:
            data.setdefault("scaffolding_tasks", []).append(t)
            existing_task_ids.add(t["id"])
            appended += 1
    for t in NEW_TARGET_TASKS:
        if t["id"] not in existing_task_ids:
            data.setdefault("target_tasks", []).append(t)
            existing_task_ids.add(t["id"])
            appended += 1

    # --- step 3: append new probes --------------------------------------
    existing_probe_ids = {p["id"] for p in data.get("probes", [])}
    probes_added = 0
    for p in NEW_PROBES:
        if p["id"] not in existing_probe_ids:
            data.setdefault("probes", []).append(p)
            existing_probe_ids.add(p["id"])
            probes_added += 1

    # --- step 4: append new hyperedges for new tasks --------------------
    edges = data["hypergraph"]["hyperedges"]
    existing = _existing_triples(edges)
    edges_added = 0

    def _push(kind: str, tail: list[str], head: list[str], weight: float, eid: str) -> None:
        nonlocal edges_added
        key = (kind, tuple(tail), tuple(head))
        if key in existing:
            return
        edges.append({"id": eid, "kind": kind, "tail": tail, "head": head, "weight": weight})
        existing.add(key)
        edges_added += 1

    all_new_tasks = NEW_SCAFFOLDING_TASKS + NEW_TARGET_TASKS
    for t in all_new_tasks:
        tid = t["id"]
        # explicit trains edge to primary skill
        for sk in t["required_skills"]:
            _push("trains", [tid], [sk], 1.0, f"TRN_{tid}__{sk}")
        # universal trains_metaskill
        _push(
            "trains_metaskill",
            [tid],
            ["MS_check_range_validity"],
            0.5,
            f"TRM_{tid}__MS_check_range_validity",
        )
        if tid in NEW_ENUM_TASKS:
            _push(
                "trains_metaskill",
                [tid],
                ["MS_enumerate_carefully"],
                0.5,
                f"TRM_{tid}__MS_enumerate_carefully",
            )
        if tid in NEW_ANDOR_TASKS:
            _push(
                "trains_metaskill",
                [tid],
                ["MS_distinguish_and_or"],
                0.5,
                f"TRM_{tid}__MS_distinguish_and_or",
            )

    JSON_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Trimmed {trimmed} existing tasks. "
        f"Appended {appended} new tasks, {probes_added} probes, {edges_added} edges."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
