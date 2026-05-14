"""Top-level simulation loop (§VII.D pseudocode).

Per student:

    initial belief b_0
    for t in 0..T-1:
        a_t = policy.select_action(b_t, ctx_t)
        o_t = env.observe(a_t)
        q_t = compute_reliability(a_t, o_t, ...)
        b_{t+1} = student_model.bayes_update(b_t, a_t, o_t, q_t)
        env.update_p_true(a_t, o_t)
    retention probes at Δt ∈ {1,3,7,14} days.

Parallelism is via :mod:`joblib`; the inner loop is sequential because each
step depends on the previous belief.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .core_types import Action, BehaviourType, Observation, StepRecord, BEHAVIOUR_TYPES
from .environment import SyntheticStudent
from .invariants import trajectory_satisfies_all
from .methodology import MethodologyData
from .metrics import (
    TrajectoryMetrics,
    m_calib,
    m_efficiency,
    m_engage,
    m_hint,
    m_mastery,
    m_meta,
    m_retention,
    m_robust,
    m_transfer,
)
from .policies import BKTPolicy, Context, L2TPolicy, PermissiveL2TPolicy, Policy, RandomPolicy
from .reliability import compute_reliability
from .student_model import BeliefState, StudentModel

log = logging.getLogger(__name__)

# Names → factory.
PolicyFactory = Callable[[MethodologyData, np.random.Generator], Policy]


# ---------------------------------------------------------------------------


def _baseline_time_for(action: Action, methodology: MethodologyData) -> float:
    md = methodology
    if action.target_id in md.tasks:
        return md.tasks[action.target_id].time_estimate_seconds
    if action.target_id in md.transfer_tasks:
        return md.transfer_tasks[action.target_id].time_estimate_seconds
    if action.target_id in md.drills:
        return md.drills[action.target_id].time_estimate_seconds
    if action.target_id in md.probes:
        return md.probes[action.target_id].time_estimate_seconds
    return 60.0


def _events_from_action(
    action: Action,
    obs: Observation,
    methodology: MethodologyData,
    context: Context,
) -> list[dict]:
    """Generate the past-LTL event records emitted by one step."""
    md = methodology
    events: list[dict] = []
    if action.type == "microtheory":
        mt = md.microtheories.get(action.target_id)
        if mt is not None:
            for c in mt.concepts:
                events.append({"event": "theory_checked", "concept": c})
                context.theory_checked.add(c)
    elif action.type == "task" or action.type == "transfer":
        task = md.tasks.get(action.target_id) or md.transfer_tasks.get(action.target_id)
        if task is not None:
            events.append(
                {"event": "task_presented", "task": task.id, "concept": task.required_concepts[0] if task.required_concepts else None}
            )
            if obs.correct:
                events.append({"event": "correct", "task": task.id})
                if action.type == "transfer":
                    events.append({"event": "transfer_passed", "task": task.id})
                    context.transfer_passed = True
                else:
                    context.completed_tasks.add(task.id)
            else:
                events.append({"event": "incorrect", "task": task.id})
            if obs.hint_requested:
                events.append({"event": "hint_requested", "task": task.id})
    elif action.type == "hint":
        events.append(
            {"event": "hint_delivered", "task": action.target_id, "hint_level": action.hint_level}
        )
    elif action.type == "drill":
        events.append({"event": "intent_received"})
        events.append({"event": "step_matched"})
        if obs.hint_requested:
            events.append({"event": "hint_requested", "task": action.target_id})
    elif action.type == "probe":
        events.append({"event": "intent_received"})
        if obs.hint_requested:
            events.append({"event": "hint_requested", "task": action.target_id})
    return events


# ---------------------------------------------------------------------------


def make_policy(name: str, methodology: MethodologyData, rng: np.random.Generator) -> Policy:
    if name == "B1_random":
        return RandomPolicy(methodology.hypergraph, methodology, rng)
    if name == "B2_bkt":
        return BKTPolicy(methodology.hypergraph, methodology, rng)
    if name == "B3_l2t_no_q":
        return L2TPolicy(methodology.hypergraph, methodology, rng, use_reliability=False)
    if name == "B4_full_l2t":
        return L2TPolicy(methodology.hypergraph, methodology, rng, use_reliability=True)
    if name == "B3p_l2t_permissive":
        return PermissiveL2TPolicy(methodology.hypergraph, methodology, rng, use_reliability=False)
    raise ValueError(f"unknown policy: {name}")


POLICY_USES_RELIABILITY = {
    "B1_random": False,
    "B2_bkt": False,
    "B3_l2t_no_q": False,
    "B4_full_l2t": True,
    "B3p_l2t_permissive": False,
}


# ---------------------------------------------------------------------------


def run_single_student(
    policy_name: str,
    methodology: MethodologyData,
    behaviour: BehaviourType,
    student_seed: int,
    student_id: int,
    t_steps: int,
    prereq_perturbation: bool,
    retention_days: Sequence[int],
) -> TrajectoryMetrics:
    """Run one student trajectory, return the per-trajectory metric vector."""

    rng_env = np.random.default_rng(student_seed)
    rng_policy = np.random.default_rng(student_seed + 1)

    student = SyntheticStudent.sample(
        methodology=methodology,
        behaviour=behaviour,
        rng=rng_env,
        prereq_perturbation=prereq_perturbation,
    )
    p_true_0 = student.p_true.copy()
    m_true_0 = student.m_true.copy()

    policy = make_policy(policy_name, methodology, rng_policy)
    use_reliability = POLICY_USES_RELIABILITY[policy_name]
    model = StudentModel(methodology.hypergraph, methodology, use_reliability=use_reliability)
    belief = model.initial_belief()
    p_hat_0 = belief.p_hat.copy()

    context = Context(t=0, methodology=methodology)
    records: list[StepRecord] = []
    events_log: list[dict] = []

    transfer_ids: set[str] = set(methodology.transfer_tasks.keys())

    for t in range(t_steps):
        context.t = t
        action = policy.select_action(belief, context)
        observation = student.observe(action)
        baseline = _baseline_time_for(action, methodology)
        q_t = compute_reliability(action, observation, baseline)

        # Update model + environment.
        new_belief = model.bayes_update(belief, action, observation, q_t)
        student.update_p_true(action, observation)
        if isinstance(policy, BKTPolicy):
            policy.observe_outcome(action, observation)

        # Generate past-LTL event stream.
        step_events = _events_from_action(action, observation, methodology, context)
        events_log.extend(step_events)
        context.last_observation = observation
        if action.type in ("task", "drill", "probe", "transfer"):
            context.recent_attempts[action.target_id] = context.recent_attempts.get(action.target_id, 0) + 1
            context.current_item = action.target_id
        if action.type == "hint":
            context.hints_used_per_task[action.target_id] = max(
                context.hints_used_per_task.get(action.target_id, 0), action.hint_level
            )

        records.append(
            StepRecord(
                t=t,
                action=action,
                observation=observation,
                q_t=q_t,
                p_hat=belief.p_hat.tolist(),  # belief BEFORE the update, for m_calib
                p_true=student.p_true.tolist(),
                fsm_state="",  # FSM kept implicit; events_log is the actual trace
                events=[e["event"] for e in step_events],
            )
        )
        belief = new_belief

    # Terminal transfer & retention probes ---------------------------------
    # To make m_transfer comparable across policies (BKT never picks transfer
    # items by construction), every student takes a fixed transfer battery at
    # the end of the main loop. Prefer the methodology's holdout set so that
    # the battery measures transfer to *novel* items rather than ones the
    # policy already drilled in-loop; fall back to the first 8 transfer tasks
    # if no holdout is configured for the methodology.
    if methodology.holdout_transfer_ids:
        sample_ids = sorted(methodology.holdout_transfer_ids)
    else:
        sample_ids = list(methodology.transfer_tasks.keys())[:8]

    def _probe(delta_days: float) -> tuple[float, list[StepRecord]]:
        orig_p = student.p_true.copy()
        orig_off = student._retention_offset_days
        student.advance_time(delta_days)
        buf: list[StepRecord] = []
        for tid in sample_ids:
            a = Action(type="transfer", target_id=tid)
            o = student.observe(a)
            buf.append(
                StepRecord(
                    t=-1,
                    action=a,
                    observation=o,
                    q_t=1.0,
                    p_hat=[],
                    p_true=[],
                    fsm_state="",
                    events=[],
                )
            )
        # restore
        student.p_true = orig_p
        student._retention_offset_days = orig_off
        return m_transfer(buf, transfer_ids), buf

    terminal_m_transfer, _ = _probe(0.0)
    retention = {}
    for dt in retention_days:
        m_t_delayed, _ = _probe(float(dt))
        retention[dt] = m_retention(terminal_m_transfer, m_t_delayed)

    # Invariants ------------------------------------------------------------
    task_concepts = {
        tid: list(t.required_concepts) for tid, t in methodology.tasks.items()
    }
    task_concepts.update({
        tid: list(t.required_concepts) for tid, t in methodology.transfer_tasks.items()
    })
    invariants_held = trajectory_satisfies_all(events_log, task_concepts, j_star=methodology.j_star_spoiler_level)

    # Aggregate metrics -----------------------------------------------------
    metrics = TrajectoryMetrics(
        methodology=methodology.name,
        policy=policy_name,
        behaviour=behaviour,
        student_id=student_id,
        seed=student_seed,
        m_mastery=m_mastery(student.p_true, p_true_0),
        m_transfer=terminal_m_transfer,
        m_retention_1d=retention.get(1, 0.0),
        m_retention_3d=retention.get(3, 0.0),
        m_retention_7d=retention.get(7, 0.0),
        m_retention_14d=retention.get(14, 0.0),
        m_hint=m_hint(records, methodology.hint_ladder_max, methodology.j_star_spoiler_level),
        m_robust=m_robust(records, transfer_ids),
        m_efficiency=m_efficiency(student.p_true, p_true_0, t_steps),
        m_meta=m_meta(student.m_true, m_true_0),
        m_engage=m_engage(records),
        m_calib=m_calib(records),
        invariants_held=invariants_held,
        n_steps=len(records),
    )
    return metrics


# ---------------------------------------------------------------------------


def sample_behaviour(
    rng: np.random.Generator, proportions: dict[str, float]
) -> BehaviourType:
    names = list(proportions.keys())
    probs = np.array([proportions[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    idx = rng.choice(len(names), p=probs)
    return names[idx]  # type: ignore[return-value]


def run_simulation(
    policy_name: str,
    methodology: MethodologyData,
    n_students: int = 1000,
    t_steps: int = 100,
    master_seed: int = 42,
    behaviour_proportions: dict[str, float] | None = None,
    retention_days: Sequence[int] = (1, 3, 7, 14),
    prereq_perturbation_eps: float = 0.10,
    n_jobs: int = 1,
) -> list[TrajectoryMetrics]:
    """Run ``n_students`` trajectories under one policy/methodology pair.

    ``master_seed`` is the *single* root of randomness; per-student seeds
    are spawned via :func:`np.random.SeedSequence` so the run is fully
    reproducible.
    """

    proportions = behaviour_proportions or {
        "honest": 0.60,
        "guesser": 0.15,
        "copier": 0.10,
        "help_seeker": 0.15,
    }
    seed_seq = np.random.SeedSequence(master_seed)
    child_seeds = seed_seq.spawn(n_students)

    # Sample behaviour up front so the proportions are exact-ish.
    selector_rng = np.random.default_rng(master_seed + 1_000)
    behaviours = [sample_behaviour(selector_rng, proportions) for _ in range(n_students)]
    perturb_flags = [bool(selector_rng.random() < prereq_perturbation_eps) for _ in range(n_students)]

    def _one(i: int) -> TrajectoryMetrics:
        student_seed = int(child_seeds[i].generate_state(1)[0])
        return run_single_student(
            policy_name=policy_name,
            methodology=methodology,
            behaviour=behaviours[i],
            student_seed=student_seed,
            student_id=i,
            t_steps=t_steps,
            prereq_perturbation=perturb_flags[i],
            retention_days=retention_days,
        )

    if n_jobs == 1 or n_students < 4:
        return [_one(i) for i in range(n_students)]

    # Optional joblib parallelism.
    try:
        from joblib import Parallel, delayed
    except ModuleNotFoundError:
        log.warning("joblib not installed; falling back to serial execution")
        return [_one(i) for i in range(n_students)]
    return list(
        Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one)(i) for i in range(n_students))
    )
