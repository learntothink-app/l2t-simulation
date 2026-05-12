"""Operational metrics ``m(τ)`` from §VI.A and Eqs. (30)–(34).

Each function takes the list of :class:`StepRecord` rows (a trajectory) plus
any side data needed and returns a single scalar. The aggregate
:class:`TrajectoryMetrics` bundles all 12 components plus the behaviour tag
and the invariants_held dictionary so downstream analysis can slice it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .core_types import Action, BehaviourType, Observation, StepRecord


# ---------------------------------------------------------------------------
# Scalar metric functions.


def m_mastery(p_T: np.ndarray, p_0: np.ndarray) -> float:
    """Eq. (30): m_mastery = mean_k (p_{T,k} − p_{0,k})."""
    if p_T.size == 0:
        return 0.0
    return float(np.mean(p_T - p_0))


def m_transfer(
    records: Iterable[StepRecord], transfer_tasks: set[str]
) -> float:
    """Eq. (31): fraction of correct answers among transfer-task attempts."""
    obs = [r.observation for r in records if r.action.target_id in transfer_tasks and r.action.type in ("transfer", "task")]
    obs = [o for o in obs if o is not None]
    if not obs:
        return 0.0
    return float(np.mean([1.0 if o.correct else 0.0 for o in obs]))


def m_retention(transfer_now: float, transfer_after: float) -> float:
    """Eq. (32): m_retention(Δt) = m_transfer(t+Δt) − m_transfer(t)."""
    return float(transfer_after - transfer_now)


def m_hint(
    records: Iterable[StepRecord],
    hint_level_max: int,
    j_star: int,
    kappa: float = 0.5,
) -> float:
    """Eq. (33): m_hint = − E[HintLevel/HintLevelMax] − κ · E[SpoilerFlag].

    By construction m_hint ∈ [−(1+κ), 0]: zero hints ⇒ 0; the more / deeper
    the hints, the more negative the score.
    """
    hint_recs = [r for r in records if r.action.type == "hint"]
    if not hint_recs:
        return 0.0
    levels = np.array([r.action.hint_level for r in hint_recs], dtype=float)
    mean_level_ratio = float(np.mean(levels / max(hint_level_max, 1)))
    spoiler_flags = (levels >= j_star).astype(float)
    return float(-mean_level_ratio - kappa * float(np.mean(spoiler_flags)))


def m_robust(
    records: Iterable[StepRecord],
    transfer_tasks: set[str],
    q_low: float = 0.30,
) -> float:
    """Eq. (34): E[1{correct} | transfer ∧ q_t ≤ q_low]."""
    cands = [
        r
        for r in records
        if r.action.target_id in transfer_tasks
        and r.q_t <= q_low
        and r.action.type in ("transfer", "task")
    ]
    if not cands:
        return float("nan")
    return float(np.mean([1.0 if r.observation.correct else 0.0 for r in cands]))


def m_efficiency(p_T: np.ndarray, p_0: np.ndarray, T: int) -> float:
    """Mastery growth per unit time: m_efficiency = m_mastery / T."""
    if T <= 0:
        return 0.0
    return m_mastery(p_T, p_0) / float(T)


def m_meta(m_T: np.ndarray, m_0: np.ndarray) -> float:
    """Average meta-skill growth."""
    if m_T.size == 0:
        return 0.0
    return float(np.mean(m_T - m_0))


def m_engage(records: Iterable[StepRecord], total_seconds: float | None = None) -> float:
    """Share of "active" steps (any answer attempt) within the session."""
    rs = list(records)
    if not rs:
        return 0.0
    active = sum(1 for r in rs if r.action.type in ("task", "drill", "probe", "transfer"))
    return float(active) / float(len(rs))


def m_calib(records: Iterable[StepRecord]) -> float:
    """Expected calibration error (ECE) between estimated mastery on
    required skills (proxy for self-assessment) and observed success.

    For each evaluative action we read ``p_hat`` *before* the update (we
    don't have explicit self-assessment, so we use mean p̂ over required
    skills as the canonical proxy)."""

    pairs: list[tuple[float, float]] = []
    for r in records:
        if r.action.type not in ("task", "drill", "probe", "transfer"):
            continue
        if not r.p_hat:
            continue
        p_mean = float(np.mean(r.p_hat))
        success = 1.0 if r.observation.correct else 0.0
        pairs.append((p_mean, success))
    if not pairs:
        return 0.0
    # Bin into 10 equal-width buckets; ECE = Σ |bin_acc − bin_conf| · bin_freq.
    p = np.array([a for a, _ in pairs])
    y = np.array([b for _, b in pairs])
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = 0.0
    n = len(pairs)
    for b in range(10):
        mask = idx == b
        if not np.any(mask):
            continue
        bin_acc = float(y[mask].mean())
        bin_conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Aggregate for one trajectory.


@dataclass
class TrajectoryMetrics:
    methodology: str
    policy: str
    behaviour: BehaviourType
    student_id: int
    seed: int

    m_mastery: float
    m_transfer: float
    m_retention_1d: float
    m_retention_3d: float
    m_retention_7d: float
    m_retention_14d: float
    m_hint: float
    m_robust: float
    m_efficiency: float
    m_meta: float
    m_engage: float
    m_calib: float

    invariants_held: dict[str, bool] = field(default_factory=dict)
    n_steps: int = 0
