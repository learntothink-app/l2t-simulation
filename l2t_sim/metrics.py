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


def m_selection_robust(
    records: Iterable[StepRecord],
    transfer_tasks: set[str],
    q_low: float = 0.50,
) -> float:
    """[v0.2.0] Eq. (34): selection-robust correctness — fraction of
    correct answers on the transfer attempts the policy chose to make
    under low-reliability conditions (q_t ≤ q_low).

    Renamed from ``m_robust`` because the metric depends on which
    transfer items the policy selects, not on inference quality alone.
    A policy that *correctly avoids* transfer probes under noise has
    fewer eligible attempts and is biased here. Use
    :func:`m_inference_robust` for a selection-free measure.

    ``q_low`` was raised from 0.30 to 0.50 in v0.1.1 so guesser
    learners (typical q_t ≈ 0.40) also contribute to the subsample;
    at 0.30 only copiers crossed the threshold.
    """
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


# Backward-compatibility alias: paper v0.1.6 cited `m_robust` for what is
# now `m_selection_robust`. Code that still uses the old name continues
# to compute the same quantity.
m_robust = m_selection_robust


def m_inference_robust(
    final_p_hat: np.ndarray,
    final_p_true: np.ndarray,
    behaviour: str,
) -> float:
    """[v0.2.0] Inference-robust belief quality — mean absolute error
    between the policy's final per-skill belief ``b_T`` and ground-truth
    mastery ``p_true_T``, restricted to *contaminated* students
    (``behaviour ∈ {"guesser", "copier"}``).

    Lower is better. Defined for all contaminated students regardless
    of what the policy chose to probe — unlike
    :func:`m_selection_robust`, this isolates the *inference* component
    from selection.

    Returns ``NaN`` for honest / help_seeker students (the metric is
    only meaningful when the observation stream carries contamination).
    """
    if behaviour not in ("guesser", "copier"):
        return float("nan")
    if final_p_hat.size == 0:
        return float("nan")
    return float(np.mean(np.abs(final_p_hat - final_p_true)))


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
    m_robust: float  # alias of m_selection_robust (paper v0.1.6 name)
    m_efficiency: float
    m_meta: float
    m_engage: float
    m_calib: float

    # v0.2.0: clean inference-quality metric on contaminated students
    # (NaN for honest/help_seeker; see m_inference_robust in metrics.py).
    m_inference_robust: float = float("nan")

    invariants_held: dict[str, bool] = field(default_factory=dict)
    n_steps: int = 0
