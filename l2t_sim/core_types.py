"""Shared dataclasses to avoid circular imports between policies/environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ActionType = Literal[
    "probe", "drill", "microtheory", "task", "hint", "transfer", "retention"
]
ACTION_TYPES: tuple[str, ...] = (
    "probe",
    "drill",
    "microtheory",
    "task",
    "hint",
    "transfer",
    "retention",
)

BehaviourType = Literal["honest", "guesser", "copier", "help_seeker"]
BEHAVIOUR_TYPES: tuple[str, ...] = ("honest", "guesser", "copier", "help_seeker")


@dataclass(frozen=True)
class Action:
    """Pedagogical action a_t ∈ A_safe (Eq. 27 of the paper)."""

    type: ActionType
    target_id: str
    hint_level: int = 0


@dataclass(frozen=True)
class Observation:
    """Observation o_t logged at every step (§II.A; §III.B)."""

    correct: bool
    answer_format: str
    time_seconds: float
    hint_requested: bool
    n_hints_used: int
    hint_level_reached: int
    self_explanation: bool
    copying_pattern: bool
    error_type: str | None = None


@dataclass
class StepRecord:
    """One row of the trajectory τ."""

    t: int
    action: Action
    observation: Observation
    q_t: float
    p_hat: list[float]
    p_true: list[float]
    fsm_state: str
    events: list[str] = field(default_factory=list)
