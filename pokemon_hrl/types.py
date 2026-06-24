"""Shared types for Pokémon HRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    EXPLORATION = "exploration"
    INTERACTIVE = "interactive"
    GROWTH = "growth"


@dataclass
class Subgoal:
    """Memory-aligned subgoal with LLM-authored human descriptions for logging."""

    success_criteria: list[str]
    where: str = ""
    what: str = ""
    how: str = ""


@dataclass
class PlannerOutput:
    """Memory-aligned plan; goal is defined by success_criteria tokens."""

    subgoal: list[Subgoal]
    hint: dict[str, Any]
    success_criteria: list[str]
    failure_criteria: list[str]

    @property
    def target_map_id(self) -> int | None:
        raw = self.hint.get("target_map_id")
        if raw is None:
            return None
        return int(raw)


@dataclass
class ModeContext:
    mode: Mode
    planner: PlannerOutput
    current_map_id: int
    max_steps: int


@dataclass
class WorldState:
    map_id: int
    x: int
    y: int
    badges: int
    flags: dict[str, bool] = field(default_factory=dict)
    party: list[dict[str, Any]] = field(default_factory=list)
    bag: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)
    goal_stack: dict[str, Any] = field(default_factory=dict)
    failure_memory: list[dict[str, Any]] = field(default_factory=list)
    success_memory: list[dict[str, Any]] = field(default_factory=list)
    recent_summary: dict[str, Any] | None = None
    map_visited: list[int] = field(default_factory=list)
    planner_output: PlannerOutput | None = None
    global_step: int = 0


@dataclass
class StateSummary:
    semantic_progression: str = ""
    exploration_coverage: str = ""
    interaction_outcome: str = ""
    failure_cause: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    steps_used: int
    moved_tile: bool = False
    map_changed: bool = False
    tile_direction_blocked: bool = False


@dataclass
class FailedTileMoveResult:
    target_map: int
    target_x: int
    target_y: int
    retry_penalty: bool = False


@dataclass
class ProgressResult:
    success: bool = False
    failure: bool = False
    subgoal_success: bool = False
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    reason: str = ""
    subgoal: str = ""
    subgoal_index: int = -1
