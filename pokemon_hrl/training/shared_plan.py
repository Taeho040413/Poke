"""Process-wide shared planner output and subgoal progress for parallel training."""

from __future__ import annotations

from dataclasses import dataclass, field

from pokemon_hrl.types import PlannerOutput


@dataclass
class SharedPlanStore:
    """Single LLM/curriculum plan shared across all training envs."""

    _planner: PlannerOutput | None = field(default=None, repr=False)
    _subgoal_index: int = 0

    @property
    def planner(self) -> PlannerOutput | None:
        return self._planner

    @property
    def subgoal_index(self) -> int:
        return self._subgoal_index

    def set_planner(self, planner: PlannerOutput) -> None:
        self._planner = planner
        self._subgoal_index = 0

    def advance_subgoal_to(self, index: int) -> int:
        next_index = max(self._subgoal_index, int(index))
        self._subgoal_index = next_index
        return self._subgoal_index

    def reset_progress(self) -> None:
        self._subgoal_index = 0


_DEFAULT_STORE = SharedPlanStore()


def get_shared_plan_store() -> SharedPlanStore:
    return _DEFAULT_STORE
