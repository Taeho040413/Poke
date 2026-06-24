"""Subgoal helpers — resolve planner-provided success criteria."""

from __future__ import annotations

from pokemon_hrl.types import Subgoal


def resolve_subgoal_criteria(
    subgoal: Subgoal,
    *,
    target_map_id: int | None = None,
) -> list[str]:
    del target_map_id
    return list(subgoal.success_criteria)


def current_subgoal(planner_subgoals: list[Subgoal], index: int) -> Subgoal | None:
    if not planner_subgoals or index < 0 or index >= len(planner_subgoals):
        return None
    return planner_subgoals[index]
