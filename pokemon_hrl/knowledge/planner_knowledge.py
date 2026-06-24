"""Build chapter-scoped knowledge context for the LLM planner."""

from __future__ import annotations

from typing import Any

from pokemon_hrl.knowledge.red_maps import (
    MapIds,
    get_maps_for_names,
    map_id_to_name,
)
from pokemon_hrl.knowledge.red_story_facts import CHAPTER_FACTS
from pokemon_hrl.types import WorldState

_DEFAULT_FALLBACK_MAPS = (
    "PALLET_TOWN",
    "VIRIDIAN_CITY",
    "OAKS_LAB",
    "VIRIDIAN_MART",
    "ROUTE_1",
)


def _chapter_goal_key(chapter_goal: dict[str, Any]) -> str | None:
    criteria = chapter_goal.get("suggested_goal_success_criteria") or []
    if not criteria:
        return None
    return str(criteria[0]).strip()


def _current_map_id(world_state: dict[str, Any], planning_context: dict[str, Any]) -> int:
    if "map_id" in world_state:
        return int(world_state["map_id"])
    return int(planning_context.get("current_map_id", 0))


def build_planner_knowledge(
    chapter_goal: dict[str, Any],
    world_state: dict[str, Any],
    planning_context: dict[str, Any],
) -> dict[str, Any]:
    """Chapter-scoped map subset and story facts for planner prompts."""
    current_map_id = _current_map_id(world_state, planning_context)
    current_map_name = map_id_to_name(current_map_id)
    goal_key = _chapter_goal_key(chapter_goal)
    chapter_facts = CHAPTER_FACTS.get(goal_key) if goal_key else None

    if chapter_facts:
        route_names = list(chapter_facts.get("required_route") or [])
        map_knowledge = get_maps_for_names(route_names)
        if current_map_name and current_map_name not in map_knowledge:
            map_knowledge[current_map_name] = current_map_id
    else:
        map_knowledge = get_maps_for_names(list(_DEFAULT_FALLBACK_MAPS))
        if current_map_name:
            map_knowledge[current_map_name] = current_map_id

    payload: dict[str, Any] = {
        "current_map": {
            "id": current_map_id,
            "name": current_map_name,
        },
        "map_knowledge": map_knowledge,
        "chapter_facts": chapter_facts,
    }
    if chapter_facts:
        payload["active_chapter_fact_label"] = chapter_facts.get("label")
        payload["required_target_map_id"] = chapter_facts.get("required_target_map_id")
    return payload


def build_planner_knowledge_from_state(state: WorldState) -> dict[str, Any]:
    """Convenience wrapper using live WorldState."""
    from pokemon_hrl.planner.progression import (
        chapter_goal_payload,
        planning_context_payload,
    )
    from pokemon_hrl.world_state.serialization import world_state_to_dict

    chapter_goal = chapter_goal_payload(state)
    planning_context = planning_context_payload(state)
    return build_planner_knowledge(
        chapter_goal,
        world_state_to_dict(state),
        planning_context,
    )


def next_route_map_id(
    planner_knowledge: dict[str, Any],
    *,
    current_map_id: int,
) -> int | None:
    """First route map after current position, or required target if already past route."""
    chapter_facts = planner_knowledge.get("chapter_facts") or {}
    required_target = chapter_facts.get("required_target_map_id")
    if required_target is not None and int(current_map_id) == int(required_target):
        return int(required_target)

    route_names = list(chapter_facts.get("required_route") or [])
    route_ids = [planner_knowledge["map_knowledge"][name] for name in route_names if name in planner_knowledge.get("map_knowledge", {})]
    if not route_ids:
        return int(required_target) if required_target is not None else None

    try:
        current_index = route_ids.index(int(current_map_id))
    except ValueError:
        return route_ids[0] if route_ids else None

    if current_index + 1 < len(route_ids):
        return route_ids[current_index + 1]
    return int(required_target) if required_target is not None else route_ids[-1]


__all__ = [
    "MapIds",
    "build_planner_knowledge",
    "build_planner_knowledge_from_state",
    "next_route_map_id",
]
