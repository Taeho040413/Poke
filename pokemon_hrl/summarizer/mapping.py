"""Map WorldState fields into StateSummary dimensions."""

from __future__ import annotations

from pokemon_hrl.planner.criteria import planner_goal_key
from pokemon_hrl.types import WorldState
from pokemon_hrl.world_state.serialization import badge_names

_MAX_FLAGS = 12
_MAX_MEMORIES = 5


def semantic_progression(state: WorldState) -> str:
    badges = badge_names(state.badges)
    flags = [k for k, v in state.flags.items() if v][:_MAX_FLAGS]
    parts = [
        f"map_id={state.map_id}",
        f"badges={','.join(badges) if badges else 'none'}",
    ]
    if flags:
        parts.append(f"events={','.join(flags)}")
    if state.planner_output is not None:
        parts.append(f"active_goal={planner_goal_key(state.planner_output)!r}")
    return "; ".join(parts)


def exploration_coverage(state: WorldState) -> str:
    visited = state.map_visited or [state.map_id]
    return (
        f"visited_maps={visited} "
        f"(count={len(visited)}); "
        f"position=({state.x},{state.y}) on map {state.map_id}"
    )


def interaction_outcome(state: WorldState) -> str:
    if not state.success_memory:
        return "no_recorded_interactions"
    recent = state.success_memory[-_MAX_MEMORIES:]
    chunks = [
        f"{entry.get('subgoal', entry.get('goal', '?'))}@step{entry.get('timestamp_step', '?')}"
        for entry in recent
    ]
    return "recent_successes: " + "; ".join(chunks)


def failure_cause(state: WorldState) -> str:
    if not state.failure_memory:
        return ""
    recent = state.failure_memory[-_MAX_MEMORIES:]
    chunks = [
        f"{entry.get('goal', '?')}:{entry.get('cause', '?')}"
        for entry in recent
    ]
    return "recent_failures: " + "; ".join(chunks)


def build_evidence(state: WorldState) -> dict:
    return {
        "map_id": state.map_id,
        "x": state.x,
        "y": state.y,
        "badges": badge_names(state.badges),
        "party_size": len(state.party),
        "bag_items": len(state.bag),
        "money": state.resources.get("money"),
        "map_visited_count": len(state.map_visited),
        "success_count": len(state.success_memory),
        "failure_count": len(state.failure_memory),
        "goal_stack": state.goal_stack,
    }
