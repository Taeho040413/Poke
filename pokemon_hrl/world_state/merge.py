"""Merge extracted game snapshot with persistent DB fields."""

from __future__ import annotations

from pokemon_hrl.types import WorldState


def merge_extracted_state(previous: WorldState, extracted: WorldState) -> WorldState:
    """Keep planner memories when refreshing coordinates/party from the emulator."""
    return WorldState(
        map_id=extracted.map_id,
        x=extracted.x,
        y=extracted.y,
        badges=extracted.badges,
        flags=extracted.flags,
        party=extracted.party,
        bag=extracted.bag,
        resources=extracted.resources,
        goal_stack=previous.goal_stack,
        failure_memory=list(previous.failure_memory),
        success_memory=list(previous.success_memory),
        recent_summary=previous.recent_summary,
        map_visited=extracted.map_visited,
        planner_output=previous.planner_output,
        global_step=extracted.global_step,
    )
