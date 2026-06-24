"""Serialize WorldState / StateSummary for LLM prompts."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from pokemon_hrl.types import StateSummary, WorldState

# Long-term HRL objective (also in curriculum header).
FINAL_GOAL_TEXT = (
    "Clear the 3rd gym leader (Lt. Surge, Vermilion City) and reach Lavender Town."
)

_BADGE_NAMES = (
    "boulder",
    "cascade",
    "thunder",
    "rainbow",
    "soul",
    "marsh",
    "volcano",
    "earth",
)


def badge_names(bitmask: int) -> list[str]:
    names = []
    for i, name in enumerate(_BADGE_NAMES):
        if bitmask & (1 << i):
            names.append(name)
    return names


def world_state_to_dict(state: WorldState) -> dict[str, Any]:
    return {
        "map_id": state.map_id,
        "position": {"x": state.x, "y": state.y},
        "badges": badge_names(state.badges),
        "badge_count": len(badge_names(state.badges)),
        "party": state.party,
        "bag": state.bag,
        "resources": state.resources,
        "active_flags": sorted(k for k, v in state.flags.items() if v),
        "map_visited": state.map_visited,
        "goal_stack": state.goal_stack,
        "success_memory": state.success_memory[-8:],
        "failure_memory": state.failure_memory[-8:],
        "planner_output": asdict(state.planner_output) if state.planner_output else None,
        "global_step": state.global_step,
    }


def state_summary_to_dict(summary: StateSummary) -> dict[str, Any]:
    return asdict(summary)


def world_state_from_dict(raw: str | dict[str, Any]) -> WorldState:
    from pokemon_hrl.planner.validation import parse_planner_dict

    payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    planner_raw = payload.pop("planner_output", None)
    planner = parse_planner_dict(planner_raw) if planner_raw else None
    state = WorldState(**{k: v for k, v in payload.items() if k in WorldState.__dataclass_fields__})
    state.planner_output = planner
    return state
