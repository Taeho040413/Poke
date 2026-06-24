"""Memory-aligned goal/subgoal criteria encoding for planner and agent obs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

from pokemonred_puffer.data.events import EVENTS_FLAGS_LENGTH, EventFlagsBits

if TYPE_CHECKING:
    from pokemon_hrl.types import PlannerOutput, Subgoal

EVENT_FLAG_DIM = EVENTS_FLAGS_LENGTH
STAT_TARGET_NAMES = (
    "first_npc_talk_count",
    "first_object_interaction_count",
    "new_npc_textbox_count",
    "item_count",
    "trainer_battle_win_count",
    "wild_battle_win_count",
    "pokecenter_heal_hp_count",
)
STAT_TARGET_DIM = len(STAT_TARGET_NAMES)
MAX_SUBGOAL_INDEX = 8
MAP_TARGET_EMPTY = 0

_EVENT_FIELD_INDEX = {
    name: idx for idx, (name, _, _) in enumerate(EventFlagsBits._fields_)
}


def criteria_label(criteria: list[str]) -> str:
    return "|".join(token.strip() for token in criteria if token.strip())


def planner_goal_key(planner: PlannerOutput) -> str:
    return criteria_label(planner.success_criteria)


def subgoal_label(subgoal: Subgoal) -> str:
    return criteria_label(subgoal.success_criteria)


def planner_signature(planner: PlannerOutput) -> tuple[Any, ...]:
    return (
        tuple(planner.success_criteria),
        tuple(tuple(sg.success_criteria) for sg in planner.subgoal),
        int(planner.target_map_id or 0),
        tuple(planner.failure_criteria),
    )


def encode_event_target(criteria: list[str]) -> np.ndarray:
    target = np.zeros(EVENT_FLAG_DIM, dtype=np.uint8)
    for token in criteria:
        raw = token.strip()
        if not raw.startswith("flag:"):
            continue
        flag_name = raw.split(":", 1)[1]
        idx = _EVENT_FIELD_INDEX.get(flag_name)
        if idx is None:
            continue
        byte_idx = idx // 8
        bit = idx % 8
        if 0 <= byte_idx < EVENT_FLAG_DIM:
            target[byte_idx] = np.uint8(int(target[byte_idx]) | (1 << bit))
    return target


def encode_map_target(criteria: list[str]) -> int:
    for token in criteria:
        raw = token.strip()
        if raw.startswith("map_reached:"):
            return int(raw.split(":", 1)[1])
    return MAP_TARGET_EMPTY


def encode_stat_target(criteria: list[str]) -> np.ndarray:
    target = np.zeros(STAT_TARGET_DIM, dtype=np.uint8)
    for token in criteria:
        raw = token.strip()
        stat_name = None
        if raw.startswith("stat:"):
            stat_name = raw.split(":", 1)[1]
        elif raw.startswith("stat_on_target_map:"):
            stat_name = raw.split(":", 1)[1]
        if stat_name is None:
            continue
        try:
            target[STAT_TARGET_NAMES.index(stat_name)] = 1
        except ValueError:
            continue
    return target


def build_hrl_obs_dict(planner: PlannerOutput | None, subgoal_index: int = 0) -> dict[str, np.ndarray]:
    if planner is None:
        return {
            "hrl_target_map_id": np.array([0], dtype=np.uint8),
            "hrl_subgoal_index": np.array([0], dtype=np.uint8),
            "hrl_goal_event_target": np.zeros(EVENT_FLAG_DIM, dtype=np.uint8),
            "hrl_subgoal_event_target": np.zeros(EVENT_FLAG_DIM, dtype=np.uint8),
            "hrl_subgoal_map_target": np.array([MAP_TARGET_EMPTY], dtype=np.uint8),
            "hrl_subgoal_stat_target": np.zeros(STAT_TARGET_DIM, dtype=np.uint8),
        }

    active = None
    if planner.subgoal and 0 <= subgoal_index < len(planner.subgoal):
        active = planner.subgoal[subgoal_index]
    active_criteria = list(active.success_criteria) if active is not None else []

    return {
        "hrl_target_map_id": np.array([int(planner.target_map_id or 0)], dtype=np.uint8),
        "hrl_subgoal_index": np.array(
            [min(max(int(subgoal_index), 0), MAX_SUBGOAL_INDEX)], dtype=np.uint8
        ),
        "hrl_goal_event_target": encode_event_target(planner.success_criteria),
        "hrl_subgoal_event_target": encode_event_target(active_criteria),
        "hrl_subgoal_map_target": np.array(
            [encode_map_target(active_criteria)], dtype=np.uint8
        ),
        "hrl_subgoal_stat_target": encode_stat_target(active_criteria),
    }


HRL_OBS_SPACES = {
    "hrl_target_map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
    "hrl_subgoal_index": spaces.Box(
        low=0, high=MAX_SUBGOAL_INDEX, shape=(1,), dtype=np.uint8
    ),
    "hrl_goal_event_target": spaces.Box(
        low=0, high=1, shape=(EVENT_FLAG_DIM,), dtype=np.uint8
    ),
    "hrl_subgoal_event_target": spaces.Box(
        low=0, high=1, shape=(EVENT_FLAG_DIM,), dtype=np.uint8
    ),
    "hrl_subgoal_map_target": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
    "hrl_subgoal_stat_target": spaces.Box(
        low=0, high=1, shape=(STAT_TARGET_DIM,), dtype=np.uint8
    ),
}


def extend_observation_space(observation_space: spaces.Space) -> spaces.Space:
    if not isinstance(observation_space, spaces.Dict):
        return observation_space
    merged = dict(observation_space.spaces)
    merged.update(HRL_OBS_SPACES)
    return spaces.Dict(merged)


def attach_hrl_obs(
    obs: dict[str, Any],
    planner: PlannerOutput | None,
    *,
    subgoal_index: int = 0,
) -> dict[str, Any]:
    merged = dict(obs)
    merged.update(build_hrl_obs_dict(planner, subgoal_index=subgoal_index))
    return merged
