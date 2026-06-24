"""Tests for code-based planner knowledge, validation, and fallback."""

from __future__ import annotations

import pytest

from pokemon_hrl.knowledge.planner_knowledge import build_planner_knowledge
from pokemon_hrl.knowledge.red_maps import MapIds
from pokemon_hrl.knowledge.red_plan_fallback import build_deterministic_fallback_plan
from pokemon_hrl.knowledge.red_plan_validator import validate_and_repair_plan
from pokemon_hrl.planner.progression import (
    chapter_goal_payload,
    planning_context_payload,
    scope_planner_to_chapter,
)
from pokemon_hrl.planner.validation import parse_planner_dict
from pokemon_hrl.types import WorldState


def _state(**kwargs) -> WorldState:
    defaults = dict(map_id=0, x=0, y=0, badges=0, flags={})
    defaults.update(kwargs)
    return WorldState(**defaults)


def _payloads(state: WorldState) -> tuple[dict, dict, dict]:
    chapter_goal = chapter_goal_payload(state)
    planning_context = planning_context_payload(state)
    world_state = {
        "map_id": state.map_id,
        "active_flags": [k for k, v in state.flags.items() if v],
    }
    return chapter_goal, planning_context, world_state


def _oak_receive_bad_plan() -> dict:
    return {
        "subgoal": [
            {
                "success_criteria": ["stat_on_target_map:first_npc_talk_count"],
                "where": "오크 연구소",
                "what": "오크 박사에게 Oak's Parcel을 받는다",
                "how": "오크 박사에게 A를 누른다",
            },
            {
                "success_criteria": ["map_reached:1"],
                "where": "연구소",
                "what": "상록시티로 간다",
                "how": "이동한다",
            },
        ],
        "hint": {"target_map_id": int(MapIds.OAKS_LAB.value)},
        "success_criteria": ["flag:EVENT_GOT_OAKS_PARCEL"],
        "failure_criteria": [],
    }


def test_planner_knowledge_oaks_parcel_from_oaks_lab():
    state = _state(
        map_id=int(MapIds.OAKS_LAB.value),
        flags={"EVENT_GOT_STARTER": True},
    )
    chapter_goal, planning_context, world_state = _payloads(state)
    knowledge = build_planner_knowledge(chapter_goal, world_state, planning_context)

    assert knowledge["current_map"]["id"] == int(MapIds.OAKS_LAB.value)
    assert knowledge["current_map"]["name"] == "OAKS_LAB"
    assert knowledge["required_target_map_id"] == int(MapIds.VIRIDIAN_MART.value)
    assert "VIRIDIAN_MART" in knowledge["map_knowledge"]
    assert knowledge["chapter_facts"]["label"] == "receive Oak's Parcel"

    validation = validate_and_repair_plan(
        _oak_receive_bad_plan(),
        chapter_goal,
        knowledge,
        planning_context,
        world_state,
    )
    assert validation.rejected

    fallback = build_deterministic_fallback_plan(
        chapter_goal,
        knowledge,
        planning_context,
        world_state,
    )
    output = scope_planner_to_chapter(parse_planner_dict(fallback), state)

    assert output.success_criteria == ["flag:EVENT_GOT_OAKS_PARCEL"]
    assert output.hint["target_map_id"] == int(MapIds.VIRIDIAN_MART.value)
    combined_text = " ".join(
        f"{sg.where} {sg.what} {sg.how}" for sg in output.subgoal
    )
    assert "오크 박사에게" not in combined_text
    assert "Parcel을 받" not in combined_text
    assert output.subgoal[0].success_criteria == [f"map_reached:{int(MapIds.PALLET_TOWN.value)}"]
    route_maps = {
        int(token.split(":", 1)[1])
        for sg in output.subgoal
        for token in sg.success_criteria
        if str(token).startswith("map_reached:")
    }
    assert int(MapIds.PALLET_TOWN.value) in route_maps
    assert int(MapIds.ROUTE_1.value) in route_maps
    assert int(MapIds.VIRIDIAN_CITY.value) in route_maps
    assert int(MapIds.VIRIDIAN_MART.value) in route_maps


def test_planner_knowledge_oaks_parcel_at_viridian_mart():
    state = _state(
        map_id=int(MapIds.VIRIDIAN_MART.value),
        flags={"EVENT_GOT_STARTER": True},
    )
    chapter_goal, planning_context, world_state = _payloads(state)
    knowledge = build_planner_knowledge(chapter_goal, world_state, planning_context)

    fallback = build_deterministic_fallback_plan(
        chapter_goal,
        knowledge,
        planning_context,
        world_state,
    )
    output = scope_planner_to_chapter(parse_planner_dict(fallback), state)

    validation_at_mart = validate_and_repair_plan(
        fallback,
        chapter_goal,
        knowledge,
        planning_context,
        world_state,
    )
    assert not validation_at_mart.rejected

    assert output.hint["target_map_id"] == int(MapIds.VIRIDIAN_MART.value)
    assert len(output.subgoal) == 1
    assert output.subgoal[0].success_criteria == ["flag:EVENT_GOT_OAKS_PARCEL"]
    text = f"{output.subgoal[0].where} {output.subgoal[0].what} {output.subgoal[0].how}"
    assert "직원" in text or "프렌들리숍" in text or "상점" in text
    assert "오크 박사" not in text
    assert "OAKS_LAB" not in text.upper()


def test_planner_knowledge_deliver_parcel_from_viridian_mart():
    state = _state(
        map_id=int(MapIds.VIRIDIAN_MART.value),
        flags={
            "EVENT_GOT_STARTER": True,
            "EVENT_GOT_OAKS_PARCEL": True,
        },
    )
    chapter_goal, planning_context, world_state = _payloads(state)
    knowledge = build_planner_knowledge(chapter_goal, world_state, planning_context)

    assert chapter_goal["suggested_goal_success_criteria"] == ["flag:EVENT_OAK_GOT_PARCEL"]
    assert knowledge["required_target_map_id"] == int(MapIds.OAKS_LAB.value)

    fallback = build_deterministic_fallback_plan(
        chapter_goal,
        knowledge,
        planning_context,
        world_state,
    )
    output = scope_planner_to_chapter(parse_planner_dict(fallback), state)

    assert output.hint["target_map_id"] == int(MapIds.OAKS_LAB.value)
    final = output.subgoal[-1]
    assert final.success_criteria == ["flag:EVENT_OAK_GOT_PARCEL"]
    final_text = f"{final.where} {final.what} {final.how}"
    assert "오크" in final_text or "박사" in final_text or "전달" in final_text


def test_validator_repairs_invented_lavender_flag():
    state = _state(map_id=0, flags={})
    chapter_goal, planning_context, world_state = _payloads(state)
    knowledge = build_planner_knowledge(chapter_goal, world_state, planning_context)

    bad_plan = {
        "subgoal": [
            {
                "success_criteria": ["flag:EVENT_GOT_STARTER"],
                "where": "연구소",
                "what": "스타터를 고른다",
                "how": "이동한다",
            },
            {
                "success_criteria": ["flag:EVENT_FOLLOWED_OAK_INTO_LAB"],
                "where": "연구소",
                "what": "연구소에 들어간다",
                "how": "이동한다",
            },
            {
                "success_criteria": ["flag:EVENT_REACHED_LAVENDER_TOWN"],
                "where": "라벤더",
                "what": "라벤더에 간다",
                "how": "이동한다",
            },
        ],
        "hint": {"target_map_id": int(MapIds.VIRIDIAN_MART.value)},
        "success_criteria": ["flag:EVENT_GOT_OAKS_PARCEL"],
        "failure_criteria": [],
    }

    validation = validate_and_repair_plan(
        bad_plan,
        chapter_goal,
        knowledge,
        planning_context,
        world_state,
    )
    repaired_tokens = [
        token
        for sg in validation.plan["subgoal"]
        for token in sg["success_criteria"]
    ]
    assert "flag:EVENT_REACHED_LAVENDER_TOWN" not in repaired_tokens
    assert f"map_reached:{int(MapIds.LAVENDER_TOWN.value)}" in repaired_tokens


def test_do_not_emit_excludes_completed_only_not_chapter_goal():
    state = _state(
        map_id=int(MapIds.OAKS_LAB.value),
        flags={"EVENT_GOT_STARTER": True},
    )
    context = planning_context_payload(state)
    assert "flag:EVENT_GOT_STARTER" in context["do_not_emit"]
    assert "flag:EVENT_GOT_OAKS_PARCEL" not in context["do_not_emit"]
