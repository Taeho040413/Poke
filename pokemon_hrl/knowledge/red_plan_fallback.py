"""Deterministic chapter-fact fallback plans when LLM output is invalid."""

from __future__ import annotations

from typing import Any

from pokemon_hrl.knowledge.red_maps import MapIds

_OAKS_LAB = int(MapIds.OAKS_LAB.value)
_PALLET_TOWN = int(MapIds.PALLET_TOWN.value)
_ROUTE_1 = int(MapIds.ROUTE_1.value)
_VIRIDIAN_CITY = int(MapIds.VIRIDIAN_CITY.value)
_VIRIDIAN_MART = int(MapIds.VIRIDIAN_MART.value)


def _current_map_id(world_state: dict[str, Any], planning_context: dict[str, Any]) -> int:
    if "map_id" in world_state:
        return int(world_state["map_id"])
    return int(planning_context.get("current_map_id", 0))


def _goal_criteria(chapter_goal: dict[str, Any]) -> list[str]:
    return list(chapter_goal.get("suggested_goal_success_criteria") or [])


def _fallback_got_oaks_parcel(
    *,
    current_map_id: int,
    goal_criteria: list[str],
) -> dict[str, Any]:
    if current_map_id == _VIRIDIAN_MART:
        return {
            "subgoal": [
                {
                    "success_criteria": list(goal_criteria),
                    "where": "상록시티 프렌들리숍 내부",
                    "what": "상점 직원 이벤트를 통해 Oak's Parcel을 받는다",
                    "how": "직원에게 접근해 A를 누르거나 자동 대화 이벤트를 발생시킨다",
                }
            ],
            "hint": {"target_map_id": _VIRIDIAN_MART},
            "success_criteria": list(goal_criteria),
            "failure_criteria": ["no_progress"],
        }

    return {
        "subgoal": [
            {
                "success_criteria": [f"map_reached:{_PALLET_TOWN}"],
                "where": "오크 연구소 출구",
                "what": "오크 연구소를 나와 태초마을로 이동한다",
                "how": "출구 타일로 이동해 연구소 밖으로 나간다",
            },
            {
                "success_criteria": [f"map_reached:{_ROUTE_1}"],
                "where": "태초마을 북쪽",
                "what": "1번도로로 진입한다",
                "how": "마을 북쪽 출구 방향으로 이동한다",
            },
            {
                "success_criteria": [f"map_reached:{_VIRIDIAN_CITY}"],
                "where": "1번도로",
                "what": "상록시티에 도착한다",
                "how": "길을 따라 북쪽으로 이동한다",
            },
            {
                "success_criteria": [f"map_reached:{_VIRIDIAN_MART}"],
                "where": "상록시티",
                "what": "프렌들리숍 내부로 들어간다",
                "how": "프렌들리숍 입구 타일로 이동해 건물 안으로 진입한다",
            },
        ],
        "hint": {"target_map_id": _VIRIDIAN_MART},
        "success_criteria": list(goal_criteria),
        "failure_criteria": ["no_progress"],
    }


def _fallback_oak_got_parcel(
    *,
    current_map_id: int,
    goal_criteria: list[str],
) -> dict[str, Any]:
    _ = current_map_id
    return {
        "subgoal": [
            {
                "success_criteria": [f"map_reached:{_VIRIDIAN_CITY}"],
                "where": "프렌들리숍",
                "what": "상록시티 밖으로 나간다",
                "how": "건물 출구로 이동해 상록시티 맵으로 나간다",
            },
            {
                "success_criteria": [f"map_reached:{_ROUTE_1}"],
                "where": "상록시티 남쪽",
                "what": "1번도로로 진입한다",
                "how": "남쪽 출구 방향으로 이동한다",
            },
            {
                "success_criteria": [f"map_reached:{_PALLET_TOWN}"],
                "where": "1번도로",
                "what": "태초마을에 도착한다",
                "how": "남쪽으로 이동해 태초마을에 진입한다",
            },
            {
                "success_criteria": list(goal_criteria),
                "where": "오크 연구소",
                "what": "오크 박사에게 Oak's Parcel을 전달한다",
                "how": "오크 박사에게 접근해 A를 눌러 대화 이벤트를 진행한다",
            },
        ],
        "hint": {"target_map_id": _OAKS_LAB},
        "success_criteria": list(goal_criteria),
        "failure_criteria": ["no_progress"],
    }


def _fallback_beat_brock(goal_criteria: list[str]) -> dict[str, Any]:
    pewter_gym = int(MapIds.PEWTER_GYM.value)
    return {
        "subgoal": [
            {
                "success_criteria": [f"map_reached:{int(MapIds.ROUTE_2.value)}"],
                "where": "상록시티 북쪽",
                "what": "2번도로로 진입한다",
                "how": "북쪽 출구 방향으로 이동한다",
            },
            {
                "success_criteria": [f"map_reached:{int(MapIds.VIRIDIAN_FOREST.value)}"],
                "where": "2번도로",
                "what": "상록시티 숲을 통과한다",
                "how": "숲 입구로 이동해 내부를 통과한다",
            },
            {
                "success_criteria": [f"map_reached:{int(MapIds.PEWTER_CITY.value)}"],
                "where": "상록시티 숲 북쪽",
                "what": "회색시티에 도착한다",
                "how": "숲을 빠져나와 회색시티로 진입한다",
            },
            {
                "success_criteria": list(goal_criteria),
                "where": "회색시티 체육관",
                "what": "브록과 체육관 배틀을 진행한다",
                "how": "브록에게 접근해 배틀을 시작한다",
            },
        ],
        "hint": {"target_map_id": pewter_gym},
        "success_criteria": list(goal_criteria),
        "failure_criteria": ["no_progress"],
    }


def build_deterministic_fallback_plan(
    chapter_goal: dict[str, Any],
    planner_knowledge: dict[str, Any],
    planning_context: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a chapter-fact plan without LLM involvement."""
    del planner_knowledge
    goal_criteria = _goal_criteria(chapter_goal)
    if not goal_criteria:
        return {
            "subgoal": [],
            "hint": {"target_map_id": _current_map_id(world_state, planning_context)},
            "success_criteria": [],
            "failure_criteria": ["no_progress"],
        }

    goal_key = goal_criteria[0]
    current_map_id = _current_map_id(world_state, planning_context)

    if goal_key == "flag:EVENT_GOT_OAKS_PARCEL":
        return _fallback_got_oaks_parcel(
            current_map_id=current_map_id,
            goal_criteria=goal_criteria,
        )
    if goal_key == "flag:EVENT_OAK_GOT_PARCEL":
        return _fallback_oak_got_parcel(
            current_map_id=current_map_id,
            goal_criteria=goal_criteria,
        )
    if goal_key == "flag:EVENT_BEAT_BROCK":
        return _fallback_beat_brock(goal_criteria)

    return {
        "subgoal": [
            {
                "success_criteria": list(goal_criteria),
                "where": "현재 위치",
                "what": "챕터 목표를 달성한다",
                "how": "목표 조건에 맞는 행동을 수행한다",
            }
        ],
        "hint": {"target_map_id": current_map_id},
        "success_criteria": list(goal_criteria),
        "failure_criteria": ["no_progress"],
    }
