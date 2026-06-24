import pytest

from pokemon_hrl.knowledge.red_maps import MapIds
from pokemon_hrl.planner.progression import (
    MAX_SUBGOALS,
    MIN_SUBGOALS,
    chapter_goal_payload,
    infer_chapter_goal,
    normalize_subgoals,
    planning_context_payload,
    scope_planner_to_chapter,
)
from pokemon_hrl.planner.validation import PlannerValidationError, parse_planner_dict
from pokemon_hrl.types import Subgoal, WorldState


def _state(**kwargs) -> WorldState:
    defaults = dict(map_id=0, x=0, y=0, badges=0, flags={})
    defaults.update(kwargs)
    return WorldState(**defaults)


def test_infer_chapter_goal_fresh_start_parcel():
    chapter = infer_chapter_goal(_state())
    assert chapter.chapter_id == "oaks_parcel"
    assert chapter.success_criteria == ["flag:EVENT_GOT_OAKS_PARCEL"]


def test_infer_chapter_goal_after_starter_parcel():
    chapter = infer_chapter_goal(
        _state(flags={"EVENT_GOT_STARTER": True}),
    )
    assert chapter.chapter_id == "oaks_parcel"
    assert chapter.success_criteria == ["flag:EVENT_GOT_OAKS_PARCEL"]


def test_infer_chapter_goal_deliver_parcel():
    chapter = infer_chapter_goal(
        _state(
            flags={
                "EVENT_GOT_STARTER": True,
                "EVENT_GOT_OAKS_PARCEL": True,
            }
        ),
    )
    assert chapter.chapter_id == "deliver_parcel"
    assert chapter.success_criteria == ["flag:EVENT_OAK_GOT_PARCEL"]


def test_scope_planner_clamps_goal_and_rejects_invalid_subgoals():
    state = _state(map_id=0, flags={})
    llm_plan = parse_planner_dict(
        {
            "subgoal": [
                {"success_criteria": ["flag:EVENT_BEAT_LT_SURGE"]},
                {"success_criteria": ["map_reached:53"]},
            ],
            "hint": {"target_map_id": 38},
            "success_criteria": ["flag:EVENT_REACHED_LAVENDER_TOWN"],
            "failure_criteria": [],
        }
    )
    with pytest.raises(PlannerValidationError, match="subgoals"):
        scope_planner_to_chapter(llm_plan, state)


def test_scope_planner_keeps_llm_subgoals_without_padding():
    state = _state(map_id=0, flags={})
    llm_plan = parse_planner_dict(
        {
            "subgoal": [
                {"success_criteria": ["flag:EVENT_GOT_STARTER"]},
                {"success_criteria": ["flag:EVENT_FOLLOWED_OAK_INTO_LAB"]},
            ],
            "hint": {"target_map_id": 0},
            "success_criteria": ["flag:EVENT_GOT_OAKS_PARCEL"],
            "failure_criteria": ["no_progress"],
        }
    )
    scoped = scope_planner_to_chapter(llm_plan, state)
    assert scoped.success_criteria == ["flag:EVENT_GOT_OAKS_PARCEL"]
    assert scoped.hint["target_map_id"] == int(MapIds.VIRIDIAN_MART.value)
    assert len(scoped.subgoal) == 2
    assert scoped.subgoal[0].success_criteria == ["flag:EVENT_GOT_STARTER"]
    assert scoped.subgoal[1].success_criteria == ["flag:EVENT_FOLLOWED_OAK_INTO_LAB"]


def test_normalize_subgoals_does_not_pad_short_plan():
    state = _state(map_id=0, flags={})
    normalized = normalize_subgoals(
        [],
        state,
        goal_criteria=["flag:EVENT_GOT_OAKS_PARCEL"],
    )
    assert normalized == []


def test_normalize_subgoals_trims_long_plan():
    state = _state(map_id=0, flags={})
    long_plan = [
        Subgoal(success_criteria=["stat_on_target_map:first_npc_talk_count"]),
        Subgoal(success_criteria=["stat_on_target_map:first_object_interaction_count"]),
        Subgoal(success_criteria=["stat_on_target_map:new_npc_textbox_count"]),
        Subgoal(success_criteria=["flag:EVENT_GOT_STARTER"]),
        Subgoal(success_criteria=["flag:EVENT_FOLLOWED_OAK_INTO_LAB"]),
    ]
    trimmed = normalize_subgoals(
        long_plan,
        state,
        goal_criteria=["flag:EVENT_GOT_OAKS_PARCEL"],
    )
    assert len(trimmed) == MAX_SUBGOALS


def test_chapter_goal_payload_includes_plan_size():
    payload = chapter_goal_payload(_state(flags={"EVENT_GOT_STARTER": True}))
    assert payload["current_chapter_id"] == "oaks_parcel"
    assert payload["plan_size_limits"]["min_subgoals"] == MIN_SUBGOALS
    assert payload["plan_size_limits"]["max_subgoals"] == MAX_SUBGOALS
    assert payload["plan_size_limits"]["min_total_steps_including_goal"] == 3
    assert "planning_context" in payload["instruction"]


def test_planning_context_marks_completed_flags_and_subgoals():
    state = _state(
        map_id=40,
        flags={"EVENT_GOT_STARTER": True},
        success_memory=[
            {
                "goal": "flag:EVENT_GOT_OAKS_PARCEL",
                "subgoal": "flag:EVENT_FOLLOWED_OAK_INTO_LAB",
                "timestamp_step": 10,
            }
        ],
        goal_stack={
            "goal_key": "flag:EVENT_GOT_OAKS_PARCEL",
            "subgoals": [
                {"success_criteria": ["flag:EVENT_GOT_STARTER"]},
                {"success_criteria": ["flag:EVENT_FOLLOWED_OAK_INTO_LAB"]},
            ],
            "current_index": 1,
        },
    )
    context = planning_context_payload(state)
    assert context["current_map_id"] == 40
    assert "flag:EVENT_GOT_STARTER" in context["completed_flags"]
    assert "flag:EVENT_FOLLOWED_OAK_INTO_LAB" in context["completed_subgoals"]
    assert "flag:EVENT_GOT_STARTER" in context["do_not_emit"]
    assert "flag:EVENT_GOT_OAKS_PARCEL" not in context["do_not_emit"]


def test_normalize_subgoals_filters_success_memory():
    state = _state(
        success_memory=[
            {
                "goal": "flag:EVENT_GOT_OAKS_PARCEL",
                "subgoal": "flag:EVENT_GOT_STARTER",
                "timestamp_step": 5,
            }
        ],
    )
    normalized = normalize_subgoals(
        [
            Subgoal(success_criteria=["flag:EVENT_GOT_STARTER"]),
            Subgoal(success_criteria=["flag:EVENT_FOLLOWED_OAK_INTO_LAB"]),
        ],
        state,
        goal_criteria=["flag:EVENT_GOT_OAKS_PARCEL"],
    )
    assert len(normalized) == 1
    assert normalized[0].success_criteria == ["flag:EVENT_FOLLOWED_OAK_INTO_LAB"]


def test_build_user_message_includes_planning_context():
    import json

    from pokemon_hrl.planner.prompt import build_user_message
    from pokemon_hrl.summarizer.rule_based import RuleBasedSummarizer

    state = _state(flags={"EVENT_GOT_STARTER": True}, map_id=40)
    summary = RuleBasedSummarizer().summarize(state)
    payload = json.loads(build_user_message(summary, state))
    assert "planning_context" in payload
    assert "planner_knowledge" in payload
    assert payload["planning_context"]["current_map_id"] == 40
    assert "flag:EVENT_GOT_STARTER" in payload["planning_context"]["do_not_emit"]
    assert "flag:EVENT_GOT_OAKS_PARCEL" not in payload["planning_context"]["do_not_emit"]
