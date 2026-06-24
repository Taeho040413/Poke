from pokemon_hrl.planner.validation import PlannerValidationError, parse_planner_dict
from pokemon_hrl.planner.replan import should_invoke_planner
from pokemon_hrl.summarizer.mapping import (
    exploration_coverage,
    failure_cause,
    semantic_progression,
)
from pokemon_hrl.summarizer.rule_based import RuleBasedSummarizer
from pokemon_hrl.types import ProgressResult, WorldState


def _state(**kwargs) -> WorldState:
    base = dict(
        map_id=2,
        x=10,
        y=12,
        badges=1,
        flags={"EVENT_GOT_POKEDEX": True},
        party=[{"species": 1, "level": 8, "hp": 20, "max_hp": 20}],
        bag=[{"item_id": 4, "quantity": 1}],
        resources={"money": 1200},
        success_memory=[{"goal": "g1", "subgoal": "talk", "timestamp_step": 10}],
        failure_memory=[{"goal": "g0", "cause": "no_progress", "count": 1}],
        map_visited=[0, 1, 2],
    )
    base.update(kwargs)
    return WorldState(**base)


def test_summarizer_full_mapping():
    summary = RuleBasedSummarizer().summarize(_state())
    assert "map_id=2" in summary.semantic_progression
    assert "EVENT_GOT_POKEDEX" in summary.semantic_progression
    assert "visited_maps=" in summary.exploration_coverage
    assert "recent_successes" in summary.interaction_outcome
    assert "recent_failures" in summary.failure_cause
    assert summary.evidence["money"] == 1200


def test_parse_planner_dict_valid():
    out = parse_planner_dict(
        {
            "subgoal": [
                {"success_criteria": ["map_reached:3"]},
            ],
            "hint": {"target_map_id": 3},
            "success_criteria": ["flag:EVENT_BEAT_BROCK"],
            "failure_criteria": ["no_progress"],
        }
    )
    assert out.target_map_id == 3
    assert out.subgoal[0].success_criteria == ["map_reached:3"]


def test_parse_planner_rejects_string_subgoal():
    try:
        parse_planner_dict(
            {
                "subgoal": ["talk to npcs"],
                "hint": {"target_map_id": 2},
                "success_criteria": ["stat_on_target_map:first_npc_talk_count"],
                "failure_criteria": [],
            }
        )
        raise AssertionError("expected PlannerValidationError")
    except PlannerValidationError:
        pass


def test_should_invoke_planner_goal_check():
    ok = ProgressResult(success=True)
    assert should_invoke_planner("goal_check", ok)
    assert not should_invoke_planner("never", ok)
    assert should_invoke_planner("goal_check", ProgressResult(), initial=True)


def test_should_invoke_on_mode_timeout():
    progress = ProgressResult(truncated=True)
    info = {"truncated_reason": "mode_max_steps"}
    assert should_invoke_planner("goal_check", progress, info=info)


def test_world_state_merge_preserves_memory():
    from pokemon_hrl.world_state.merge import merge_extracted_state

    prev = _state(success_memory=[{"goal": "keep"}])
    extracted = _state(map_id=99, success_memory=[])
    merged = merge_extracted_state(prev, extracted)
    assert merged.map_id == 99
    assert merged.success_memory == [{"goal": "keep"}]


def test_rule_based_planner_uses_summary():
    from pokemon_hrl.planner.rule_based import RuleBasedPlanner

    planner = RuleBasedPlanner("training/curriculum.yaml")
    state = _state(map_id=2)
    summary = RuleBasedSummarizer().summarize(state)
    out = planner.plan(summary, state)
    assert out.target_map_id is not None
    assert out.success_criteria


def test_prompt_loads():
    from pokemon_hrl.planner.prompt import load_system_prompt

    text = load_system_prompt()
    assert "objective" in text.lower() or "goal" in text.lower()


def test_parse_planner_rejects_missing_target():
    try:
        parse_planner_dict(
            {
                "hint": {},
                "success_criteria": ["flag:EVENT_BEAT_BROCK"],
            }
        )
        raise AssertionError("expected PlannerValidationError")
    except PlannerValidationError:
        pass


def test_parse_planner_rejects_missing_success_criteria():
    try:
        parse_planner_dict(
            {
                "hint": {"target_map_id": 2},
                "success_criteria": [],
            }
        )
        raise AssertionError("expected PlannerValidationError")
    except PlannerValidationError:
        pass
