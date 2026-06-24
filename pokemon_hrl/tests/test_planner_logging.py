from pokemon_hrl.planner.logging import log_active_goal_state, log_planner_output
from pokemon_hrl.types import PlannerOutput, Subgoal


def _planner() -> PlannerOutput:
    return PlannerOutput(
        subgoal=[
            Subgoal(
                success_criteria=["stat_on_target_map:first_npc_talk_count"],
                where="상록시티(map_id=2) 거리",
                what="아직 대화하지 않은 NPC와 첫 대화하기",
                how="NPC 앞까지 타일 이동 후 A 입력",
            ),
            Subgoal(
                success_criteria=["stat_on_target_map:first_object_interaction_count"],
                where="상록시티(map_id=2)",
                what="표지판이나 상점 카운터 첫 조사",
                how="오브젝트 앞까지 이동 후 A 입력",
            ),
        ],
        hint={"target_map_id": 2},
        success_criteria=["stat_on_target_map:first_npc_talk_count"],
        failure_criteria=["no_progress"],
    )


def test_log_planner_output_training(capsys):
    log_planner_output(_planner(), source="training", scenario_index=0)
    out = capsys.readouterr().out
    assert "[planner:training] scenario=0" in out
    assert "goal_key:" in out
    assert "PEWTER_CITY" in out
    assert "어디: 상록시티" in out
    assert "무엇: 아직 대화하지 않은 NPC" in out
    assert "방법: NPC 앞까지" in out
    assert "stat_on_target_map:first_npc_talk_count" in out
    assert "no_progress" in out


def test_log_active_goal_state(capsys):
    log_active_goal_state(_planner(), subgoal_index=0, map_id=2, env_id=0)
    out = capsys.readouterr().out
    assert "[goal:active]" in out
    assert "PEWTER_CITY" in out
    assert "target_map=2" in out
    assert "아직 대화하지 않은 NPC와 첫 대화하기" in out
