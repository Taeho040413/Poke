from pokemon_hrl.types import Mode, PlannerOutput, Subgoal


def test_planner_target_map_id():
    planner = PlannerOutput(
        subgoal=[Subgoal(success_criteria=["map_reached:5"])],
        hint={"target_map_id": 5},
        success_criteria=["flag:EVENT_BEAT_BROCK"],
        failure_criteria=[],
    )
    assert planner.target_map_id == 5
