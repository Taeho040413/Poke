import numpy as np

from pokemon_hrl.planner.criteria import (
    STAT_TARGET_DIM,
    attach_hrl_obs,
    build_hrl_obs_dict,
    encode_event_target,
    encode_stat_target,
    planner_goal_key,
    subgoal_label,
)
from pokemon_hrl.types import PlannerOutput, Subgoal


def test_encode_event_target_sets_flag_bit():
    target = encode_event_target(["flag:EVENT_BEAT_BROCK"])
    assert target.sum() > 0


def test_build_hrl_obs_dict_shapes():
    planner = PlannerOutput(
        subgoal=[
            Subgoal(success_criteria=["stat_on_target_map:first_npc_talk_count"]),
            Subgoal(success_criteria=["map_reached:3"]),
        ],
        hint={"target_map_id": 2},
        success_criteria=["flag:EVENT_BEAT_BROCK"],
        failure_criteria=["no_progress"],
    )
    obs = build_hrl_obs_dict(planner, subgoal_index=0)
    assert obs["hrl_target_map_id"].shape == (1,)
    assert obs["hrl_goal_event_target"].shape == (320,)
    assert obs["hrl_subgoal_stat_target"].shape == (STAT_TARGET_DIM,)
    assert obs["hrl_subgoal_index"][0] == 0


def test_attach_hrl_obs_merges_into_env_obs():
    planner = PlannerOutput(
        subgoal=[Subgoal(success_criteria=["stat_on_target_map:first_npc_talk_count"])],
        hint={"target_map_id": 2},
        success_criteria=["flag:EVENT_BEAT_BROCK"],
        failure_criteria=[],
    )
    merged = attach_hrl_obs({"screen": np.zeros((1, 1, 1))}, planner, subgoal_index=0)
    assert "hrl_goal_event_target" in merged
    assert merged["hrl_target_map_id"][0] == 2


def test_planner_goal_key_from_criteria():
    planner = PlannerOutput(
        subgoal=[],
        hint={"target_map_id": 2},
        success_criteria=["flag:EVENT_BEAT_BROCK", "map_reached:5"],
        failure_criteria=[],
    )
    assert planner_goal_key(planner) == "flag:EVENT_BEAT_BROCK|map_reached:5"
    assert subgoal_label(Subgoal(success_criteria=["stat:item_count"])) == "stat:item_count"


def test_encode_stat_target_one_hot():
    vec = encode_stat_target(["stat_on_target_map:first_npc_talk_count"])
    assert vec[0] == 1
    assert vec.sum() == 1
