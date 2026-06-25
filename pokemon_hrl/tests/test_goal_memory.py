import pytest

from pokemon_hrl.env.goal_memory import (
    GoalMemoryConfig,
    GoalMemoryTracker,
    front_tile_coords,
    goal_context_from_planner_dict,
    goal_memory_observation_space,
)
from pokemon_hrl.execution.action_space import HrlAction, tile_target_coords
from pokemon_hrl.execution.tile_blocked import TileBlockedTracker


class _Events:
    def __init__(self, flags: dict[str, bool] | None = None):
        self._flags = dict(flags or {})

    def get_event(self, name: str) -> bool:
        return bool(self._flags.get(name, False))


def test_local_maps_center_player_and_blocked_channel():
    cfg = GoalMemoryConfig(enabled=True, local_radius=2)
    tracker = GoalMemoryTracker(cfg)
    tracker.record_visit(1, 3, 4)
    tracker.record_visit(1, 5, 4)

    blocked = TileBlockedTracker(
        enabled=True, ttl_steps=5, confidence_threshold=1
    )
    blocked.record_failed_move(1, 3, 4, int(HrlAction.TILE_RIGHT), step=0)

    obs = tracker.build_obs(map_id=1, player_x=3, player_y=4, tile_blocked=blocked)
    local = obs["goal_memory_local"]
    assert local.shape == (7, 5, 5)
    center = cfg.local_radius
    assert local[0, center, center] == 1.0  # visited at player
    assert local[2, center, center + 1] == 1.0  # blocked east of player


def test_seen_memory_records_5x5_neighborhood():
    cfg = GoalMemoryConfig(
        enabled=True,
        local_radius=2,
        seen_radius=2,
        include_visited=False,
        include_seen=True,
        include_blocked=False,
        include_event_sources=False,
        include_warps=False,
        include_interactions=False,
    )
    tracker = GoalMemoryTracker(cfg)
    tracker.record_position_context(1, 10, 10)

    obs = tracker.build_obs(map_id=1, player_x=10, player_y=10, tile_blocked=None)
    local = obs["goal_memory_local"]
    assert local.shape == (1, 5, 5)
    assert local[0].sum() == 25.0
    assert local[0, 0, 0] == 1.0
    assert local[0, 2, 2] == 1.0
    assert local[0, 4, 4] == 1.0


def test_goal_vector_points_toward_target():
    cfg = GoalMemoryConfig(enabled=True, local_radius=5, include_goal_vector=True)
    tracker = GoalMemoryTracker(cfg)
    tracker.set_context(
        {
            "target_map_id": 7,
            "target_x": 10,
            "target_y": 8,
            "goal_key": "test",
        }
    )
    goal = tracker.build_goal_vector(map_id=7, player_x=7, player_y=8)
    assert goal.shape == (9,)
    assert goal[1] == 1.0  # on_target_map
    assert goal[2] > 0  # player_x_norm
    assert goal[3] > 0  # player_y_norm
    assert goal[4] > 0  # dx toward east
    assert goal[5] == 0.0  # dy


def test_goal_memory_goal_observation_space_shape_is_9():
    spaces = goal_memory_observation_space(GoalMemoryConfig(enabled=True))
    assert spaces["goal_memory_goal"].shape == (9,)


def test_player_norm_nonzero_when_coords_nonzero():
    cfg = GoalMemoryConfig(enabled=True)
    tracker = GoalMemoryTracker(cfg)
    goal = tracker.build_goal_vector(map_id=1, player_x=10, player_y=20)
    assert goal[2] == pytest.approx(10 / 255.0)
    assert goal[3] == pytest.approx(20 / 255.0)


def test_warp_memory_records_target_tile_for_tile_movement():
    cfg = GoalMemoryConfig(enabled=True, local_radius=2, include_warps=True)
    tracker = GoalMemoryTracker(cfg)
    x0, y0 = 5, 7
    action = int(HrlAction.TILE_RIGHT)
    warp_x, warp_y = tile_target_coords(x0, y0, action)
    tracker.record_warp(1, warp_x, warp_y)

    obs = tracker.build_obs(map_id=1, player_x=x0, player_y=y0, tile_blocked=None)
    local = obs["goal_memory_local"]
    # channels: visited, seen, blocked, event_source, warps, interact_success, interact_fail
    warp_channel = local[4]
    center = cfg.local_radius
    assert warp_channel[center, center + 1] == 1.0
    assert warp_channel[center, center] == 0.0


def test_interaction_memory_records_front_tile_not_player_tile():
    cfg = GoalMemoryConfig(
        enabled=True,
        local_radius=2,
        include_interactions=True,
        include_seen=False,
        include_event_sources=False,
        include_visited=False,
        include_blocked=False,
        include_warps=False,
    )
    tracker = GoalMemoryTracker(cfg)
    player_x, player_y = 5, 5
    front_x, front_y = front_tile_coords(player_x, player_y, 0)  # facing down
    assert (front_x, front_y) == (5, 6)
    tracker.record_interact_success(1, front_x, front_y)

    obs = tracker.build_obs(
        map_id=1, player_x=player_x, player_y=player_y, tile_blocked=None
    )
    interact_channel = obs["goal_memory_local"][0]
    center = cfg.local_radius
    assert interact_channel[center + 1, center] == 1.0  # south of player
    assert interact_channel[center, center] == 0.0


def test_front_tile_coords_fallback_none_for_unknown_facing():
    assert front_tile_coords(1, 2, None) is None
    assert front_tile_coords(1, 2, 99) is None


def test_map_enter_reward_once_per_goal():
    cfg = GoalMemoryConfig(
        enabled=True,
        target_map_enter_reward=0.5,
        target_map_explore_reward=0.02,
    )
    tracker = GoalMemoryTracker(cfg)
    tracker.set_context({"target_map_id": 9, "goal_key": "go"})
    events = _Events()

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=9,
        x=1,
        y=1,
        map_before=8,
        map_after=9,
        new_tile_on_map=True,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.5 + 0.02

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=9,
        x=2,
        y=1,
        map_before=9,
        map_after=9,
        new_tile_on_map=True,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.02

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=9,
        x=3,
        y=1,
        map_before=8,
        map_after=9,
        new_tile_on_map=True,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.02


def test_event_rising_edge_reward():
    cfg = GoalMemoryConfig(enabled=True, target_event_done_reward=2.0)
    tracker = GoalMemoryTracker(cfg)
    tracker.set_context({"target_event_id": "EVENT_GOT_PARCEL", "goal_key": "parcel"})
    events = _Events({"EVENT_GOT_PARCEL": False})

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=1,
        x=0,
        y=0,
        map_before=1,
        map_after=1,
        new_tile_on_map=False,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.0

    events._flags["EVENT_GOT_PARCEL"] = True
    tracker.on_step_start(_Events({"EVENT_GOT_PARCEL": False}))
    tracker.on_post_step(
        map_id=1,
        x=0,
        y=0,
        map_before=1,
        map_after=1,
        new_tile_on_map=False,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 2.0


def test_event_source_memory_records_interaction_tile_on_rising_edge():
    cfg = GoalMemoryConfig(
        enabled=True,
        local_radius=2,
        target_event_done_reward=2.0,
        include_visited=False,
        include_seen=False,
        include_blocked=False,
        include_event_sources=True,
        include_warps=False,
        include_interactions=False,
    )
    tracker = GoalMemoryTracker(cfg)
    tracker.set_context({"target_event_id": "EVENT_GOT_PARCEL", "goal_key": "parcel"})

    events_after = _Events({"EVENT_GOT_PARCEL": True})
    tracker.on_step_start(_Events({"EVENT_GOT_PARCEL": False}))
    tracker.on_post_step(
        map_id=1,
        x=5,
        y=5,
        map_before=1,
        map_after=1,
        new_tile_on_map=False,
        interaction_success=True,
        interaction_fail=False,
        events_reader=events_after,
        interaction_x=6,
        interaction_y=5,
    )

    obs = tracker.build_obs(map_id=1, player_x=5, player_y=5, tile_blocked=None)
    channel = obs["goal_memory_local"][0]
    center = cfg.local_radius
    assert channel[center, center + 1] == 1.0


def test_object_only_hint_does_not_grant_interaction_reward():
    cfg = GoalMemoryConfig(enabled=True, target_interaction_reward=0.2)
    tracker = GoalMemoryTracker(cfg)
    tracker.set_context({"target_object_id": "oak_lab_pc", "goal_key": "obj"})
    events = _Events()

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=1,
        x=5,
        y=5,
        map_before=1,
        map_after=1,
        new_tile_on_map=False,
        interaction_success=True,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.0


def test_goal_memory_disabled_has_no_obs_spaces():
    assert goal_memory_observation_space(GoalMemoryConfig(enabled=False)) == {}


def test_new_tile_explore_reward_uses_destination_coord():
    cfg = GoalMemoryConfig(
        enabled=True,
        target_map_explore_reward=0.02,
    )
    tracker = GoalMemoryTracker(cfg)
    tracker.set_context({"target_map_id": 9, "goal_key": "go"})
    events = _Events()

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=9,
        x=3,
        y=1,
        map_before=9,
        map_after=9,
        new_tile_on_map=True,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.02

    tracker.on_step_start(events)
    tracker.on_post_step(
        map_id=9,
        x=3,
        y=1,
        map_before=9,
        map_after=9,
        new_tile_on_map=False,
        interaction_success=False,
        interaction_fail=False,
        events_reader=events,
    )
    assert tracker.consume_step_reward() == 0.0


def test_goal_context_from_planner_uses_subgoal_criteria():
    from pokemon_hrl.types import PlannerOutput, Subgoal

    planner = PlannerOutput(
        subgoal=[Subgoal(success_criteria=["flag:EVENT_GOT_PARCEL"])],
        hint={"target_map_id": 3, "target_x": 12, "target_y": 8},
        success_criteria=[],
        failure_criteria=[],
    )
    ctx = goal_context_from_planner_dict(planner, subgoal_index=0)
    assert ctx["target_map_id"] == 3
    assert ctx["target_x"] == 12
    assert ctx["target_y"] == 8
    assert ctx["target_event_id"] == "EVENT_GOT_PARCEL"
