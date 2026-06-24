import gymnasium as gym

from pokemon_hrl.mode.progress import ProgressCheck, progress_from_info, progress_to_info
from pokemon_hrl.types import PlannerOutput, ProgressResult, Subgoal, WorldState


def _planner(**kwargs) -> PlannerOutput:
    defaults = dict(
        subgoal=[],
        hint={"target_map_id": 3},
        success_criteria=["map_reached:3"],
        failure_criteria=[],
    )
    defaults.update(kwargs)
    return PlannerOutput(**defaults)


def _state(map_id: int, **resources) -> WorldState:
    return WorldState(
        map_id=map_id,
        x=0,
        y=0,
        badges=0,
        resources=resources,
    )


def test_progress_check_target_map_reached():
    check = ProgressCheck()
    planner = _planner(success_criteria=[])
    before = _state(2)
    after = _state(3)
    result = check.check_criteria(planner, before, after)
    assert result.success
    assert result.reason == "target_map_reached"


def test_progress_check_stat_on_target_map_requires_correct_map():
    check = ProgressCheck()
    planner = _planner(
        hint={"target_map_id": 2},
        success_criteria=["stat_on_target_map:first_npc_talk_count"],
    )
    wrong_map = check.check_criteria(
        planner,
        _state(37, first_npc_talk_count=0),
        _state(37, first_npc_talk_count=1),
    )
    assert not wrong_map.success

    correct_map = check.check_criteria(
        planner,
        _state(2, first_npc_talk_count=0),
        _state(2, first_npc_talk_count=1),
    )
    assert correct_map.success
    assert correct_map.reason == "success_criteria_met"


def test_progress_check_map_reached_requires_transition():
    check = ProgressCheck()
    planner = _planner(success_criteria=["map_reached:2"])
    same_map = check.check_criteria(planner, _state(2), _state(2))
    assert not same_map.success
    arrived = check.check_criteria(planner, _state(1), _state(2))
    assert arrived.success
    assert arrived.reason == "success_criteria_met"


def test_progress_to_info_roundtrip():
    planner = _planner()
    after = _state(3)
    progress = ProgressResult(success=True, reason="target_map_reached")
    info = progress_to_info(progress, planner, after)
    restored = progress_from_info(info)
    assert restored is not None
    assert restored.success
    assert restored.reason == "target_map_reached"


def test_progress_check_wrapper_step(monkeypatch):
    from pokemon_hrl.env.progress_wrapper import ProgressCheckWrapper

    class _InnerEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Discrete(1)
            self.action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 0, 1.0, False, False, {}

    base = _InnerEnv()
    monkeypatch.setattr(
        "pokemon_hrl.env.progress_wrapper.unwrap_hrl_env",
        lambda _env: base,
    )
    monkeypatch.setattr(
        "pokemon_hrl.env.progress_wrapper.extract_world_state",
        lambda env, global_step=0: _state(3 if global_step > 0 else 2),
    )

    wrapper = ProgressCheckWrapper(
        _InnerEnv(), _planner(success_criteria=[]), log_goal_events=False
    )
    wrapper.reset()
    _, _, terminated, _, info = wrapper.step(0)

    assert info["hrl_progress_success"] == 1
    assert info["hrl_progress_reason"] == "target_map_reached"
    assert info["hrl_goal_event"] == 1
    assert terminated is True


def test_progress_check_subgoal_uses_explicit_criteria():
    check = ProgressCheck()
    planner = _planner(
        hint={"target_map_id": 2},
        subgoal=[
            Subgoal(
                success_criteria=["stat_on_target_map:first_object_interaction_count"],
            )
        ],
    )
    assert check.check_subgoal_met(
        planner.subgoal[0],
        planner,
        _state(2, first_object_interaction_count=0),
        _state(2, first_object_interaction_count=1),
    )
    assert not check.check_subgoal_met(
        planner.subgoal[0],
        planner,
        _state(2, first_npc_talk_count=0),
        _state(2, first_npc_talk_count=1),
    )


def test_progress_wrapper_subgoal_reward(monkeypatch):
    from pokemon_hrl.env.progress_wrapper import ProgressCheckWrapper

    class _InnerEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Discrete(1)
            self.action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 0, 0.0, False, False, {}

    states = [
        _state(2, first_npc_talk_count=0),
        _state(2, first_npc_talk_count=1),
    ]

    def _extract(_env, global_step=0):
        idx = min(global_step, len(states) - 1)
        return states[idx]

    monkeypatch.setattr(
        "pokemon_hrl.env.progress_wrapper.unwrap_hrl_env",
        lambda _env: _InnerEnv(),
    )
    monkeypatch.setattr(
        "pokemon_hrl.env.progress_wrapper.extract_world_state",
        _extract,
    )

    planner = _planner(
        hint={"target_map_id": 2},
        subgoal=[
            Subgoal(
                success_criteria=["stat_on_target_map:first_npc_talk_count"],
            ),
            Subgoal(
                success_criteria=["stat_on_target_map:first_object_interaction_count"],
            ),
        ],
        success_criteria=["stat_on_target_map:first_object_interaction_count"],
    )
    wrapper = ProgressCheckWrapper(
        _InnerEnv(),
        planner,
        log_goal_events=False,
        subgoal_success_reward=2.0,
    )
    wrapper.reset()
    _, reward, terminated, _, info = wrapper.step(0)

    assert info["hrl_subgoal_success"] == 1
    assert info["hrl_subgoal_completed"] == "stat_on_target_map:first_npc_talk_count"
    assert (
        info["hrl_current_subgoal"]
        == "stat_on_target_map:first_object_interaction_count"
    )
    assert reward == 2.0
    assert not terminated


def test_progress_wrapper_reward_floor_breach(monkeypatch):
    from pokemon_hrl.env.progress_wrapper import ProgressCheckWrapper

    class _InnerEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Discrete(1)
            self.action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 0, -5.0, False, False, {}

    monkeypatch.setattr(
        "pokemon_hrl.env.progress_wrapper.unwrap_hrl_env",
        lambda _env: _InnerEnv(),
    )
    monkeypatch.setattr(
        "pokemon_hrl.env.progress_wrapper.extract_world_state",
        lambda env, global_step=0: _state(2),
    )

    wrapper = ProgressCheckWrapper(
        _InnerEnv(),
        _planner(success_criteria=[]),
        log_goal_events=False,
        reward_floor=-10.0,
        reward_floor_rollback_penalty=-1.0,
    )
    wrapper.reset()
    _, reward1, _, _, info1 = wrapper.step(0)
    assert reward1 == -5.0
    assert info1.get("hrl_reward_floor_breach", 0) == 0

    _, reward2, _, _, info2 = wrapper.step(0)
    assert info2["hrl_reward_floor_breach"] == 1
    assert reward2 == -6.0
    assert info2["hrl_reward_since_checkpoint"] <= -10.0
