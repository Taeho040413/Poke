"""Gym wrappers shared by HRL mode environments."""

from __future__ import annotations

import gymnasium as gym

from pokemon_hrl.env.goal_memory import goal_context_from_planner_dict
from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.types import PlannerOutput


def _sync_target_map_to_base(
    env: gym.Env,
    planner: PlannerOutput,
    *,
    subgoal_index: int = 0,
) -> None:
    try:
        base = unwrap_hrl_env(env)
    except TypeError:
        return
    setter = getattr(base, "set_goal_context", None)
    if callable(setter):
        setter(goal_context_from_planner_dict(planner, subgoal_index=subgoal_index))


class CurriculumWrapper(gym.Wrapper):
    """Attach curriculum planner metadata to info each step."""

    def __init__(self, env: gym.Env, planner: PlannerOutput):
        super().__init__(env)
        self.planner = planner
        _sync_target_map_to_base(env, planner)

    def set_planner(self, planner: PlannerOutput) -> None:
        self.planner = planner
        _sync_target_map_to_base(self.env, planner)

    def reset(self, *, seed=None, options=None):
        _sync_target_map_to_base(self.env, self.planner)
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})
        info["planner_output"] = self.planner
        info["target_map_id"] = self.planner.target_map_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info["planner_output"] = self.planner
        info["target_map_id"] = self.planner.target_map_id
        return obs, reward, terminated, truncated, info
