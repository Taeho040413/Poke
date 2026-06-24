"""Cap episode length per active HRL mode."""

from __future__ import annotations

import gymnasium as gym


class ModeMaxStepsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        if self._steps >= self.max_steps:
            truncated = True
            info = dict(info or {})
            info["truncated_reason"] = "mode_max_steps"
        return obs, reward, terminated, truncated, info
