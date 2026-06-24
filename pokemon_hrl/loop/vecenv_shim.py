"""Single-env vecenv shim so CleanPuffeRL can drive HrlOrchestrator rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OrchestratorVecShim:
    """Minimal pufferlib vecenv surface for one env stepped via HrlOrchestrator."""

    env: Any
    orchestrator: Any
    _obs: Any = None
    _reward: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=np.float32))
    _done: np.ndarray = field(default_factory=lambda: np.array([False], dtype=bool))
    _truncated: np.ndarray = field(default_factory=lambda: np.array([False], dtype=bool))
    _info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.driver_env = self.env
        self.emulated = getattr(self.env, "emulated", self.env)
        self.num_agents = 1
        self.agents_per_batch = 1
        obs_space = getattr(self.env, "single_observation_space", self.env.observation_space)
        atn_space = getattr(self.env, "single_action_space", self.env.action_space)
        self.single_observation_space = obs_space
        self.single_action_space = atn_space

    def _batch_obs(self, obs: Any) -> np.ndarray:
        """Match pufferlib vecenv recv: one row per agent, shape (1, *obs_shape)."""
        batched = np.asarray(obs)
        target_shape = (1, *self.single_observation_space.shape)
        if batched.shape != target_shape:
            batched = batched.reshape(target_shape)
        return batched

    def async_reset(self, seed: int | None = None) -> None:
        if self._obs is None:
            if seed is not None:
                self._obs, _ = self.env.reset(seed=seed)
            else:
                self._obs, _ = self.env.reset()
        self._reward = np.array([0.0], dtype=np.float32)
        self._done = np.array([False], dtype=bool)
        self._truncated = np.array([False], dtype=bool)
        self._info = {}

    def recv(self) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[dict], list[int], np.ndarray]:
        return (
            self._batch_obs(self._obs),
            self._reward,
            self._done,
            self._truncated,
            [dict(self._info)],
            [0],
            np.array([True], dtype=bool),
        )

    def send(self, actions: np.ndarray) -> None:
        action = int(np.asarray(actions).reshape(-1)[0])
        progress, obs, info = self.orchestrator.step_once(action)
        self._obs = obs
        self._reward = np.array([float(progress.reward)], dtype=np.float32)
        terminated = bool(progress.done or progress.truncated)
        self._done = np.array([terminated], dtype=bool)
        self._truncated = np.array([bool(progress.truncated)], dtype=bool)
        self._info = dict(info or {})
        if terminated:
            self._obs, _ = self.env.reset()
            self.orchestrator.on_episode_reset()

    def close(self) -> None:
        self.env.close()
