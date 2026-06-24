"""Tests for OrchestratorVecShim."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from pokemon_hrl.loop.vecenv_shim import OrchestratorVecShim
from pokemon_hrl.types import ProgressResult


def test_orchestrator_vec_shim_recv_send_roundtrip():
    obs0 = np.zeros(4, dtype=np.uint8)
    obs1 = np.ones(4, dtype=np.uint8)
    env = MagicMock()
    env.reset.return_value = (obs1, {})
    env.observation_space.shape = (4,)
    env.action_space.n = 7
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space

    orchestrator = MagicMock()
    orchestrator.step_once.return_value = (
        ProgressResult(reward=0.5),
        obs0,
        {"hrl_progress_success": 0},
    )

    shim = OrchestratorVecShim(env, orchestrator)
    shim._obs = obs0
    shim.async_reset()

    recv_obs, reward, done, truncated, info, env_id, mask = shim.recv()
    assert recv_obs.shape == (1, 4)
    assert np.array_equal(recv_obs[0], obs0)
    assert reward[0] == 0.0
    assert not done[0]

    shim.send(np.array([3]))
    orchestrator.step_once.assert_called_once_with(3)
    assert np.array_equal(shim._reward, np.array([0.5], dtype=np.float32))

    recv_obs2, reward2, done2, *_ = shim.recv()
    assert recv_obs2.shape == (1, 4)
    assert np.array_equal(recv_obs2[0], obs0)
    assert reward2[0] == 0.5
    assert not done2[0]
