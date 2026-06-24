"""Tests for HRL env unwrapping through Puffer wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pufferlib
import pytest

from pokemon_hrl.env.interactive_env import HrlInteractiveRewardEnv
from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.training.hrl_checkpoint import TrainingCheckpointCoordinator


def _bare_hrl_env() -> HrlInteractiveRewardEnv:
    return object.__new__(HrlInteractiveRewardEnv)


def test_unwrap_follows_puffer_gymnasium_env_chain():
    base = _bare_hrl_env()
    puffer = SimpleNamespace(env=base)
    assert unwrap_hrl_env(puffer) is base


def test_unwrap_rejects_puffer_dtype_namespace():
    ns = pufferlib.namespace(observation_dtype=np.dtype("u1"))
    with pytest.raises(TypeError, match="HrlInteractiveRewardEnv"):
        unwrap_hrl_env(ns)


def test_checkpoint_base_env_uses_driver_env_not_emulated(tmp_path):
    cfg = {
        "hrl": {
            "checkpoint": {
                "directory": str(tmp_path),
                "save_game_state": True,
                "save_policy": False,
                "rollback_game_only": True,
            },
        },
    }
    from omegaconf import OmegaConf

    coordinator = TrainingCheckpointCoordinator.from_config(OmegaConf.create(cfg))

    base = _bare_hrl_env()
    base.pyboy = MagicMock()
    base.pyboy.save_state.side_effect = lambda buf: buf.write(b"game-state")

    puffer = SimpleNamespace(env=base)
    puffer.emulated = pufferlib.namespace(observation_dtype=np.dtype("u1"))

    trainer = MagicMock()
    trainer.vecenv = SimpleNamespace(envs=[SimpleNamespace(env=base)])
    trainer.global_step = 1
    trainer.uncompiled_policy = None
    trainer.config.save_checkpoint = False
    trainer.infos = {"hrl_progress_success": [1], "hrl_goal_save_state": [b"game-state"]}

    coordinator.process_rollout(trainer)

    assert coordinator.store.save_point_path is not None
    assert coordinator.store.save_point_path.read_bytes() == b"game-state"
