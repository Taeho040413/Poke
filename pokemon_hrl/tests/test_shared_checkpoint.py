"""Tests for shared-plan checkpoint coordination across vecenv agents."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from pokemon_hrl.env.interactive_env import HrlInteractiveRewardEnv
from pokemon_hrl.training.hrl_checkpoint import TrainingCheckpointCoordinator
from pokemon_hrl.training.shared_plan import SharedPlanStore


def _bare_hrl_env(*, env_id: int = 1) -> HrlInteractiveRewardEnv:
    env = object.__new__(HrlInteractiveRewardEnv)
    env.env_id = env_id
    env.pyboy = MagicMock()
    env.init_state_path = None
    env.init_state_name = None
    return env


def _puffer_agent(base: HrlInteractiveRewardEnv) -> SimpleNamespace:
    return SimpleNamespace(env=base)


def _trainer(vecenv, infos: dict, *, step: int = 10) -> MagicMock:
    trainer = MagicMock()
    trainer.vecenv = vecenv
    trainer.global_step = step
    trainer.uncompiled_policy = torch.nn.Linear(2, 2)
    trainer.config.save_checkpoint = False
    trainer.infos = infos
    return trainer


def test_subgoal_soft_sync_advances_shared_index(tmp_path):
    shared_plan = SharedPlanStore()
    cfg = OmegaConf.create(
        {"hrl": {"checkpoint": {"directory": str(tmp_path), "rollback_game_only": True}}}
    )
    coordinator = TrainingCheckpointCoordinator.from_config(cfg, shared_plan=shared_plan)

    base = _bare_hrl_env()
    vecenv = SimpleNamespace(envs=[_puffer_agent(base)])
    trainer = _trainer(
        vecenv,
        {
            "hrl_subgoal_event": [1],
            "hrl_subgoal_new_index": [2],
            "hrl_subgoal_success_env_id": [1],
        },
    )

    with patch(
        "pokemon_hrl.training.hrl_checkpoint.broadcast_subgoal_soft_sync"
    ) as soft_sync:
        coordinator.process_rollout(trainer)

    assert shared_plan.subgoal_index == 2
    soft_sync.assert_called_once_with(vecenv, 2, global_step=10)


def test_goal_success_uses_winner_save_state_and_hard_resets(tmp_path):
    shared_plan = SharedPlanStore()
    cfg = OmegaConf.create(
        {
            "hrl": {
                "checkpoint": {
                    "directory": str(tmp_path),
                    "save_game_state": True,
                    "save_policy": True,
                    "rollback_game_only": True,
                },
            },
        }
    )
    coordinator = TrainingCheckpointCoordinator.from_config(cfg, shared_plan=shared_plan)

    winner = _bare_hrl_env(env_id=2)
    other = _bare_hrl_env(env_id=3)
    vecenv = SimpleNamespace(envs=[_puffer_agent(other), _puffer_agent(winner)])
    trainer = _trainer(
        vecenv,
        {
            "hrl_progress_success": [1],
            "hrl_goal_save_state": [b"winner-state"],
            "hrl_goal_success_env_id": [2],
        },
        step=42,
    )

    with patch("pokemon_hrl.training.hrl_checkpoint.broadcast_hard_reset") as hard_reset:
        coordinator.process_rollout(trainer)

    assert coordinator.store.save_point_path is not None
    assert coordinator.store.save_point_path.read_bytes() == b"winner-state"
    assert shared_plan.subgoal_index == 0
    hard_reset.assert_called_once()
    assert winner.init_state_path == coordinator.store.save_point_path.resolve()
    assert other.init_state_path == coordinator.store.save_point_path.resolve()


def test_failure_rollback_broadcasts_to_all_agents(tmp_path):
    shared_plan = SharedPlanStore()
    cfg = OmegaConf.create(
        {
            "hrl": {
                "checkpoint": {
                    "directory": str(tmp_path),
                    "rollback_game_only": True,
                },
            },
        }
    )
    coordinator = TrainingCheckpointCoordinator.from_config(cfg, shared_plan=shared_plan)
    coordinator.store.save_game_state(b"rollback-me")

    agents = [_bare_hrl_env(env_id=i + 1) for i in range(3)]
    vecenv = SimpleNamespace(envs=[_puffer_agent(base) for base in agents])
    trainer = _trainer(vecenv, {"hrl_progress_failure": [1]}, step=11)

    coordinator.process_rollout(trainer)

    for base in agents:
        base.pyboy.load_state.assert_called_once()
