"""Tests for goal-success save points and model_latest resume."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from pokemon_hrl.env.interactive_env import HrlInteractiveRewardEnv
from pokemon_hrl.training.checkpoint import find_latest_saved_model
from pokemon_hrl.training.hrl_checkpoint import (
    TrainingCheckpointCoordinator,
    apply_save_point_to_env_config,
    resolve_resume_game_state,
    run_dir_game_checkpoint_path,
)


def test_find_latest_saved_model_prefers_model_latest(tmp_path: Path):
    run_dir = tmp_path / "hrl-interactive-v1"
    run_dir.mkdir()
    older = run_dir / "model_000010.pt"
    latest = run_dir / "model_latest.pt"
    older.write_bytes(b"old")
    latest.write_bytes(b"latest")
    older.touch()
    latest.touch()

    picked = find_latest_saved_model(tmp_path, "hrl-interactive-v1")
    assert picked == latest


def test_apply_save_point_to_env_config(tmp_path: Path):
    save_path = tmp_path / "save_point.state"
    save_path.write_bytes(b"pyboy-state")
    cfg = OmegaConf.create(
        {
            "env": {"init_state": "red", "init_state_path": "assets/red.state"},
            "hrl": {"checkpoint": {"directory": str(tmp_path)}},
        }
    )
    applied = apply_save_point_to_env_config(cfg, fresh=False)
    assert applied == save_path
    assert cfg.env.init_state_path == str(save_path.resolve())


def test_apply_save_point_skipped_when_fresh(tmp_path: Path):
    save_path = tmp_path / "save_point.state"
    save_path.write_bytes(b"pyboy-state")
    cfg = OmegaConf.create(
        {
            "env": {"init_state": "red", "init_state_path": "assets/red.state"},
            "hrl": {"checkpoint": {"directory": str(tmp_path)}},
        }
    )
    assert apply_save_point_to_env_config(cfg, fresh=True) is None
    assert cfg.env.init_state_path == "assets/red.state"


def test_apply_run_game_checkpoint_on_resume(tmp_path: Path):
    run_dir = tmp_path / "hrl-interactive-v1"
    run_dir.mkdir()
    model_pt = run_dir / "model_latest.pt"
    model_pt.write_bytes(b"model")
    game_path = run_dir / "game_latest.state"
    game_path.write_bytes(b"in-progress-game")

    cfg = OmegaConf.create(
        {
            "env": {"init_state": "red", "init_state_path": "assets/red.state"},
            "hrl": {"checkpoint": {"directory": str(tmp_path / "checkpoints")}},
        }
    )
    applied = apply_save_point_to_env_config(cfg, fresh=False, resume_model_pt=model_pt)
    assert applied == game_path
    assert cfg.env.init_state_path == str(game_path.resolve())
    assert cfg.env.init_state == "game_latest"


def test_goal_save_point_takes_priority_over_run_game(tmp_path: Path):
    run_dir = tmp_path / "runs" / "hrl-interactive-v1"
    run_dir.mkdir(parents=True)
    model_pt = run_dir / "model_latest.pt"
    model_pt.write_bytes(b"model")
    (run_dir / "game_latest.state").write_bytes(b"run-game")

    goal_dir = tmp_path / "checkpoints"
    goal_dir.mkdir()
    goal_path = goal_dir / "save_point.state"
    goal_path.write_bytes(b"goal-game")

    cfg = OmegaConf.create(
        {
            "env": {"init_state": "red", "init_state_path": "assets/red.state"},
            "hrl": {"checkpoint": {"directory": str(goal_dir)}},
        }
    )
    applied = apply_save_point_to_env_config(cfg, fresh=False, resume_model_pt=model_pt)
    assert applied == goal_path
    assert cfg.env.init_state_path == str(goal_path.resolve())


def test_resolve_resume_game_state(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_pt = run_dir / "model_latest.pt"
    model_pt.write_bytes(b"x")
    assert resolve_resume_game_state(model_pt) is None
    game_path = run_dir_game_checkpoint_path(model_pt)
    game_path.write_bytes(b"game")
    assert resolve_resume_game_state(model_pt) == game_path


def _bare_hrl_env() -> HrlInteractiveRewardEnv:
    env = object.__new__(HrlInteractiveRewardEnv)
    env.env_id = 1
    env.pyboy = MagicMock()
    env.init_state_path = None
    env.init_state_name = None
    return env


def test_training_checkpoint_saves_on_goal_success(tmp_path: Path):
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
    coordinator = TrainingCheckpointCoordinator.from_config(cfg)

    base_env = _bare_hrl_env()
    policy = torch.nn.Linear(2, 2)
    trainer = MagicMock()
    trainer.vecenv = SimpleNamespace(envs=[SimpleNamespace(env=base_env)])
    trainer.global_step = 42
    trainer.uncompiled_policy = policy
    trainer.config.save_checkpoint = False
    trainer.infos = {
        "hrl_progress_success": [1],
        "hrl_goal_save_state": [b"game-state"],
        "hrl_goal_success_env_id": [1],
    }

    coordinator.process_rollout(trainer)

    assert coordinator.store.save_point_path is not None
    assert coordinator.store.save_point_path.read_bytes() == b"game-state"
    assert coordinator.store.policy_checkpoint_path is not None
    assert base_env.init_state_path == coordinator.store.save_point_path.resolve()


def test_training_checkpoint_rollback_on_failure(tmp_path: Path):
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
    coordinator = TrainingCheckpointCoordinator.from_config(cfg)
    coordinator.store.save_game_state(b"rollback-me")

    base_env = _bare_hrl_env()
    trainer = MagicMock()
    trainer.vecenv = SimpleNamespace(envs=[SimpleNamespace(env=base_env)])
    trainer.global_step = 10
    trainer.config.save_checkpoint = False
    trainer.infos = {"hrl_progress_failure": [1]}

    coordinator.process_rollout(trainer)
    base_env.pyboy.load_state.assert_called_once()


def test_training_checkpoint_rollback_on_reward_floor(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "hrl": {
                "checkpoint": {
                    "directory": str(tmp_path),
                    "save_game_state": True,
                    "save_policy": True,
                    "rollback_game_only": True,
                    "reward_floor": -10.0,
                },
            },
        }
    )
    coordinator = TrainingCheckpointCoordinator.from_config(cfg)
    coordinator.store.save_game_state(b"rollback-me")

    base_env = _bare_hrl_env()
    trainer = MagicMock()
    trainer.vecenv = SimpleNamespace(envs=[SimpleNamespace(env=base_env)])
    trainer.global_step = 11
    trainer.config.save_checkpoint = False
    trainer.infos = {"hrl_reward_floor_breach": [1]}

    coordinator.process_rollout(trainer)
    base_env.pyboy.load_state.assert_called_once()


def _minimal_cleanrl_trainer(tmp_path: Path, **config_overrides):
    pytest.importorskip("pufferlib")
    from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL

    run_dir = tmp_path / "test-exp"
    run_dir.mkdir()
    policy = torch.nn.Linear(2, 2)
    config = SimpleNamespace(
        data_dir=str(tmp_path),
        exp_id="test-exp",
        save_on_exit=False,
        save_policy_on_interrupt=True,
        verbose=False,
        **config_overrides,
    )
    vecenv = MagicMock()
    trainer = object.__new__(CleanPuffeRL)
    trainer.config = config
    trainer.exp_name = "test-exp"
    trainer.uncompiled_policy = policy
    trainer.vecenv = vecenv
    trainer.wandb_client = None
    trainer.global_step = 999
    trainer.epoch = 50
    trainer.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return trainer


def test_keyboard_interrupt_saves_policy_only_not_resume_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "test-exp"
    run_dir.mkdir()
    existing_latest = run_dir / "model_latest.pt"
    existing_latest.write_bytes(b"goal-checkpoint")
    existing_game = run_dir / "game_latest.state"
    existing_game.write_bytes(b"goal-game")
    existing_trainer = run_dir / "trainer_state.pt"
    existing_trainer.write_bytes(b"trainer")

    trainer = _minimal_cleanrl_trainer(tmp_path)
    trainer.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert (run_dir / "model_interrupt.pt").is_file()
    assert existing_latest.read_bytes() == b"goal-checkpoint"
    assert existing_game.read_bytes() == b"goal-game"
    assert existing_trainer.read_bytes() == b"trainer"
    trainer.vecenv.close.assert_called_once()


def test_normal_exit_without_save_on_exit_preserves_resume_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "test-exp"
    run_dir.mkdir()
    existing_latest = run_dir / "model_latest.pt"
    existing_latest.write_bytes(b"goal-checkpoint")

    trainer = _minimal_cleanrl_trainer(tmp_path, save_on_exit=False)
    trainer.__exit__(None, None, None)

    assert existing_latest.read_bytes() == b"goal-checkpoint"
    assert not (run_dir / "model_interrupt.pt").exists()


def test_normal_exit_with_save_on_exit_updates_latest(tmp_path: Path):
    run_dir = tmp_path / "test-exp"
    run_dir.mkdir()
    stale = run_dir / "model_latest.pt"
    stale.write_bytes(b"old")

    trainer = _minimal_cleanrl_trainer(tmp_path, save_on_exit=True)
    trainer.__exit__(None, None, None)

    assert stale.is_file()
    assert stale.read_bytes() != b"old"
    loaded = torch.load(stale, weights_only=False)
    assert isinstance(loaded, torch.nn.Linear)
