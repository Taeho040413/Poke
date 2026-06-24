"""Tests for closed-loop orchestrator, checkpoint flags, and policy loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from pokemon_hrl.loop.checkpoint import CheckpointConfig, CheckpointManager
from pokemon_hrl.loop.orchestrator import HrlOrchestrator
from pokemon_hrl.loop.policy_loader import resolve_policy_checkpoint
from pokemon_hrl.types import Mode, PlannerOutput, ProgressResult, StateSummary, Subgoal, WorldState
from pokemon_hrl.world_state.merge import merge_extracted_state
from pokemon_hrl.world_state.store import WorldStateStore


def _minimal_planner() -> PlannerOutput:
    return PlannerOutput(
        subgoal=[Subgoal(success_criteria=["map_reached:1"])],
        hint={"target_map_id": 1},
        success_criteria=["flag:EVENT_BEAT_BROCK"],
        failure_criteria=["no_progress"],
    )


def test_checkpoint_manager_respects_save_flags(tmp_path: Path):
    store = WorldStateStore(tmp_path)
    env = MagicMock()
    env.pyboy.save_state.side_effect = lambda buf: buf.write(b"game-state")

    policy = torch.nn.Linear(2, 2)

    cfg = CheckpointConfig(
        save_game_state=False,
        save_policy=True,
        rollback_game_only=True,
        directory=tmp_path,
    )
    manager = CheckpointManager(store, cfg)
    manager.save_both(env, policy=policy)
    assert store.save_point_path is None
    assert store.policy_checkpoint_path is not None
    assert store.policy_checkpoint_path.exists()

    cfg2 = CheckpointConfig(
        save_game_state=True,
        save_policy=False,
        rollback_game_only=True,
        directory=tmp_path,
    )
    manager2 = CheckpointManager(store, cfg2)
    manager2.save_both(env, policy=policy)
    assert store.save_point_path is not None
    assert store.save_point_path.read_bytes() == b"game-state"


def test_checkpoint_rollback_skipped_when_disabled(tmp_path: Path):
    store = WorldStateStore(tmp_path)
    save_path = store.save_game_state(b"rollback-me")
    env = MagicMock()

    manager = CheckpointManager(
        store,
        CheckpointConfig(rollback_game_only=False, directory=tmp_path),
    )
    assert manager.rollback_game_only(env) is False
    env.pyboy.load_state.assert_not_called()

    manager_enabled = CheckpointManager(
        store,
        CheckpointConfig(rollback_game_only=True, directory=tmp_path),
    )
    assert manager_enabled.rollback_game_only(env) is True
    env.pyboy.load_state.assert_called_once()
    assert save_path.exists()


def test_store_bootstraps_existing_save_point(tmp_path: Path):
    save_path = tmp_path / "save_point.state"
    save_path.write_bytes(b"existing-save")
    store = WorldStateStore(tmp_path)
    assert store.save_point_path == save_path


def test_checkpoint_rollback_bootstraps_from_disk_without_prior_save(tmp_path: Path):
    save_path = tmp_path / "save_point.state"
    save_path.write_bytes(b"rollback-me")
    store = WorldStateStore(tmp_path)
    store.save_point_path = None
    env = MagicMock()
    manager = CheckpointManager(
        store,
        CheckpointConfig(rollback_game_only=True, directory=tmp_path),
    )
    assert manager.rollback_game_only(env) is True
    assert store.save_point_path == save_path
    env.pyboy.load_state.assert_called_once()


def test_resolve_policy_checkpoint_fresh_returns_none(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "train": {"data_dir": str(tmp_path), "exp_id": "missing-run"},
            "hrl": {"training": {"exp_id": "missing-run"}},
            "policies": {},
        }
    )
    assert resolve_policy_checkpoint(cfg, fresh=True) is None


def test_orchestrator_step_once_returns_obs(tmp_path: Path):
    planner = _minimal_planner()
    base_env = MagicMock()
    base_env.events = None
    base_env.get_game_coords.return_value = (0, 0, 1)

    env = MagicMock()
    env.env = None
    env.step.return_value = ({"screen": [0]}, 0.1, False, False, {"hrl_progress_success": 0})
    env.reset.return_value = ({"screen": [0]}, {})

    config = OmegaConf.create(
        {
            "hrl": {
                "planner": {"enabled": False, "call_on": "never"},
                "mode_selector": {"enabled": False, "forced_mode": "interactive"},
                "checkpoint": {
                    "directory": str(tmp_path),
                    "save_game_state": False,
                    "save_policy": False,
                    "rollback_game_only": False,
                },
                "training": {"exp_id": "test"},
            },
            "train": {"data_dir": str(tmp_path), "exp_id": "test"},
        }
    )

    with patch("pokemon_hrl.loop.orchestrator.unwrap_hrl_env", return_value=base_env):
        with patch("pokemon_hrl.loop.orchestrator.extract_world_state") as extract:
            extract.return_value = WorldState(map_id=1, x=0, y=0, badges=0)
            orchestrator = HrlOrchestrator(
                config=config,
                env=env,
                store=WorldStateStore(tmp_path),
                selector=MagicMock(select=MagicMock(return_value=Mode.INTERACTIVE)),
                summarizer=MagicMock(
                    summarize=MagicMock(return_value=StateSummary()),
                ),
                progress=MagicMock(),
                updater=MagicMock(),
                checkpoints=CheckpointManager(
                    WorldStateStore(tmp_path),
                    CheckpointConfig(
                        directory=tmp_path,
                        save_game_state=False,
                        save_policy=False,
                    ),
                ),
                planner=MagicMock(plan=MagicMock(return_value=planner)),
            )
            orchestrator.store.state.planner_output = planner
            orchestrator.store.set_planner_output(planner)

            with patch(
                "pokemon_hrl.loop.orchestrator.merge_extracted_state",
                side_effect=merge_extracted_state,
            ):
                with patch(
                    "pokemon_hrl.loop.orchestrator.progress_from_info",
                    return_value=ProgressResult(),
                ):
                    progress, obs, _info = orchestrator.step_once(0)

    assert isinstance(progress, ProgressResult)
    assert obs == {"screen": [0]}
    env.step.assert_called_once_with(0)
