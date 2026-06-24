"""Goal-success save points and rollout checkpoint hooks for training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pokemon_hrl.loop.checkpoint import CheckpointConfig, CheckpointManager, SAVE_POINT_NAME
from pokemon_hrl.training.shared_plan import SharedPlanStore, get_shared_plan_store
from pokemon_hrl.training.vecenv_utils import (
    agent_index_for_env_id,
    base_env_for_agent,
    broadcast_hard_reset,
    broadcast_subgoal_soft_sync,
    iter_vecenv_agents,
    pin_save_point_on_all_agents,
)
from pokemon_hrl.world_state.store import WorldStateStore

GAME_CHECKPOINT_LATEST = "game_latest.state"


def run_dir_game_checkpoint_path(model_pt: Path) -> Path:
    return model_pt.expanduser().resolve().parent / GAME_CHECKPOINT_LATEST


def resolve_resume_game_state(model_pt: Path | None) -> Path | None:
    if model_pt is None:
        return None
    path = run_dir_game_checkpoint_path(model_pt)
    return path if path.is_file() else None


def save_point_path(config: DictConfig) -> Path:
    directory = Path(OmegaConf.select(config, "hrl.checkpoint.directory", default="checkpoints"))
    return directory / SAVE_POINT_NAME


def apply_save_point_to_env_config(
    config: DictConfig,
    *,
    fresh: bool = False,
    resume_model_pt: Path | None = None,
) -> Path | None:
    """Use goal-success save point or run-dir game checkpoint as env init state."""
    if fresh:
        return None
    path = save_point_path(config)
    if path.is_file():
        config.env.init_state_path = str(path.resolve())
        config.env.init_state = path.stem
        print(f"[checkpoint] goal 세이브 포인트에서 시작: {path.resolve()}", flush=True)
        return path

    game_path = resolve_resume_game_state(resume_model_pt)
    if game_path is not None:
        config.env.init_state_path = str(game_path.resolve())
        config.env.init_state = game_path.stem
        print(f"[checkpoint] run 게임 세이브에서 시작: {game_path.resolve()}", flush=True)
        return game_path
    return None


def _driver_env(vecenv: Any):
    agents = list(iter_vecenv_agents(vecenv))
    if not agents:
        return vecenv
    return agents[0]


def _any_positive(infos: dict[str, list], key: str) -> bool:
    if key not in infos:
        return False
    return any(int(v) > 0 for v in infos[key])


def _first_goal_save(infos: dict[str, list]) -> tuple[bytes | None, int | None]:
    states = infos.get("hrl_goal_save_state")
    env_ids = infos.get("hrl_goal_success_env_id")
    if not states:
        return None, None
    state = states[0]
    if isinstance(state, (bytes, bytearray)):
        winner_id = int(env_ids[0]) if env_ids else None
        return bytes(state), winner_id
    return None, None


def _max_subgoal_index_from_events(infos: dict[str, list]) -> int | None:
    indices = infos.get("hrl_subgoal_new_index")
    if not indices:
        return None
    return max(int(v) for v in indices)


@dataclass
class TrainingCheckpointCoordinator:
    config: DictConfig
    store: WorldStateStore
    checkpoints: CheckpointManager
    shared_plan: SharedPlanStore = field(default_factory=get_shared_plan_store)
    _last_goal_save_step: int = -1
    _last_subgoal_sync_index: int = -1

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        *,
        store: WorldStateStore | None = None,
        shared_plan: SharedPlanStore | None = None,
    ) -> TrainingCheckpointCoordinator:
        ckpt_cfg = config.hrl.checkpoint
        directory = Path(ckpt_cfg.directory)
        store = store or WorldStateStore(directory)
        checkpoint_config = CheckpointConfig(
            save_game_state=bool(ckpt_cfg.get("save_game_state", True)),
            save_policy=bool(ckpt_cfg.get("save_policy", True)),
            rollback_game_only=bool(ckpt_cfg.get("rollback_game_only", True)),
            directory=directory,
        )
        return cls(
            config=config,
            store=store,
            checkpoints=CheckpointManager(store, checkpoint_config),
            shared_plan=shared_plan or get_shared_plan_store(),
        )

    def _base_env(self, vecenv: Any, *, agent_index: int = 0):
        return base_env_for_agent(vecenv, agent_index)

    def _resolve_save_point(self) -> Path | None:
        if self.store.save_point_path is not None and self.store.save_point_path.is_file():
            return self.store.save_point_path
        return self.store.bootstrap_save_point()

    def _save_goal_checkpoint(
        self,
        trainer: Any,
        *,
        game_state: bytes | None,
        winner_env_id: int | None,
    ) -> None:
        vecenv = trainer.vecenv
        policy = getattr(trainer, "uncompiled_policy", None)

        if game_state is not None:
            self.store.save_game_state(game_state)
        else:
            agent_index = agent_index_for_env_id(vecenv, winner_env_id)
            base = self._base_env(vecenv, agent_index=agent_index)
            self.checkpoints.save_both(base, policy=policy)
            save_path = self._resolve_save_point()
            if save_path is not None:
                pin_save_point_on_all_agents(vecenv, save_path)
            return

        if policy is not None and self.checkpoints.config.save_policy:
            path = self.store.checkpoint_dir / "policy_checkpoint.pt"
            import torch

            torch.save(policy.state_dict(), path)
            self.store.save_policy_path(path)

        save_path = self._resolve_save_point()
        if save_path is not None:
            pin_save_point_on_all_agents(vecenv, save_path)

    def _process_subgoal_soft_sync(self, trainer: Any, infos: dict[str, list]) -> bool:
        if not _any_positive(infos, "hrl_subgoal_event"):
            return False

        new_index = _max_subgoal_index_from_events(infos)
        if new_index is None:
            return False
        if new_index <= self.shared_plan.subgoal_index:
            return False
        if new_index == self._last_subgoal_sync_index:
            return False

        step = int(getattr(trainer, "global_step", 0))
        synced = self.shared_plan.advance_subgoal_to(new_index)
        self._last_subgoal_sync_index = synced
        broadcast_subgoal_soft_sync(
            trainer.vecenv,
            synced,
            global_step=step,
        )
        print(
            f"[checkpoint] subgoal 달성 — soft sync (index={synced}, step={step})",
            flush=True,
        )
        return True

    def process_rollout(self, trainer: Any) -> None:
        infos = getattr(trainer, "infos", None)
        if not isinstance(infos, dict) or not infos:
            return

        step = int(getattr(trainer, "global_step", 0))
        vecenv = trainer.vecenv

        self._process_subgoal_soft_sync(trainer, infos)

        goal_success = _any_positive(infos, "hrl_progress_success")
        goal_failure = _any_positive(infos, "hrl_progress_failure")
        reward_floor_breach = _any_positive(infos, "hrl_reward_floor_breach")
        if not goal_success and not goal_failure and not reward_floor_breach:
            return

        if goal_success:
            if step == self._last_goal_save_step:
                return
            self._last_goal_save_step = step

            game_state, winner_env_id = _first_goal_save(infos)
            self._save_goal_checkpoint(
                trainer,
                game_state=game_state,
                winner_env_id=winner_env_id,
            )
            self.shared_plan.reset_progress()
            self._last_subgoal_sync_index = 0

            save_path = self._resolve_save_point()
            broadcast_hard_reset(
                vecenv,
                save_path,
                global_step=step,
                subgoal_index=0,
            )
            winner_label = "?" if winner_env_id is None else str(winner_env_id)
            print(
                f"[checkpoint] goal 성공 — env {winner_label} 기준 승격 + 전 env reset "
                f"(step={step})",
                flush=True,
            )
            if bool(getattr(trainer.config, "save_checkpoint", True)):
                trainer.save_checkpoint(force=True)
            return

        if not self.checkpoints.config.rollback_game_only:
            return

        save_path = self._resolve_save_point()
        if save_path is None:
            return

        broadcast_hard_reset(
            vecenv,
            save_path,
            global_step=step,
            subgoal_index=0,
        )
        self.shared_plan.reset_progress()
        self._last_subgoal_sync_index = 0

        if reward_floor_breach:
            print(
                f"[checkpoint] 누적 리워드 {self._reward_floor_label()} 이하 "
                f"— 전 env 세이브 포인트 롤백 (step={step})",
                flush=True,
            )
        else:
            print(
                f"[checkpoint] goal 실패 — 전 env 세이브 포인트 롤백 (step={step})",
                flush=True,
            )

    def _reward_floor_label(self) -> str:
        raw = OmegaConf.select(self.config, "hrl.checkpoint.reward_floor", default=-10.0)
        return str(raw)


def infos_from_trainer(trainer: Any) -> dict[str, list]:
    raw = getattr(trainer, "infos", None)
    if isinstance(raw, defaultdict):
        return dict(raw)
    if isinstance(raw, dict):
        return raw
    return {}
