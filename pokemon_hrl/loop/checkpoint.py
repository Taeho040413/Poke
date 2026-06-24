"""Checkpoint helpers — game save point + policy path."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from pokemon_hrl.world_state.store import WorldStateStore

SAVE_POINT_NAME = "save_point.state"


@dataclass
class CheckpointConfig:
    save_game_state: bool = True
    save_policy: bool = True
    rollback_game_only: bool = True
    directory: str | Path = "checkpoints"


@dataclass
class CheckpointManager:
    store: WorldStateStore
    config: CheckpointConfig

    def save_both(
        self,
        env,
        *,
        policy: nn.Module | None = None,
        policy_path: Path | None = None,
    ) -> None:
        if self.config.save_game_state:
            buffer = io.BytesIO()
            env.pyboy.save_state(buffer)
            self.store.save_game_state(buffer.getvalue())

        if not self.config.save_policy:
            return

        if policy is not None:
            path = self.store.checkpoint_dir / "policy_checkpoint.pt"
            torch.save(policy.state_dict(), path)
            self.store.save_policy_path(path)
        elif policy_path is not None:
            self.store.save_policy_path(policy_path)

    def _resolve_save_point_path(self) -> Path | None:
        if self.store.save_point_path is not None and self.store.save_point_path.is_file():
            return self.store.save_point_path
        bootstrapped = self.store.bootstrap_save_point()
        return bootstrapped

    def rollback_game_only(self, env) -> bool:
        if not self.config.rollback_game_only:
            return False
        path = self._resolve_save_point_path()
        if path is None:
            return False
        with open(path, "rb") as state_file:
            env.pyboy.load_state(state_file)
        return True
