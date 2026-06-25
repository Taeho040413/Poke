from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym

from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.training.shared_plan import SharedPlanStore


class SharedProgressFiles:
    def __init__(
        self,
        directory: str | Path,
        *,
        state_name: str = "shared_goal.state",
        meta_name: str = "shared_progress.json",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / state_name
        self.meta_path = self.directory / meta_name

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def read_meta(self) -> dict[str, Any]:
        if not self.meta_path.is_file():
            return {"subgoal_index": 0, "state_version": 0}
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"subgoal_index": 0, "state_version": 0}
        return {
            "subgoal_index": int(data.get("subgoal_index", 0)),
            "state_version": int(data.get("state_version", 0)),
        }

    def write_meta(self, *, subgoal_index: int | None = None, bump_state_version: bool = False) -> dict[str, Any]:
        meta = self.read_meta()
        if subgoal_index is not None:
            meta["subgoal_index"] = max(0, int(subgoal_index))
        if bump_state_version:
            meta["state_version"] = int(meta.get("state_version", 0)) + 1
        self._atomic_write_text(self.meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def read_subgoal_index(self) -> int:
        return int(self.read_meta().get("subgoal_index", 0))

    def advance_subgoal_to(self, index: int) -> int:
        meta = self.read_meta()
        next_index = max(int(meta.get("subgoal_index", 0)), int(index))
        self.write_meta(subgoal_index=next_index)
        return next_index

    def reset_subgoal_index(self) -> None:
        self.write_meta(subgoal_index=0)

    def save_state(self, state_bytes: bytes) -> int:
        self._atomic_write_bytes(self.state_path, bytes(state_bytes))
        meta = self.write_meta(subgoal_index=0, bump_state_version=True)
        return int(meta["state_version"])

    def load_state(self) -> bytes | None:
        if not self.state_path.is_file():
            return None
        try:
            return self.state_path.read_bytes()
        except Exception:
            return None


class PersistentProgressWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        directory: str | Path,
        shared_plan: SharedPlanStore | None = None,
        state_name: str = "shared_goal.state",
        meta_name: str = "shared_progress.json",
        load_on_reset: bool = True,
        rollback_on_failure: bool = True,
        rollback_on_reward_floor: bool = True,
    ):
        super().__init__(env)
        self.files = SharedProgressFiles(directory, state_name=state_name, meta_name=meta_name)
        self.shared_plan = shared_plan
        self.load_on_reset = bool(load_on_reset)
        self.rollback_on_failure = bool(rollback_on_failure)
        self.rollback_on_reward_floor = bool(rollback_on_reward_floor)
        self._step = 0
        self._loaded_state_version = 0

    def _base_env(self):
        return unwrap_hrl_env(self.env)

    def _sync_progress_wrapper(self, *, subgoal_index: int) -> None:
        current = self.env
        while current is not None:
            sync = getattr(current, "sync_after_external_state_change", None)
            if callable(sync):
                sync(global_step=self._step, subgoal_index=max(0, int(subgoal_index)))
                return
            current = getattr(current, "env", None)

    def _fresh_obs(self, fallback_obs):
        base = self._base_env()
        get_obs = getattr(base, "_get_obs", None)
        if not callable(get_obs):
            return fallback_obs

        obs = get_obs()
        attach = getattr(self.env, "_attach_obs", None)
        if callable(attach):
            obs = attach(obs)
        return obs

    def _load_shared_state_into_env(self) -> bool:
        state_bytes = self.files.load_state()
        if not state_bytes:
            return False

        base = self._base_env()
        pyboy = getattr(base, "pyboy", None)
        load_state = getattr(pyboy, "load_state", None)
        if not callable(load_state):
            return False

        load_state(io.BytesIO(state_bytes))
        self._loaded_state_version = int(self.files.read_meta().get("state_version", 0))
        return True

    def _save_current_state(self, info: dict[str, Any]) -> bool:
        state_bytes = info.get("hrl_goal_save_state")

        if not isinstance(state_bytes, (bytes, bytearray)):
            base = self._base_env()
            pyboy = getattr(base, "pyboy", None)
            save_state = getattr(pyboy, "save_state", None)
            if not callable(save_state):
                return False
            buf = io.BytesIO()
            save_state(buf)
            state_bytes = buf.getvalue()

        self._loaded_state_version = self.files.save_state(bytes(state_bytes))
        if self.shared_plan is not None:
            self.shared_plan.reset_progress()
        return True

    def reset(self, *, seed=None, options=None):
        self._step = 0
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})

        loaded = False
        if self.load_on_reset:
            loaded = self._load_shared_state_into_env()

        subgoal_index = self.files.read_subgoal_index()
        if self.shared_plan is not None:
            subgoal_index = max(subgoal_index, int(self.shared_plan.subgoal_index))

        self._sync_progress_wrapper(subgoal_index=subgoal_index)
        obs = self._fresh_obs(obs)

        info["hrl_shared_subgoal_index"] = subgoal_index
        if loaded:
            info["hrl_loaded_shared_state"] = 1
            info["hrl_shared_state_version"] = self._loaded_state_version

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        info = dict(info or {})

        if int(info.get("hrl_subgoal_event", 0)) > 0:
            next_index = int(info.get("hrl_subgoal_new_index", info.get("hrl_subgoal_index", 0)))
            shared_index = self.files.advance_subgoal_to(next_index)
            if self.shared_plan is not None:
                shared_index = self.shared_plan.advance_subgoal_to(shared_index)
            self._sync_progress_wrapper(subgoal_index=shared_index)
            info["hrl_shared_subgoal_index"] = shared_index

        if int(info.get("hrl_progress_success", 0)) > 0:
            if self._save_current_state(info):
                info["hrl_shared_state_saved"] = 1
                info["hrl_shared_state_version"] = self._loaded_state_version

        rollback = False
        if self.rollback_on_failure and int(info.get("hrl_progress_failure", 0)) > 0:
            rollback = True
        if self.rollback_on_reward_floor and int(info.get("hrl_reward_floor_breach", 0)) > 0:
            rollback = True

        if rollback and self._load_shared_state_into_env():
            subgoal_index = self.files.read_subgoal_index()
            if self.shared_plan is not None:
                subgoal_index = max(subgoal_index, int(self.shared_plan.subgoal_index))
            self._sync_progress_wrapper(subgoal_index=subgoal_index)
            obs = self._fresh_obs(obs)
            terminated = False
            truncated = False
            info["hrl_shared_state_rollback"] = 1
            info["hrl_shared_subgoal_index"] = subgoal_index
            info["hrl_shared_state_version"] = self._loaded_state_version

        return obs, reward, terminated, truncated, info