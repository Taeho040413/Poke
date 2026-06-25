"""Gym wrapper that runs ProgressCheck each step and exposes results in info."""

from __future__ import annotations

import io
from typing import Any

import gymnasium as gym

from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.mode.progress import (
    ProgressCheck,
    log_goal_event,
    log_subgoal_event,
    progress_to_info,
)
from pokemon_hrl.planner.logging import log_active_goal_state
from pokemon_hrl.mode.subgoal import current_subgoal
from pokemon_hrl.planner.criteria import (
    attach_hrl_obs,
    extend_observation_space,
    planner_signature,
    subgoal_label,
)
from pokemon_hrl.types import PlannerOutput, ProgressResult, WorldState
from pokemon_hrl.world_state.extractor import extract_world_state

GOAL_SUCCESS_REWARD = 5.0
SUBGOAL_SUCCESS_REWARD = 3.0
DEFAULT_REWARD_FLOOR = -10.0
DEFAULT_REWARD_FLOOR_ROLLBACK_PENALTY = -1.0


class ProgressCheckWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        planner: PlannerOutput,
        *,
        log_goal_events: bool = True,
        env_id: int = 0,
        goal_success_reward: float = GOAL_SUCCESS_REWARD,
        subgoal_success_reward: float = SUBGOAL_SUCCESS_REWARD,
        reward_floor: float | None = DEFAULT_REWARD_FLOOR,
        reward_floor_rollback_penalty: float = DEFAULT_REWARD_FLOOR_ROLLBACK_PENALTY,
    ):
        super().__init__(env)
        self.planner = planner
        self.log_goal_events = log_goal_events
        self.env_id = int(env_id)
        self.goal_success_reward = float(goal_success_reward)
        self.subgoal_success_reward = float(subgoal_success_reward)
        self.reward_floor = (
            None if reward_floor is None else float(reward_floor)
        )
        self.reward_floor_rollback_penalty = float(reward_floor_rollback_penalty)
        self.progress = ProgressCheck()
        self._before: WorldState | None = None
        self._step = 0
        self._logged_success = False
        self._logged_failure = False
        self._subgoal_index = 0
        self._reward_since_checkpoint = 0.0
        self.observation_space = extend_observation_space(self.env.observation_space)

    def set_planner(self, planner: PlannerOutput) -> None:
        if planner_signature(planner) != planner_signature(self.planner):
            self._subgoal_index = 0
        self.planner = planner

    def _sync_goal_context(self) -> None:
        base = self._base_env()
        setter = getattr(base, "set_goal_context", None)
        if callable(setter):
            from pokemon_hrl.env.goal_memory import goal_context_from_planner_dict

            setter(goal_context_from_planner_dict(self.planner, subgoal_index=self._subgoal_index))

    def _attach_obs(self, obs: Any) -> Any:
        if not isinstance(obs, dict):
            return obs
        return attach_hrl_obs(obs, self.planner, subgoal_index=self._subgoal_index)

    def sync_subgoal_index(self, index: int) -> None:
        self._subgoal_index = max(0, int(index))
        self._sync_goal_context()

    def sync_after_external_state_change(
        self,
        *,
        global_step: int | None = None,
        subgoal_index: int | None = None,
    ) -> None:
        if subgoal_index is not None:
            self._subgoal_index = max(0, int(subgoal_index))
        base = self._base_env()
        step = self._step if global_step is None else int(global_step)
        self._before = extract_world_state(base, global_step=step)
        self._reward_since_checkpoint = 0.0

    def _reset_reward_since_checkpoint(self) -> None:
        self._reward_since_checkpoint = 0.0

    def _apply_reward_floor(self, reward: float, merged: dict[str, Any]) -> float:
        self._reward_since_checkpoint += float(reward)
        merged["hrl_reward_since_checkpoint"] = self._reward_since_checkpoint
        if (
            self.reward_floor is not None
            and self._reward_since_checkpoint <= self.reward_floor
        ):
            merged["hrl_reward_floor_breach"] = 1
            reward += self.reward_floor_rollback_penalty
        return reward

    def _base_env(self):
        return unwrap_hrl_env(self.env)

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self._logged_success = False
        self._logged_failure = False
        self._subgoal_index = 0
        self._reset_reward_since_checkpoint()
        obs, info = self.env.reset(seed=seed, options=options)
        base = self._base_env()
        self.env_id = int(getattr(base, "env_id", self.env_id))
        self._before = extract_world_state(base, global_step=0)
        if self.log_goal_events:
            log_active_goal_state(
                self.planner,
                subgoal_index=self._subgoal_index,
                map_id=self._before.map_id,
                env_id=self.env_id,
            )
        info = dict(info or {})
        info.update(
            progress_to_info(
                ProgressResult(),
                self.planner,
                self._before,
                subgoal_index=self._subgoal_index,
            )
        )
        return self._attach_obs(obs), info

    def step(self, action):
        base = self._base_env()
        before = self._before if self._before is not None else extract_world_state(base, global_step=self._step)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1

        after = extract_world_state(base, global_step=self._step)
        progress = ProgressResult()
        subgoal_hit = False
        completed_subgoal = ""
        completed_index = -1

        active_subgoal = current_subgoal(self.planner.subgoal, self._subgoal_index)
        if active_subgoal and self.progress.check_subgoal_met(
            active_subgoal, self.planner, before, after
        ):
            subgoal_hit = True
            completed_subgoal = subgoal_label(active_subgoal)
            completed_index = self._subgoal_index
            if self.log_goal_events:
                log_subgoal_event(
                    completed_subgoal,
                    self.planner,
                    after,
                    env_id=self.env_id,
                    index=self._subgoal_index,
                )
            reward += self.subgoal_success_reward
            self._subgoal_index += 1
            self._sync_goal_context()

        if self.progress.all_subgoals_complete(self.planner, self._subgoal_index):
            goal_progress = self.progress.check_success(self.planner, before, after)
            if goal_progress.success:
                progress = goal_progress

        failure_progress = self.progress.check_failure(
            self.planner, before, after, info=info
        )
        if failure_progress.failure:
            progress = failure_progress

        if subgoal_hit and not (progress.success or progress.failure):
            progress = ProgressResult(
                subgoal_success=True,
                reason="subgoal_completed",
                subgoal=completed_subgoal,
                subgoal_index=completed_index,
            )
        elif subgoal_hit:
            progress.subgoal_success = True
            progress.subgoal = completed_subgoal
            progress.subgoal_index = completed_index

        merged = dict(info or {})
        trainer_losses = int(getattr(base, "_trainer_battle_loss_count", 0))
        if trainer_losses > 0:
            merged["hrl_trainer_battle_loss"] = trainer_losses
            base._trainer_battle_loss_count = 0
        merged.update(
            progress_to_info(
                progress,
                self.planner,
                after,
                subgoal_index=self._subgoal_index,
            )
        )

        rising_success = progress.success and not self._logged_success
        rising_failure = progress.failure and not self._logged_failure
        if rising_success:
            self._logged_success = True
        if rising_failure:
            self._logged_failure = True

        if self.log_goal_events and (rising_success or rising_failure):
            log_goal_event(progress, self.planner, after, env_id=self.env_id)

        if rising_success or rising_failure:
            merged["hrl_goal_event"] = 1
        if progress.subgoal_success:
            merged["hrl_subgoal_event"] = 1
            merged["hrl_subgoal_success_env_id"] = self.env_id
            merged["hrl_subgoal_new_index"] = self._subgoal_index

        if rising_success:
            merged["hrl_goal_success_env_id"] = self.env_id
            save_state = getattr(getattr(base, "pyboy", None), "save_state", None)
            if callable(save_state):
                state_buf = io.BytesIO()
                save_state(state_buf)
                merged["hrl_goal_save_state"] = state_buf.getvalue()
            reward += self.goal_success_reward
            terminated = True

        reward = self._apply_reward_floor(reward, merged)

        self._before = after
        return self._attach_obs(obs), reward, terminated, truncated, merged
