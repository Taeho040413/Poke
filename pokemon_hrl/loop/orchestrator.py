"""Closed-loop orchestrator (LLM optional via planner.enabled only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pokemon_hrl.env.interactive_env import HrlInteractiveRewardEnv
from pokemon_hrl.env.planner_sync import sync_planner_to_env, sync_subgoal_index_to_env
from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.loop.checkpoint import CheckpointConfig, CheckpointManager
from pokemon_hrl.mode.progress import ProgressCheck, progress_from_info
from pokemon_hrl.mode.selector import ModeSelector
from pokemon_hrl.planner.factory import build_planner
from pokemon_hrl.planner.replan import should_invoke_planner
from pokemon_hrl.summarizer.rule_based import RuleBasedSummarizer
from typing import TYPE_CHECKING

from pokemon_hrl.types import Mode, PlannerOutput, ProgressResult
from pokemon_hrl.update.information import UpdateInformation
from pokemon_hrl.world_state.extractor import extract_world_state
from pokemon_hrl.world_state.merge import merge_extracted_state
from pokemon_hrl.world_state.store import WorldStateStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pokemon_hrl.mode.agents.interactive import InteractiveModeAgent


@dataclass
class HrlOrchestrator:
    config: object
    env: object
    base_env: HrlInteractiveRewardEnv = field(init=False)
    store: WorldStateStore
    selector: ModeSelector
    summarizer: RuleBasedSummarizer
    progress: ProgressCheck
    updater: UpdateInformation
    checkpoints: CheckpointManager
    planner: object
    agent: InteractiveModeAgent | None = None
    global_step: int = 0
    last_mode: Mode = Mode.INTERACTIVE
    _rollback_skip_logged: bool = False

    def __post_init__(self) -> None:
        self.base_env = unwrap_hrl_env(self.env)
        self._bootstrap_store_from_env()

    @classmethod
    def from_config(
        cls,
        config,
        env,
        *,
        agent: InteractiveModeAgent | None = None,
    ) -> HrlOrchestrator:
        ckpt_cfg = config.hrl.checkpoint
        store = WorldStateStore(ckpt_cfg.directory)
        checkpoint_config = CheckpointConfig(
            save_game_state=bool(ckpt_cfg.get("save_game_state", True)),
            save_policy=bool(ckpt_cfg.get("save_policy", True)),
            rollback_game_only=bool(ckpt_cfg.get("rollback_game_only", True)),
            directory=ckpt_cfg.directory,
        )
        orchestrator = cls(
            config=config,
            env=env,
            store=store,
            selector=ModeSelector(
                enabled=bool(config.hrl.mode_selector.enabled),
                forced_mode=Mode(config.hrl.mode_selector.forced_mode),
            ),
            summarizer=RuleBasedSummarizer(),
            progress=ProgressCheck(),
            updater=UpdateInformation(),
            checkpoints=CheckpointManager(store, checkpoint_config),
            planner=build_planner(config),
            agent=agent,
        )
        orchestrator._bootstrap_policy_path_from_config()
        return orchestrator

    def _bootstrap_policy_path_from_config(self) -> None:
        from pokemon_hrl.loop.policy_loader import resolve_policy_checkpoint

        model_pt = resolve_policy_checkpoint(self.config)
        if model_pt is not None:
            self.store.save_policy_path(model_pt)

    def _call_on(self) -> str:
        return str(self.config.hrl.planner.get("call_on", "goal_check"))

    def _bootstrap_store_from_env(self) -> None:
        if not hasattr(self.base_env, "events"):
            self.env.reset()
        snap = extract_world_state(self.base_env, global_step=0)
        self.store.replace(merge_extracted_state(self.store.state, snap))

    def _invoke_planner(
        self,
        *,
        initial: bool = False,
        progress: ProgressResult | None = None,
        info: dict | None = None,
    ) -> PlannerOutput:
        progress = progress or ProgressResult()
        if not should_invoke_planner(
            self._call_on(), progress, info=info, initial=initial
        ):
            if self.store.state.planner_output is None:
                raise RuntimeError(
                    "Planner was not invoked (call_on policy) but no planner_output in store"
                )
            return self.store.state.planner_output

        state = self.store.state
        summary = self.summarizer.summarize(state)
        self.store.set_recent_summary(summary)
        planner_output = self.planner.plan(summary, state)
        self.store.set_planner_output(planner_output)
        sync_planner_to_env(self.env, planner_output)
        return planner_output

    def _sync_progress_wrapper_after_rollback(self, *, global_step: int) -> None:
        goal_index = int(self.store.state.goal_stack.get("current_index", 0))
        current = self.env
        while current is not None:
            sync_hook = getattr(current, "sync_after_external_state_change", None)
            if callable(sync_hook):
                sync_hook(global_step=global_step, subgoal_index=goal_index)
                return
            current = getattr(current, "env", None)

    def step_once(self, action: int):
        before_snap = extract_world_state(self.base_env, global_step=self.global_step)
        self.store.replace(merge_extracted_state(self.store.state, before_snap))

        planner = self.store.state.planner_output
        if planner is None:
            planner = self._invoke_planner(initial=True)

        obs, reward, terminated, truncated, info = self.env.step(action)
        step_index = self.global_step + 1

        after_snap = extract_world_state(self.base_env, global_step=step_index)
        after_state = merge_extracted_state(self.store.state, after_snap)
        self.store.replace(after_state)

        self.last_mode = self.selector.select(after_state)

        progress = progress_from_info(info)
        if progress is None:
            progress = self.progress.check_criteria(
                planner, before_snap, after_snap, info=info
            )

        progress.reward = float(reward)
        progress.done = bool(terminated)
        progress.truncated = bool(truncated)

        self.updater.apply(
            self.store,
            self.base_env,
            progress=progress,
            global_step=step_index,
            info=info,
        )
        self._invoke_planner(progress=progress, info=info)
        goal_index = int(self.store.state.goal_stack.get("current_index", 0))
        sync_subgoal_index_to_env(self.env, goal_index)

        if progress.success:
            policy = self.agent.to_policy() if self.agent is not None else None
            self.checkpoints.save_both(
                self.base_env,
                policy=policy,
                policy_path=self.store.policy_checkpoint_path,
            )
            self._rollback_skip_logged = False
        if progress.failure:
            if self.checkpoints.rollback_game_only(self.base_env):
                self._on_game_rollback()
                self._sync_progress_wrapper_after_rollback(global_step=step_index)
                self._rollback_skip_logged = False
            elif not self._rollback_skip_logged:
                logger.warning("Rollback skipped: no save point on disk yet")
                self._rollback_skip_logged = True
        elif int(info.get("hrl_reward_floor_breach", 0)) > 0:
            if self.checkpoints.rollback_game_only(self.base_env):
                self._on_game_rollback()
                self._sync_progress_wrapper_after_rollback(global_step=step_index)
                self._rollback_skip_logged = False
                logger.info(
                    "Reward floor breach — rolled back to save point (policy unchanged)"
                )
            elif not self._rollback_skip_logged:
                logger.warning("Reward floor rollback skipped: no save point on disk yet")
                self._rollback_skip_logged = True

        self.global_step = step_index
        return progress, obs, info

    def select_action(self, obs, *, deterministic: bool = False) -> int:
        if self.agent is not None:
            return self.agent.act(obs, deterministic=deterministic)
        return int(self.env.action_space.sample())

    def on_episode_reset(self) -> None:
        """Align World State DB subgoal index with env after terminal reset."""
        if self.store.state.planner_output is not None:
            self.store.state.goal_stack = dict(self.store.state.goal_stack)
            self.store.state.goal_stack["current_index"] = 0
        sync_subgoal_index_to_env(self.env, 0)

    def _on_game_rollback(self) -> None:
        if self.store.state.planner_output is not None:
            self.store.state.goal_stack = dict(self.store.state.goal_stack)
            self.store.state.goal_stack["current_index"] = 0
        sync_subgoal_index_to_env(self.env, 0)
