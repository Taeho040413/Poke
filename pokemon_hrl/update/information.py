"""Apply Mode Layer results to World State DB."""

from __future__ import annotations

from dataclasses import dataclass

from pokemon_hrl.planner.criteria import planner_goal_key, subgoal_label
from pokemon_hrl.types import ProgressResult
from pokemon_hrl.world_state.extractor import extract_world_state
from pokemon_hrl.world_state.merge import merge_extracted_state
from pokemon_hrl.world_state.store import WorldStateStore


@dataclass
class UpdateInformation:
    def apply(
        self,
        store: WorldStateStore,
        env,
        *,
        progress: ProgressResult,
        global_step: int,
        info: dict | None = None,
    ):
        extracted = extract_world_state(env, global_step=global_step)
        merged = merge_extracted_state(store.state, extracted)
        goal_key = (
            planner_goal_key(merged.planner_output)
            if merged.planner_output is not None
            else ""
        )

        if progress.subgoal_success and merged.planner_output is not None:
            planner = merged.planner_output
            subgoal_key = progress.subgoal
            if not subgoal_key and 0 <= progress.subgoal_index < len(planner.subgoal):
                subgoal_key = subgoal_label(planner.subgoal[progress.subgoal_index])
            if not subgoal_key:
                subgoal_key = goal_key
            store.record_success(
                goal_key, subgoal_key, step=global_step, advance_subgoal=True
            )
            merged = store.state

        if progress.success and merged.planner_output is not None:
            store.record_success(goal_key, goal_key, step=global_step)
            merged = store.state
            if merged.success_memory:
                merged.success_memory[-1]["kind"] = "goal_complete"

        if progress.failure and merged.planner_output is not None:
            store.record_failure(goal_key, progress.reason)
            merged = store.state

        trainer_losses = int((info or {}).get("hrl_trainer_battle_loss", 0))
        if trainer_losses > 0 and merged.planner_output is not None:
            store.record_failure(goal_key, "repeated_battle_loss")
            merged = store.state

        store.replace(merged)
        return merged
