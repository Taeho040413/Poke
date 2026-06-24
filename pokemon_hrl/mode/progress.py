"""Progress check — criteria, reward shaping signals, done."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pokemon_hrl.mode.subgoal import current_subgoal, resolve_subgoal_criteria
from pokemon_hrl.planner.criteria import planner_goal_key, subgoal_label
from pokemon_hrl.types import PlannerOutput, ProgressResult, Subgoal, WorldState


@dataclass
class ProgressCheck:
    def check_criteria(
        self,
        planner: PlannerOutput,
        before: WorldState,
        after: WorldState,
        *,
        info: dict | None = None,
    ) -> ProgressResult:
        failure = self.check_failure(planner, before, after, info=info)
        if failure.failure:
            return failure
        success = self.check_success(planner, before, after)
        if success.success:
            return success
        return ProgressResult()

    def check_failure(
        self,
        planner: PlannerOutput,
        before: WorldState,
        after: WorldState,
        *,
        info: dict | None = None,
    ) -> ProgressResult:
        if self._criteria_met(
            planner.failure_criteria, before, after, failure=True, planner=planner
        ):
            return ProgressResult(failure=True, reason="failure_criteria_met")
        if info and str(info.get("truncated_reason", "")) == "mode_max_steps":
            if "no_progress" in planner.failure_criteria:
                return ProgressResult(failure=True, reason="no_progress")
        return ProgressResult()

    def check_success(
        self,
        planner: PlannerOutput,
        before: WorldState,
        after: WorldState,
    ) -> ProgressResult:
        if self._criteria_met(
            planner.success_criteria, before, after, planner=planner
        ):
            return ProgressResult(success=True, reason="success_criteria_met")
        if before.map_id != after.map_id and planner.target_map_id == after.map_id:
            return ProgressResult(success=True, reason="target_map_reached")
        return ProgressResult()

    def check_subgoal_met(
        self,
        subgoal: Subgoal,
        planner: PlannerOutput,
        before: WorldState,
        after: WorldState,
    ) -> bool:
        criteria = resolve_subgoal_criteria(
            subgoal, target_map_id=planner.target_map_id
        )
        if not criteria:
            return False
        return self._criteria_met(criteria, before, after, planner=planner)

    def all_subgoals_complete(self, planner: PlannerOutput, subgoal_index: int) -> bool:
        if not planner.subgoal:
            return True
        return subgoal_index >= len(planner.subgoal)

    @staticmethod
    def _criteria_met(
        criteria: list[str],
        before: WorldState,
        after: WorldState,
        *,
        failure: bool = False,
        planner: PlannerOutput | None = None,
    ) -> bool:
        for raw in criteria:
            token = raw.strip()
            if token.startswith("flag:"):
                flag_name = token.split(":", 1)[1]
                if after.flags.get(flag_name):
                    return True
            elif token.startswith("map_reached:"):
                map_id = int(token.split(":", 1)[1])
                if failure:
                    if after.map_id == map_id:
                        return True
                elif before.map_id != after.map_id and after.map_id == map_id:
                    return True
            elif token.startswith("stat_on_target_map:"):
                if planner is None or planner.target_map_id is None:
                    continue
                stat_name = token.split(":", 1)[1]
                if after.map_id != planner.target_map_id:
                    continue
                before_val = int(before.resources.get(stat_name, 0))
                after_val = int(after.resources.get(stat_name, 0))
                if after_val > before_val:
                    return True
            elif token.startswith("stat:"):
                stat_name = token.split(":", 1)[1]
                before_val = int(before.resources.get(stat_name, 0))
                after_val = int(after.resources.get(stat_name, 0))
                if after_val > before_val:
                    return True
            elif token == "party_healed":
                before_fully_healed = bool(before.party) and all(
                    p["hp"] >= p["max_hp"] for p in before.party
                )
                after_fully_healed = bool(after.party) and all(
                    p["hp"] >= p["max_hp"] for p in after.party
                )
                if (not before_fully_healed) and after_fully_healed:
                    return True
        return False

    def should_replan(self, result: ProgressResult) -> bool:
        return result.success or result.failure


def progress_to_info(
    progress: ProgressResult,
    planner: PlannerOutput,
    after: WorldState,
    *,
    subgoal_index: int = 0,
) -> dict[str, Any]:
    active = current_subgoal(planner.subgoal, subgoal_index)
    return {
        "hrl_progress_success": int(progress.success),
        "hrl_progress_failure": int(progress.failure),
        "hrl_subgoal_success": int(progress.subgoal_success),
        "hrl_progress_reason": progress.reason,
        "hrl_goal_key": planner_goal_key(planner),
        "hrl_target_map_id": planner.target_map_id,
        "hrl_map_id": after.map_id,
        "hrl_subgoal_index": subgoal_index,
        "hrl_current_subgoal": subgoal_label(active) if active else "",
        "hrl_subgoal_completed": progress.subgoal if progress.subgoal_success else "",
    }


def progress_from_info(info: dict[str, Any] | None) -> ProgressResult | None:
    if not info or "hrl_progress_success" not in info:
        return None
    return ProgressResult(
        success=bool(info.get("hrl_progress_success")),
        failure=bool(info.get("hrl_progress_failure")),
        subgoal_success=bool(info.get("hrl_subgoal_success")),
        reason=str(info.get("hrl_progress_reason", "")),
        subgoal=str(info.get("hrl_subgoal_completed", "")),
        subgoal_index=int(info.get("hrl_subgoal_index", -1)),
    )


def log_goal_event(
    progress: ProgressResult,
    planner: PlannerOutput,
    after: WorldState,
    *,
    env_id: int = 0,
) -> None:
    tag = "success" if progress.success else "failure"
    target = planner.target_map_id
    print(
        f"[goal:{tag}] env={env_id} reason={progress.reason} "
        f"goal={planner_goal_key(planner)!r} map={after.map_id} target_map={target}",
        flush=True,
    )


def log_subgoal_event(
    subgoal: str,
    planner: PlannerOutput,
    after: WorldState,
    *,
    env_id: int = 0,
    index: int = 0,
) -> None:
    print(
        f"[subgoal:success] env={env_id} index={index} subgoal={subgoal!r} "
        f"goal={planner_goal_key(planner)!r} map={after.map_id} target_map={planner.target_map_id}",
        flush=True,
    )
