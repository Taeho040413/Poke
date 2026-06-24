"""In-memory World State DB with optional PyBoy checkpoint paths."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pokemon_hrl.planner.criteria import planner_goal_key, subgoal_label
from pokemon_hrl.planner.validation import subgoal_to_dict
from pokemon_hrl.types import PlannerOutput, StateSummary, WorldState

SAVE_POINT_NAME = "save_point.state"


class WorldStateStore:
    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        self.state = WorldState(map_id=0, x=0, y=0, badges=0)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_point_path: Path | None = None
        self.policy_checkpoint_path: Path | None = None
        self.bootstrap_save_point()

    def default_save_point_path(self) -> Path:
        return self.checkpoint_dir / SAVE_POINT_NAME

    def bootstrap_save_point(self, name: str = SAVE_POINT_NAME) -> Path | None:
        """Register an on-disk goal save point for rollback (e.g. after resume)."""
        path = self.checkpoint_dir / name
        if path.is_file():
            self.save_point_path = path
            return path
        return None

    def replace(self, state: WorldState) -> None:
        self.state = state

    def set_planner_output(self, planner: PlannerOutput) -> None:
        self.state.planner_output = planner
        self.state.goal_stack = {
            "goal_key": planner_goal_key(planner),
            "subgoals": [subgoal_to_dict(sg) for sg in planner.subgoal],
            "current_index": 0,
        }

    def set_recent_summary(self, summary: StateSummary) -> None:
        self.state.recent_summary = asdict(summary)

    def record_success(
        self,
        goal_key: str,
        subgoal_key: str,
        *,
        step: int,
        advance_subgoal: bool = False,
    ) -> None:
        self.state.success_memory.append(
            {"goal": goal_key, "subgoal": subgoal_key, "timestamp_step": step}
        )
        if self.state.goal_stack.get("goal_key") != goal_key:
            return
        if advance_subgoal:
            idx = int(self.state.goal_stack.get("current_index", 0))
            self.state.goal_stack = dict(self.state.goal_stack)
            self.state.goal_stack["current_index"] = idx + 1

    def record_failure(self, goal: str, cause: str) -> None:
        for entry in self.state.failure_memory:
            if entry.get("goal") == goal and entry.get("cause") == cause:
                entry["count"] = int(entry.get("count", 0)) + 1
                return
        self.state.failure_memory.append({"goal": goal, "cause": cause, "count": 1})

    def load_json(self, raw: str) -> None:
        from pokemon_hrl.world_state.serialization import world_state_from_dict

        self.state = world_state_from_dict(raw)

    def save_game_state(self, pyboy_state_bytes: bytes, name: str = "save_point.state") -> Path:
        path = self.checkpoint_dir / name
        path.write_bytes(pyboy_state_bytes)
        self.save_point_path = path
        return path

    def save_policy_path(self, path: Path) -> None:
        self.policy_checkpoint_path = path

    def to_json(self) -> str:
        payload = asdict(self.state)
        if self.state.planner_output is not None:
            payload["planner_output"] = asdict(self.state.planner_output)
        return json.dumps(payload, ensure_ascii=False, indent=2)
