"""Mode selector — v1 forces INTERACTIVE when disabled."""

from __future__ import annotations

from dataclasses import dataclass

from pokemon_hrl.types import Mode, WorldState


@dataclass
class ModeSelector:
    enabled: bool = False
    forced_mode: Mode = Mode.INTERACTIVE

    def select(self, state: WorldState) -> Mode:
        if not self.enabled:
            return self.forced_mode

        if self._growth_triggered(state):
            return Mode.GROWTH

        planner = state.planner_output
        target = planner.target_map_id if planner is not None else None
        if target is not None and int(target) == int(state.map_id):
            return Mode.INTERACTIVE
        return Mode.EXPLORATION

    @staticmethod
    def _growth_triggered(state: WorldState) -> bool:
        for entry in state.failure_memory:
            if entry.get("cause") == "repeated_battle_loss" and int(entry.get("count", 0)) >= 2:
                return True
        return False
