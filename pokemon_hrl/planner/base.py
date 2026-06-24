"""Planner protocol."""

from __future__ import annotations

from typing import Protocol

from pokemon_hrl.types import PlannerOutput, StateSummary, WorldState


class Planner(Protocol):
    def plan(self, summary: StateSummary, state: WorldState) -> PlannerOutput: ...
