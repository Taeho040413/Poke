"""Summarizer protocol."""

from __future__ import annotations

from typing import Protocol

from pokemon_hrl.types import StateSummary, WorldState


class Summarizer(Protocol):
    def summarize(self, state: WorldState) -> StateSummary: ...
