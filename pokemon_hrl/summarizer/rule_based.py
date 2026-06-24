"""Rule-based summarizer — full WorldState → StateSummary mapping."""

from __future__ import annotations

from pokemon_hrl.summarizer.mapping import (
    build_evidence,
    exploration_coverage,
    failure_cause,
    interaction_outcome,
    semantic_progression,
)
from pokemon_hrl.types import StateSummary, WorldState


class RuleBasedSummarizer:
    def summarize(self, state: WorldState) -> StateSummary:
        return StateSummary(
            semantic_progression=semantic_progression(state),
            exploration_coverage=exploration_coverage(state),
            interaction_outcome=interaction_outcome(state),
            failure_cause=failure_cause(state),
            evidence=build_evidence(state),
        )
