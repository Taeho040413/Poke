"""Select LLM or rule-based planner (only ``planner.enabled`` toggles LLM)."""

from __future__ import annotations

from pokemon_hrl.planner.base import Planner
from pokemon_hrl.planner.rule_based import RuleBasedPlanner


def build_planner(cfg) -> Planner:
    curriculum_path = str(cfg.hrl.curriculum.path)
    scenario_index = int(cfg.hrl.training.get("scenario_index", 0))
    log_output = bool(cfg.hrl.logging.get("goal_events", True))
    fallback = RuleBasedPlanner(
        curriculum_path,
        scenario_index=scenario_index,
        log_output=log_output,
    )

    if not bool(cfg.hrl.planner.get("enabled", False)):
        return fallback

    from pokemon_hrl.planner.openrouter import OpenRouterPlanner

    return OpenRouterPlanner(
        cfg.hrl.planner,
        curriculum_path=curriculum_path,
        scenario_index=scenario_index,
        log_output=log_output,
    )
