"""Rule-based planner (LLM disabled) — uses summary + curriculum."""

from __future__ import annotations

from pokemon_hrl.planner.criteria import planner_goal_key
from pokemon_hrl.planner.logging import log_planner_output
from pokemon_hrl.planner.validation import parse_planner_dict, planner_output_to_dict
from pokemon_hrl.training.curriculum import CurriculumScenario, load_curriculum
from pokemon_hrl.types import PlannerOutput, StateSummary, WorldState


class RuleBasedPlanner:
    """Select and adapt curriculum goals using StateSummary (no LLM)."""

    def __init__(
        self,
        curriculum_path: str,
        scenario_index: int = 0,
        *,
        log_output: bool = True,
    ):
        self.curriculum_path = curriculum_path
        self.scenario_index = int(scenario_index)
        self.log_output = bool(log_output)

    def plan(self, summary: StateSummary, state: WorldState) -> PlannerOutput:
        del summary
        scenarios = load_curriculum(self.curriculum_path)
        if not scenarios:
            return self._emit(self._fallback_plan(state))

        scenario = self._pick_scenario(scenarios, state)
        data = planner_output_to_dict(scenario.planner)

        if state.failure_memory and any(
            entry.get("cause") == "no_progress" for entry in state.failure_memory
        ):
            failures = data.get("failure_criteria") or []
            if "no_progress" not in failures:
                failures = list(failures) + ["no_progress"]
            data["failure_criteria"] = failures

        if data["hint"].get("target_map_id") is None:
            data["hint"]["target_map_id"] = int(state.map_id)

        return self._emit(parse_planner_dict(data))

    def _emit(self, output: PlannerOutput) -> PlannerOutput:
        if self.log_output:
            log_planner_output(
                output,
                source="rule-based",
                scenario_index=self.scenario_index,
            )
        return output

    def _pick_scenario(
        self,
        scenarios: list[CurriculumScenario],
        state: WorldState,
    ) -> CurriculumScenario:
        completed = {m.get("goal") for m in state.success_memory}

        target = state.planner_output.target_map_id if state.planner_output else None
        if target is not None:
            for scenario in scenarios:
                key = planner_goal_key(scenario.planner)
                if scenario.planner.target_map_id == target and key not in completed:
                    return scenario

        for scenario in scenarios:
            hint_map = scenario.planner.target_map_id
            key = planner_goal_key(scenario.planner)
            if hint_map is not None and hint_map == state.map_id and key not in completed:
                return scenario

        for scenario in scenarios:
            if planner_goal_key(scenario.planner) not in completed:
                return scenario

        return scenarios[self.scenario_index % len(scenarios)]

    @staticmethod
    def _fallback_plan(state: WorldState) -> PlannerOutput:
        return parse_planner_dict(
            {
                "subgoal": [
                    {
                        "success_criteria": [
                            "stat_on_target_map:first_npc_talk_count"
                        ],
                    },
                    {
                        "success_criteria": [
                            "stat_on_target_map:first_object_interaction_count"
                        ],
                    },
                ],
                "hint": {"target_map_id": int(state.map_id)},
                "success_criteria": [f"map_reached:{int(state.map_id)}"],
                "failure_criteria": ["no_progress"],
            }
        )
