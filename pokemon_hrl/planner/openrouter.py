"""OpenRouter LLM planner with validation and rule-based fallback."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from pokemon_hrl.knowledge.planner_knowledge import build_planner_knowledge
from pokemon_hrl.knowledge.red_plan_fallback import build_deterministic_fallback_plan
from pokemon_hrl.knowledge.red_plan_validator import (
    knowledge_log_fields,
    validate_and_repair_plan,
)
from pokemon_hrl.planner.logging import log_planner_knowledge, log_planner_output
from pokemon_hrl.planner.progression import (
    chapter_goal_payload,
    planning_context_payload,
    scope_planner_to_chapter,
)
from pokemon_hrl.planner.prompt import build_chat_messages, load_system_prompt
from pokemon_hrl.planner.rule_based import RuleBasedPlanner
from pokemon_hrl.planner.validation import (
    PlannerValidationError,
    parse_planner_dict,
    validate_subgoal_descriptions,
)
from pokemon_hrl.types import PlannerOutput, StateSummary, WorldState
from pokemon_hrl.world_state.serialization import world_state_to_dict

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterPlanner:
    def __init__(
        self,
        planner_cfg,
        *,
        curriculum_path: str,
        scenario_index: int = 0,
        log_output: bool = True,
    ):
        self.model = str(planner_cfg.get("model", "openai/gpt-oss-120b"))
        self.api_key_env = str(planner_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        self.timeout_sec = int(planner_cfg.get("timeout_sec", 120))
        self.max_retries = int(planner_cfg.get("max_retries", 2))
        self.prompt_path = planner_cfg.get("prompt_path")
        self.log_output = bool(log_output)
        self.scenario_index = int(scenario_index)
        self._fallback = RuleBasedPlanner(
            curriculum_path,
            scenario_index=scenario_index,
            log_output=False,
        )

    def plan(self, summary: StateSummary, state: WorldState) -> PlannerOutput:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            logger.warning("%s not set — using rule-based planner", self.api_key_env)
            return self._log_rule_based(self._fallback.plan(summary, state))

        chapter_goal = chapter_goal_payload(state)
        planning_context = planning_context_payload(state)
        world_state = world_state_to_dict(state)
        planner_knowledge = build_planner_knowledge(
            chapter_goal,
            world_state,
            planning_context,
        )

        system_prompt = load_system_prompt(self.prompt_path)
        messages = build_chat_messages(summary, state, system_prompt=system_prompt)

        last_error: Exception | None = None
        used_knowledge_fallback = False
        validation_result = None
        for attempt in range(self.max_retries + 1):
            try:
                content = self._call_api(api_key, messages)
                parsed = self._parse_content(content)
                validation_result = validate_and_repair_plan(
                    parsed,
                    chapter_goal,
                    planner_knowledge,
                    planning_context,
                    world_state,
                )
                if validation_result.rejected:
                    parsed = build_deterministic_fallback_plan(
                        chapter_goal,
                        planner_knowledge,
                        planning_context,
                        world_state,
                    )
                    used_knowledge_fallback = True
                    validation_result = validate_and_repair_plan(
                        parsed,
                        chapter_goal,
                        planner_knowledge,
                        planning_context,
                        world_state,
                    )
                output = parse_planner_dict(validation_result.plan)
                output = scope_planner_to_chapter(output, state)
                validate_subgoal_descriptions(output.subgoal)
                if self.log_output:
                    log_planner_knowledge(
                        knowledge_log_fields(planner_knowledge, validation_result),
                        used_fallback=used_knowledge_fallback,
                    )
                    log_planner_output(
                        output,
                        source="knowledge-fallback" if used_knowledge_fallback else "llm",
                        model=self.model,
                        scenario_index=self.scenario_index,
                    )
                return output
            except (
                PlannerValidationError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                logger.warning("LLM planner parse failed (attempt %s): %s", attempt + 1, exc)
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. "
                            f"Error: {exc}. Return ONLY corrected JSON."
                        ),
                    }
                ]
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                logger.warning("LLM planner HTTP failed: %s", exc)
                break

        logger.warning("Falling back to deterministic knowledge planner: %s", last_error)
        parsed = build_deterministic_fallback_plan(
            chapter_goal,
            planner_knowledge,
            planning_context,
            world_state,
        )
        validation_result = validate_and_repair_plan(
            parsed,
            chapter_goal,
            planner_knowledge,
            planning_context,
            world_state,
        )
        output = parse_planner_dict(validation_result.plan)
        output = scope_planner_to_chapter(output, state)
        if self.log_output:
            log_planner_knowledge(
                knowledge_log_fields(planner_knowledge, validation_result),
                used_fallback=True,
            )
            log_planner_output(
                output,
                source="knowledge-fallback",
                scenario_index=self.scenario_index,
            )
        return output

    def _log_rule_based(self, output: PlannerOutput) -> PlannerOutput:
        if self.log_output:
            log_planner_output(
                output,
                source="rule-based",
                scenario_index=self.scenario_index,
            )
        return output

    def _call_api(self, api_key: str, messages: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])

    @staticmethod
    def _parse_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        if not isinstance(data, dict):
            raise PlannerValidationError("LLM response root must be an object")
        return data
