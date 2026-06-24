"""LLM prompt construction for the planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pokemon_hrl.knowledge.planner_knowledge import build_planner_knowledge
from pokemon_hrl.planner.progression import (
    chapter_goal_payload,
    plan_size_limits_payload,
    planning_context_payload,
)
from pokemon_hrl.paths import CONFIG_DIR, HRL_ROOT
from pokemon_hrl.types import StateSummary, WorldState
from pokemon_hrl.world_state.serialization import (
    FINAL_GOAL_TEXT,
    state_summary_to_dict,
    world_state_to_dict,
)

DEFAULT_PROMPT_PATH = CONFIG_DIR / "planner_system_prompt.txt"


def load_system_prompt(path: str | Path | None = None) -> str:
    if path is None:
        prompt_path = DEFAULT_PROMPT_PATH
    else:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            prompt_path = candidate.resolve()
        else:
            prompt_path = (HRL_ROOT / candidate).resolve()
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8").strip()
    return _FALLBACK_SYSTEM_PROMPT


def build_user_message(summary: StateSummary, state: WorldState) -> str:
    chapter_goal = chapter_goal_payload(state)
    planning_context = planning_context_payload(state)
    world_state = world_state_to_dict(state)
    planner_knowledge = build_planner_knowledge(chapter_goal, world_state, planning_context)
    payload = {
        "north_star_final_objective": FINAL_GOAL_TEXT,
        "chapter_goal": chapter_goal,
        "planning_context": planning_context,
        "planner_knowledge": planner_knowledge,
        "state_summary": state_summary_to_dict(summary),
        "world_state": world_state,
        "plan_size_limits": plan_size_limits_payload(),
        "output_schema": {
            "subgoal": [
                {
                    "success_criteria": [
                        "flag:EVENT_NAME | map_reached:N | party_healed | stat:NAME | stat_on_target_map:NAME"
                    ],
                    "where": "Korean — where (map/place for this step)",
                    "what": "Korean — what to achieve",
                    "how": "Korean — how (tile move, press A, etc.)",
                }
            ],
            "hint": {
                "target_map_id": "int — map id for Interactive mode routing",
                "target_x": "optional int — goal tile x on target_map_id",
                "target_y": "optional int — goal tile y on target_map_id",
                "target_object_id": "optional int or string — interactable object key",
                "target_event_id": "optional int or string — event/flag key",
            },
            "success_criteria": [
                "flag:EVENT_NAME | map_reached:N | party_healed | stat:NAME | stat_on_target_map:NAME"
            ],
            "failure_criteria": ["no_progress | flag:... | map_reached:N"],
            "note": (
                "Emit 2–4 subgoals so goal + subgoals = 3–5 steps total. "
                "Do not emit completed flags or completed subgoals. "
                "Final subgoal may use chapter goal token when it directly completes the chapter."
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_chat_messages(
    summary: StateSummary,
    state: WorldState,
    *,
    system_prompt: str | None = None,
    prompt_path: str | Path | None = None,
) -> list[dict[str, str]]:
    system = system_prompt or load_system_prompt(prompt_path)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(summary, state)},
    ]


_FALLBACK_SYSTEM_PROMPT = """You are the high-level planner for a Pokemon Red HRL agent.

Given a structured state summary and world state JSON, propose the next goal plan.

Rules:
- Return ONLY valid JSON matching the output_schema in the user message.
- success_criteria must match chapter_goal.suggested_goal_success_criteria (goal is chapter-fixed).
- Subgoals must be reasoned from world_state, planning_context, and chapter goal — not copied from a fixed list.
- hint.target_map_id must be the player's current map_id for Interactive-mode steps unless travel is required.
- hint may also include target_x, target_y, target_object_id, and target_event_id for goal-conditioned memory.
- success_criteria / failure_criteria / subgoal.success_criteria use memory tokens only.
- Emit 2–4 subgoals; each step has memory tokens plus Korean where/what/how for this situation.
- Never put planning_context.do_not_emit tokens into subgoals (already completed flags/steps).
- Never jump to late-game milestones beyond the current chapter.
- If planning_context.recent_failures mention repeated issues, adjust subgoals to recover before advancing.
"""
