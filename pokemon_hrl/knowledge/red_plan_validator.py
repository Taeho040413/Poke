"""Validate and repair LLM planner output using chapter knowledge."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from pokemon_hrl.knowledge.red_maps import MapIds
from pokemon_hrl.planner.progression import MAX_SUBGOALS, MIN_SUBGOALS

_OAKS_LAB = int(MapIds.OAKS_LAB.value)
_VIRIDIAN_MART = int(MapIds.VIRIDIAN_MART.value)
_LAVENDER_TOWN = int(MapIds.LAVENDER_TOWN.value)

_INVENTED_FLAG_REPAIRS: dict[str, str] = {
    "flag:EVENT_REACHED_LAVENDER_TOWN": f"map_reached:{_LAVENDER_TOWN}",
}

_NPC_INTERACTION_TOKENS = (
    "stat:first_npc_talk_count",
    "stat_on_target_map:first_npc_talk_count",
    "stat:new_npc_textbox_count",
    "stat_on_target_map:new_npc_textbox_count",
)

_FORBIDDEN_OAK_PARCEL_RECEIVE_PATTERNS = (
    re.compile(r"professor\s+oak", re.IGNORECASE),
    re.compile(r"오크\s*박사"),
    re.compile(r"oak.?s?\s*lab", re.IGNORECASE),
    re.compile(r"오크\s*연구소.*받"),
    re.compile(r"오크\s*박사.*받"),
    re.compile(r"professor\s+oak.*받", re.IGNORECASE),
    re.compile(r"박사.*받"),
    re.compile(r"oak.*에게.*받", re.IGNORECASE),
)


@dataclass
class PlanValidationResult:
    plan: dict[str, Any]
    repaired: bool = False
    rejected: bool = False
    repair_notes: list[str] = field(default_factory=list)


def _goal_criteria(chapter_goal: dict[str, Any]) -> list[str]:
    return list(chapter_goal.get("suggested_goal_success_criteria") or [])


def _current_map_id(world_state: dict[str, Any], planning_context: dict[str, Any]) -> int:
    if "map_id" in world_state:
        return int(world_state["map_id"])
    return int(planning_context.get("current_map_id", 0))


def _subgoal_text(subgoal: dict[str, Any]) -> str:
    return " ".join(
        str(subgoal.get(key, ""))
        for key in ("where", "what", "how")
    )


def _repair_invented_flag(token: str) -> tuple[str, bool]:
    raw = token.strip()
    if raw in _INVENTED_FLAG_REPAIRS:
        return _INVENTED_FLAG_REPAIRS[raw], True
    if raw.startswith("flag:EVENT_REACHED_") and raw.endswith("_TOWN"):
        town = raw.removeprefix("flag:EVENT_REACHED_").removesuffix("_TOWN")
        from pokemon_hrl.knowledge.red_maps import map_name_to_id

        map_id = map_name_to_id(town)
        if map_id is not None:
            return f"map_reached:{map_id}", True
    return raw, False


def _contains_forbidden_oak_receive_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in _FORBIDDEN_OAK_PARCEL_RECEIVE_PATTERNS)


def _is_npc_interaction_subgoal(subgoal: dict[str, Any]) -> bool:
    criteria = subgoal.get("success_criteria") or []
    return any(str(token) in _NPC_INTERACTION_TOKENS for token in criteria)


def _subgoal_mentions_forbidden_target(
    subgoal: dict[str, Any],
    forbidden_targets: list[str],
) -> bool:
    text = _subgoal_text(subgoal)
    lowered = text.lower()
    for target in forbidden_targets:
        if target.lower() in lowered:
            return True
    return False


def _remove_completed_tokens(
    subgoals: list[dict[str, Any]],
    do_not_emit: set[str],
    *,
    goal_criteria: list[str],
    repair_notes: list[str],
) -> list[dict[str, Any]]:
    goal_key = goal_criteria[0] if goal_criteria else None
    cleaned: list[dict[str, Any]] = []
    for index, subgoal in enumerate(subgoals):
        criteria = [str(token) for token in (subgoal.get("success_criteria") or [])]
        is_final = index == len(subgoals) - 1
        filtered: list[str] = []
        for token in criteria:
            if token in do_not_emit and not (token == goal_key and is_final):
                repair_notes.append(f"removed completed token {token!r} from subgoal[{index}]")
                continue
            if token == goal_key and not is_final:
                repair_notes.append(f"removed chapter goal token from non-final subgoal[{index}]")
                continue
            filtered.append(token)
        if filtered:
            updated = dict(subgoal)
            updated["success_criteria"] = filtered
            cleaned.append(updated)
    return cleaned


def _parcel_deliver_plan_is_invalid(
    plan: dict[str, Any],
    *,
    current_map_id: int,
    chapter_facts: dict[str, Any],
) -> bool:
    del current_map_id
    subgoals = plan.get("subgoal") or []
    if not subgoals:
        return True

    mart_receive_patterns = (
        re.compile(r"mart\s+clerk", re.IGNORECASE),
        re.compile(r"직원.*받"),
        re.compile(r"프렌들리숍.*받"),
    )
    for subgoal in subgoals:
        text = _subgoal_text(subgoal)
        if any(pattern.search(text) for pattern in mart_receive_patterns):
            return True
        if "받는다" in text and "전달" not in text:
            return True

    hint = plan.get("hint") or {}
    target_map_id = int(hint.get("target_map_id", -1))
    required_target = int(chapter_facts.get("required_target_map_id", _OAKS_LAB))
    if target_map_id != required_target:
        return True
    return False


def _parcel_receive_plan_is_invalid(
    plan: dict[str, Any],
    *,
    current_map_id: int,
    chapter_facts: dict[str, Any],
) -> bool:
    subgoals = plan.get("subgoal") or []
    if not subgoals:
        return True

    required_target = int(chapter_facts.get("required_target_map_id", _VIRIDIAN_MART))
    at_target = current_map_id == required_target
    if at_target and len(subgoals) == 1:
        final = subgoals[0]
        criteria = final.get("success_criteria") or []
        if criteria == ["flag:EVENT_GOT_OAKS_PARCEL"]:
            return False

    forbidden_targets = list(chapter_facts.get("forbidden_targets") or [])
    for subgoal in subgoals:
        text = _subgoal_text(subgoal)
        if _contains_forbidden_oak_receive_text(text):
            return True
        if _subgoal_mentions_forbidden_target(subgoal, forbidden_targets):
            return True

    if current_map_id == _OAKS_LAB:
        first = subgoals[0]
        if _is_npc_interaction_subgoal(first):
            return True
        if _subgoal_mentions_forbidden_target(first, forbidden_targets):
            return True

    final = subgoals[-1]
    final_text = _subgoal_text(final)
    if _contains_forbidden_oak_receive_text(final_text):
        return True

    hint = plan.get("hint") or {}
    target_map_id = int(hint.get("target_map_id", -1))
    required_target = int(chapter_facts.get("required_target_map_id", _VIRIDIAN_MART))
    if target_map_id == _OAKS_LAB:
        return True
    if target_map_id != required_target:
        return True

    for subgoal in subgoals:
        criteria = subgoal.get("success_criteria") or []
        if any(str(token).startswith("stat_on_target_map:") for token in criteria):
            if target_map_id != required_target:
                return True
    return False


def validate_and_repair_plan(
    plan: dict[str, Any],
    chapter_goal: dict[str, Any],
    planner_knowledge: dict[str, Any],
    planning_context: dict[str, Any],
    world_state: dict[str, Any],
) -> PlanValidationResult:
    """Validate planner JSON and repair obvious knowledge violations."""
    repaired_plan = copy.deepcopy(plan)
    repair_notes: list[str] = []
    repaired = False
    rejected = False

    goal_criteria = _goal_criteria(chapter_goal)
    if goal_criteria and repaired_plan.get("success_criteria") != goal_criteria:
        repaired_plan["success_criteria"] = list(goal_criteria)
        repair_notes.append("corrected success_criteria to match chapter goal")
        repaired = True

    chapter_facts = planner_knowledge.get("chapter_facts") or {}
    required_target = chapter_facts.get("required_target_map_id")
    current_map_id = _current_map_id(world_state, planning_context)
    do_not_emit = set(planning_context.get("do_not_emit") or [])

    hint = dict(repaired_plan.get("hint") or {})
    if required_target is not None:
        target_map_id = int(hint.get("target_map_id", current_map_id))
        if target_map_id != int(required_target):
            hint["target_map_id"] = int(required_target)
            repair_notes.append(
                f"corrected hint.target_map_id to required_target_map_id={required_target}"
            )
            repaired = True
    elif "target_map_id" not in hint:
        hint["target_map_id"] = current_map_id
        repaired = True
    repaired_plan["hint"] = hint

    subgoals = list(repaired_plan.get("subgoal") or [])
    for index, subgoal in enumerate(subgoals):
        criteria = [str(token) for token in (subgoal.get("success_criteria") or [])]
        new_criteria: list[str] = []
        for token in criteria:
            fixed, changed = _repair_invented_flag(token)
            if changed:
                repair_notes.append(f"repaired invented flag {token!r} -> {fixed!r}")
                repaired = True
            new_criteria.append(fixed)
        if new_criteria != criteria:
            subgoal = dict(subgoal)
            subgoal["success_criteria"] = new_criteria
            subgoals[index] = subgoal

    cleaned_subgoals = _remove_completed_tokens(
        subgoals,
        do_not_emit,
        goal_criteria=goal_criteria,
        repair_notes=repair_notes,
    )
    if cleaned_subgoals != subgoals:
        repaired = True
    subgoals = cleaned_subgoals
    repaired_plan["subgoal"] = subgoals

    goal_key = goal_criteria[0] if goal_criteria else None
    if goal_key == "flag:EVENT_GOT_OAKS_PARCEL" and chapter_facts:
        if _parcel_receive_plan_is_invalid(
            repaired_plan,
            current_map_id=current_map_id,
            chapter_facts=chapter_facts,
        ):
            rejected = True
            repair_notes.append("reject: invalid Oak parcel receive plan")
    elif goal_key == "flag:EVENT_OAK_GOT_PARCEL" and chapter_facts:
        if _parcel_deliver_plan_is_invalid(
            repaired_plan,
            current_map_id=current_map_id,
            chapter_facts=chapter_facts,
        ):
            rejected = True
            repair_notes.append("reject: invalid Oak parcel deliver plan")

    subgoal_count = len(repaired_plan.get("subgoal") or [])
    at_target = required_target is not None and current_map_id == int(required_target)
    min_allowed = 1 if at_target else MIN_SUBGOALS
    if subgoal_count < min_allowed or subgoal_count > MAX_SUBGOALS:
        if not rejected:
            rejected = True
            repair_notes.append(
                f"reject: subgoal count {subgoal_count} outside {min_allowed}-{MAX_SUBGOALS}"
            )

    if "no_progress" not in (repaired_plan.get("failure_criteria") or []):
        repaired_plan["failure_criteria"] = list(repaired_plan.get("failure_criteria") or []) + [
            "no_progress"
        ]
        repaired = True

    return PlanValidationResult(
        plan=repaired_plan,
        repaired=repaired,
        rejected=rejected,
        repair_notes=repair_notes,
    )


def knowledge_log_fields(
    planner_knowledge: dict[str, Any],
    validation: PlanValidationResult,
) -> dict[str, Any]:
    """Structured fields for planner logging."""
    chapter_facts = planner_knowledge.get("chapter_facts") or {}
    current_map = planner_knowledge.get("current_map") or {}
    return {
        "current_map_id": current_map.get("id"),
        "current_map_name": current_map.get("name"),
        "active_chapter_fact_label": planner_knowledge.get("active_chapter_fact_label")
        or chapter_facts.get("label"),
        "map_knowledge": planner_knowledge.get("map_knowledge"),
        "required_target_map_id": planner_knowledge.get("required_target_map_id")
        or chapter_facts.get("required_target_map_id"),
        "validator_repaired": validation.repaired,
        "validator_rejected": validation.rejected,
        "validator_notes": list(validation.repair_notes),
    }
