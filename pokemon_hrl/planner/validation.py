"""Validate and parse PlannerOutput from LLM or curriculum."""

from __future__ import annotations

import re
from typing import Any

from pokemon_hrl.planner.criteria import criteria_label
from pokemon_hrl.types import PlannerOutput, Subgoal

_CRITERIA_FLAG = re.compile(r"^flag:[A-Za-z0-9_]+$")
_CRITERIA_MAP = re.compile(r"^map_reached:\d+$")
_CRITERIA_STAT = re.compile(r"^stat:[A-Za-z0-9_]+$")
_CRITERIA_STAT_ON_TARGET_MAP = re.compile(r"^stat_on_target_map:[A-Za-z0-9_]+$")
_KNOWN_FAILURE = frozenset({"no_progress"})
_SUCCESS_SIMPLE = frozenset({"party_healed"})


class PlannerValidationError(ValueError):
    pass


def validate_criteria_token(token: str, *, kind: str) -> str:
    raw = token.strip()
    if not raw:
        raise PlannerValidationError(f"Empty {kind} criterion")
    if kind == "failure" and raw in _KNOWN_FAILURE:
        return raw
    if kind == "success" and raw in _SUCCESS_SIMPLE:
        return raw
    if (
        _CRITERIA_FLAG.match(raw)
        or _CRITERIA_MAP.match(raw)
        or _CRITERIA_STAT.match(raw)
        or _CRITERIA_STAT_ON_TARGET_MAP.match(raw)
    ):
        return raw
    raise PlannerValidationError(f"Invalid {kind} criterion: {raw!r}")


def parse_subgoal_list(
    raw: list[Any],
    *,
    target_map_id: int | None = None,
) -> list[Subgoal]:
    del target_map_id
    subgoals: list[Subgoal] = []
    for item in raw:
        if isinstance(item, str):
            raise PlannerValidationError(
                "subgoal must be objects with success_criteria (memory tokens), not strings"
            )
        if not isinstance(item, dict):
            if hasattr(item, "items"):
                item = dict(item)
            else:
                raise PlannerValidationError("subgoal items must be objects")
        criteria_raw = item.get("success_criteria", [])
        if not isinstance(criteria_raw, list):
            raise PlannerValidationError("subgoal.success_criteria must be a list")
        criteria = [
            validate_criteria_token(str(token), kind="success")
            for token in criteria_raw
            if str(token).strip()
        ]
        if not criteria:
            raise PlannerValidationError("subgoal.success_criteria is required")
        where = str(item.get("where", "")).strip()
        what = str(item.get("what", "")).strip()
        how = str(item.get("how", "")).strip()
        subgoals.append(
            Subgoal(
                success_criteria=criteria,
                where=where,
                what=what,
                how=how,
            )
        )
    return subgoals


def validate_subgoal_descriptions(subgoals: list[Subgoal]) -> None:
    """LLM plans must include situational descriptions — not template-mapped text."""
    for index, subgoal in enumerate(subgoals):
        missing = [
            label
            for label, value in (
                ("where", subgoal.where),
                ("what", subgoal.what),
                ("how", subgoal.how),
            )
            if not value
        ]
        if missing:
            raise PlannerValidationError(
                f"subgoal[{index}] missing description field(s): {', '.join(missing)}. "
                "Write where/what/how for this specific situation in Korean."
            )


def parse_planner_dict(data: dict[str, Any]) -> PlannerOutput:
    if not isinstance(data, dict):
        raise PlannerValidationError("Planner response must be a JSON object")

    hint_raw = data.get("hint", {})
    if not isinstance(hint_raw, dict):
        raise PlannerValidationError("hint must be an object")
    hint = dict(hint_raw)
    if "target_map_id" not in hint:
        raise PlannerValidationError("hint.target_map_id is required")
    try:
        hint["target_map_id"] = int(hint["target_map_id"])
    except (TypeError, ValueError) as exc:
        raise PlannerValidationError("hint.target_map_id must be an integer") from exc

    subgoal_raw = data.get("subgoal", [])
    if not isinstance(subgoal_raw, list):
        raise PlannerValidationError("subgoal must be a list")
    subgoal = parse_subgoal_list(subgoal_raw, target_map_id=hint["target_map_id"])

    success_raw = data.get("success_criteria", [])
    if not isinstance(success_raw, list):
        raise PlannerValidationError("success_criteria must be a list")
    success = [validate_criteria_token(str(x), kind="success") for x in success_raw]
    if not success:
        raise PlannerValidationError("success_criteria is required (memory tokens)")

    failure_raw = data.get("failure_criteria", [])
    if not isinstance(failure_raw, list):
        raise PlannerValidationError("failure_criteria must be a list")
    failure = [validate_criteria_token(str(x), kind="failure") for x in failure_raw]

    return PlannerOutput(
        subgoal=subgoal,
        hint=hint,
        success_criteria=success,
        failure_criteria=failure,
    )


def subgoal_to_dict(subgoal: Subgoal) -> dict[str, Any]:
    payload: dict[str, Any] = {"success_criteria": list(subgoal.success_criteria)}
    if subgoal.where:
        payload["where"] = subgoal.where
    if subgoal.what:
        payload["what"] = subgoal.what
    if subgoal.how:
        payload["how"] = subgoal.how
    return payload


def planner_output_to_dict(planner: PlannerOutput) -> dict[str, Any]:
    return {
        "goal_key": criteria_label(planner.success_criteria),
        "subgoal": [subgoal_to_dict(sg) for sg in planner.subgoal],
        "hint": dict(planner.hint),
        "success_criteria": list(planner.success_criteria),
        "failure_criteria": list(planner.failure_criteria),
    }
