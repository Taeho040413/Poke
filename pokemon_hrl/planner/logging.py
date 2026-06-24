"""Terminal logging for planner (LLM / rule-based) output."""

from __future__ import annotations

from pokemon_hrl.mode.subgoal import current_subgoal
from pokemon_hrl.planner.criteria import planner_goal_key, subgoal_label
from pokemon_hrl.planner.descriptions import format_map_id
from pokemon_hrl.types import PlannerOutput, Subgoal


def _completion_line(subgoal: Subgoal) -> str:
    tokens = ", ".join(f"`{token}`" for token in subgoal.success_criteria)
    return f"{tokens} — 메모리 카운터/플래그 충족 시 완료"


def _print_subgoal_details(subgoal: Subgoal) -> None:
    if subgoal.where:
        print(f"        어디: {subgoal.where}", flush=True)
    if subgoal.what:
        print(f"        무엇: {subgoal.what}", flush=True)
    if subgoal.how:
        print(f"        방법: {subgoal.how}", flush=True)
    print(f"        완료조건: {_completion_line(subgoal)}", flush=True)


def _subgoal_summary(subgoal: Subgoal | None) -> str:
    if subgoal is None or not subgoal.success_criteria:
        return "(없음)"
    if subgoal.what:
        return subgoal.what
    return subgoal_label(subgoal)


def log_planner_knowledge(
    fields: dict,
    *,
    used_fallback: bool = False,
) -> None:
    """Print injected planner knowledge and validator metadata."""
    print("[planner:knowledge]", flush=True)
    current_map_id = fields.get("current_map_id")
    current_map_name = fields.get("current_map_name")
    if current_map_id is not None:
        print(
            f"  current_map: {current_map_name} (map_id={current_map_id})",
            flush=True,
        )
    label = fields.get("active_chapter_fact_label")
    if label:
        print(f"  active_chapter_fact: {label}", flush=True)
    map_knowledge = fields.get("map_knowledge")
    if map_knowledge:
        print(f"  map_knowledge: {map_knowledge}", flush=True)
    required_target = fields.get("required_target_map_id")
    if required_target is not None:
        print(f"  required_target_map_id: {required_target}", flush=True)
    repaired = fields.get("validator_repaired")
    rejected = fields.get("validator_rejected")
    if repaired is not None or rejected is not None:
        print(
            f"  validator: repaired={bool(repaired)} rejected={bool(rejected)}",
            flush=True,
        )
    notes = fields.get("validator_notes") or []
    for note in notes:
        print(f"    - {note}", flush=True)
    if used_fallback:
        print("  used_deterministic_fallback: true", flush=True)


def log_planner_output(
    planner: PlannerOutput,
    *,
    source: str = "llm",
    model: str | None = None,
    step: int | None = None,
    scenario_index: int | None = None,
) -> None:
    """Print memory-aligned goal/subgoal criteria to stdout."""
    header = f"[planner:{source}]"
    if scenario_index is not None:
        header += f" scenario={scenario_index}"
    if step is not None:
        header += f" step={step}"
    if model:
        header += f" model={model}"
    print(header, flush=True)

    target_map = planner.target_map_id
    print(f"  goal_key: {planner_goal_key(planner)}", flush=True)
    print(
        f"  hint: {dict(planner.hint)} "
        f"(Interactive 모드 = 플레이어 map_id == {target_map}, "
        f"목표 맵: {format_map_id(target_map)})",
        flush=True,
    )

    if planner.success_criteria:
        print("  success_criteria (챕터 목표):", flush=True)
        for token in planner.success_criteria:
            print(f"    - {token}", flush=True)
    else:
        print("  success_criteria: (none)", flush=True)

    if planner.failure_criteria:
        print("  failure_criteria:", flush=True)
        for token in planner.failure_criteria:
            print(f"    - {token}", flush=True)

    if planner.subgoal:
        print("  subgoals:", flush=True)
        for idx, sg in enumerate(planner.subgoal):
            print(f"    [{idx}] {subgoal_label(sg)}", flush=True)
            _print_subgoal_details(sg)
    else:
        print("  subgoals: (none)", flush=True)


def log_active_goal_state(
    planner: PlannerOutput,
    *,
    source: str = "training",
    subgoal_index: int = 0,
    map_id: int | None = None,
    env_id: int = 0,
) -> None:
    """Print the active subgoal and map context (e.g. on env reset during training)."""
    active = current_subgoal(planner.subgoal, subgoal_index)
    active_label = subgoal_label(active) if active else "(none)"
    summary = _subgoal_summary(active)
    map_part = f" map={map_id} ({format_map_id(map_id)})" if map_id is not None else ""
    print(
        f"[goal:active] source={source} env={env_id}{map_part} "
        f"target_map={planner.target_map_id} ({format_map_id(planner.target_map_id)}) "
        f"subgoal_index={subgoal_index} active={active_label!r} "
        f"goal={planner_goal_key(planner)!r} — {summary}",
        flush=True,
    )
