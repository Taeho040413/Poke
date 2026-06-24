"""Story chapter scoping — medium/long-term north star, one immediate goal at a time."""

from __future__ import annotations

from dataclasses import dataclass

from pokemon_hrl.planner.criteria import criteria_label, planner_goal_key
from pokemon_hrl.planner.validation import PlannerValidationError
from pokemon_hrl.types import PlannerOutput, Subgoal, WorldState
from pokemon_hrl.world_state.serialization import FINAL_GOAL_TEXT, badge_names

# Goal counts as 1 step; subgoals fill the rest (total plan length 3–5).
MIN_PLAN_STEPS = 3
MAX_PLAN_STEPS = 5
MIN_SUBGOALS = MIN_PLAN_STEPS - 1
MAX_SUBGOALS = MAX_PLAN_STEPS - 1

# Ordered main-story milestones (immediate goal = first incomplete entry).
CHAPTER_MILESTONES: tuple[tuple[str, list[str]], ...] = (
    ("oaks_parcel", ["flag:EVENT_GOT_OAKS_PARCEL"]),
    ("deliver_parcel", ["flag:EVENT_OAK_GOT_PARCEL"]),
    ("pokedex", ["flag:EVENT_GOT_POKEDEX"]),
    ("beat_brock", ["flag:EVENT_BEAT_BROCK"]),
    ("beat_misty", ["flag:EVENT_BEAT_MISTY"]),
    ("beat_lt_surge", ["flag:EVENT_BEAT_LT_SURGE"]),
    ("lavender_town", ["map_reached:4"]),
)

_MILESTONE_RANK = {
    criteria_label(criteria): idx
    for idx, (_, criteria) in enumerate(CHAPTER_MILESTONES)
}


@dataclass(frozen=True)
class ChapterGoal:
    chapter_id: str
    label: str
    success_criteria: list[str]
    complete: bool = False


def _flag(state: WorldState, name: str) -> bool:
    return bool(state.flags.get(name))


def _milestone_complete(state: WorldState, criteria: list[str]) -> bool:
    for token in criteria:
        if token.startswith("flag:"):
            if not _flag(state, token.split(":", 1)[1]):
                return False
        elif token.startswith("map_reached:"):
            target = int(token.split(":", 1)[1])
            if state.map_id != target:
                return False
        else:
            return False
    return True


def infer_chapter_goal(state: WorldState) -> ChapterGoal:
    """Next immediate story goal from flags/badges (not the long-term north star)."""
    labels = {
        "oaks_parcel": "Receive Oak's Parcel from the Viridian Mart clerk",
        "deliver_parcel": "Deliver Oak's Parcel to Professor Oak in Oak's Lab",
        "pokedex": "Obtain the Pokedex",
        "beat_brock": "Beat Brock (Pewter Gym)",
        "beat_misty": "Beat Misty (Cerulean Gym)",
        "beat_lt_surge": "Beat Lt. Surge (Vermilion Gym)",
        "lavender_town": "Reach Lavender Town",
    }
    for chapter_id, criteria in CHAPTER_MILESTONES:
        if not _milestone_complete(state, criteria):
            return ChapterGoal(
                chapter_id=chapter_id,
                label=labels[chapter_id],
                success_criteria=list(criteria),
            )
    return ChapterGoal(
        chapter_id="complete",
        label="Final objective achieved",
        success_criteria=[],
        complete=True,
    )


def plan_size_limits_payload() -> dict[str, int]:
    return {
        "min_total_steps_including_goal": MIN_PLAN_STEPS,
        "max_total_steps_including_goal": MAX_PLAN_STEPS,
        "min_subgoals": MIN_SUBGOALS,
        "max_subgoals": MAX_SUBGOALS,
    }


def _completed_flag_tokens(state: WorldState) -> list[str]:
    return [f"flag:{name}" for name, value in sorted(state.flags.items()) if value]


def _completed_subgoal_keys(
    state: WorldState,
    *,
    goal_key: str | None = None,
) -> set[str]:
    keys: set[str] = set()
    for entry in state.success_memory:
        entry_goal = entry.get("goal")
        if goal_key is not None and entry_goal and str(entry_goal) != goal_key:
            continue
        subgoal = entry.get("subgoal")
        if subgoal:
            keys.add(str(subgoal))
        if entry.get("kind") == "goal_complete" and entry_goal:
            keys.add(str(entry_goal))

    stack = state.goal_stack or {}
    stack_goal = stack.get("goal_key")
    if goal_key is None or stack_goal == goal_key:
        subgoals = stack.get("subgoals") or []
        current_index = int(stack.get("current_index", 0))
        for raw in subgoals[:current_index]:
            if not isinstance(raw, dict):
                continue
            criteria = raw.get("success_criteria") or []
            if criteria:
                keys.add(criteria_label([str(token) for token in criteria]))
    return keys


def _do_not_emit_tokens(
    state: WorldState,
    *,
    goal_criteria: list[str] | None = None,
) -> list[str]:
    del goal_criteria
    blocked = set(_completed_flag_tokens(state))
    blocked.update(_completed_subgoal_keys(state))
    return sorted(blocked)


def planning_context_payload(
    state: WorldState,
    *,
    goal_criteria: list[str] | None = None,
) -> dict:
    """Structured world-DB hints so the LLM skips already-finished steps."""
    chapter = infer_chapter_goal(state)
    goal_criteria = goal_criteria or chapter.success_criteria
    goal_key = criteria_label(goal_criteria) if goal_criteria else None
    recent_failures = [
        {
            "goal": entry.get("goal"),
            "cause": entry.get("cause"),
            "count": entry.get("count", 1),
        }
        for entry in state.failure_memory[-5:]
    ]
    return {
        "current_map_id": int(state.map_id),
        "goal_stack_index": int((state.goal_stack or {}).get("current_index", 0)),
        "completed_flags": _completed_flag_tokens(state),
        "completed_subgoals": sorted(_completed_subgoal_keys(state, goal_key=goal_key)),
        "recent_failures": recent_failures,
        "do_not_emit": _do_not_emit_tokens(state, goal_criteria=goal_criteria),
        "instruction": (
            "Read world_state, state_summary, planner_knowledge, and this block before planning. "
            "Do not emit completed flags or completed subgoals. "
            "The final subgoal MAY use the same token as chapter_goal.suggested_goal_success_criteria "
            "if that subgoal directly completes the active chapter. "
            "Never emit a token already in completed_flags or completed_subgoals. "
            "Set hint.target_map_id to the next unfinished subgoal map when travel is required. "
            "After backend scoping you need 2–4 surviving subgoals (1 allowed when already on target map)."
        ),
    }


def chapter_goal_payload(state: WorldState) -> dict:
    chapter = infer_chapter_goal(state)
    badges = badge_names(state.badges)
    return {
        "north_star_final_objective": FINAL_GOAL_TEXT,
        "current_chapter_goal": chapter.label,
        "current_chapter_id": chapter.chapter_id,
        "suggested_goal_success_criteria": chapter.success_criteria,
        "chapter_complete": chapter.complete,
        "active_flags": sorted(k for k, v in state.flags.items() if v),
        "badges": badges,
        "plan_size_limits": plan_size_limits_payload(),
        "instruction": (
            "Set success_criteria to suggested_goal_success_criteria (goal is fixed for this chapter). "
            "You must reason over world_state, planning_context, and state_summary to emit 2–4 subgoals "
            "(goal + subgoals = 3–5 steps total). Each subgoal is one next step toward the goal — "
            "do not pick from a fixed list; choose memory tokens that match the current situation. "
            "Respect planning_context.do_not_emit — do not repeat completed flags or subgoals. "
            "Do not skip to Lavender, Lt. Surge, or other late-game milestones."
        ),
    }


def _criteria_rank(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    return _MILESTONE_RANK.get(criteria_label([token]))


def _subgoal_max_rank(subgoal: Subgoal) -> int | None:
    ranks = [_criteria_rank(token) for token in subgoal.success_criteria]
    ranks = [r for r in ranks if r is not None]
    return max(ranks) if ranks else None


def _flag_subgoal_done(state: WorldState, token: str) -> bool:
    if not token.startswith("flag:"):
        return False
    return _flag(state, token.split(":", 1)[1])


def _subgoal_already_satisfied(state: WorldState, subgoal: Subgoal) -> bool:
    for token in subgoal.success_criteria:
        if _flag_subgoal_done(state, token):
            return True
    return False


def _subgoal_already_completed(
    state: WorldState,
    subgoal: Subgoal,
    *,
    goal_key: str,
) -> bool:
    return criteria_label(subgoal.success_criteria) in _completed_subgoal_keys(
        state,
        goal_key=goal_key,
    )


def _dedupe_subgoals(subgoals: list[Subgoal]) -> list[Subgoal]:
    seen: set[str] = set()
    merged: list[Subgoal] = []
    for sg in subgoals:
        key = criteria_label(sg.success_criteria)
        if key in seen:
            continue
        seen.add(key)
        merged.append(sg)
    return merged


def normalize_subgoals(
    subgoals: list[Subgoal],
    state: WorldState,
    *,
    goal_criteria: list[str],
    allow_final_goal_token: bool = True,
) -> list[Subgoal]:
    """Dedupe, drop satisfied/completed subgoals, trim to max. Never pad from a pool."""
    goal_key = criteria_label(goal_criteria)
    merged = _dedupe_subgoals(subgoals)
    filtered: list[Subgoal] = []
    for index, sg in enumerate(merged):
        sg_key = criteria_label(sg.success_criteria)
        is_final = index == len(merged) - 1
        if sg_key == goal_key and not (allow_final_goal_token and is_final):
            continue
        if _subgoal_already_satisfied(state, sg):
            continue
        if _subgoal_already_completed(state, sg, goal_key=goal_key):
            continue
        filtered.append(sg)
    return filtered[:MAX_SUBGOALS]


def _subgoal_allowed_for_chapter(subgoal: Subgoal, chapter_rank: int) -> bool:
    """Reject only subgoals that jump ahead of the current story chapter."""
    max_rank = _subgoal_max_rank(subgoal)
    if max_rank is not None and max_rank > chapter_rank:
        return False
    return True


def scope_planner_to_chapter(planner: PlannerOutput, state: WorldState) -> PlannerOutput:
    """Clamp LLM output to the current story chapter and plan size (3–5 steps)."""
    from pokemon_hrl.knowledge.planner_knowledge import build_planner_knowledge_from_state

    chapter = infer_chapter_goal(state)
    if chapter.complete:
        planner.subgoal = planner.subgoal[:MAX_SUBGOALS]
        return planner

    knowledge = build_planner_knowledge_from_state(state)
    chapter_facts = knowledge.get("chapter_facts") or {}
    required_target = chapter_facts.get("required_target_map_id")

    chapter_rank = _MILESTONE_RANK.get(criteria_label(chapter.success_criteria), 0)
    goal_key = planner_goal_key(planner)
    goal_rank = _MILESTONE_RANK.get(goal_key)

    if goal_rank is not None and goal_rank > chapter_rank:
        planner.success_criteria = list(chapter.success_criteria)

    if goal_rank is None and planner.success_criteria != chapter.success_criteria:
        planner.success_criteria = list(chapter.success_criteria)

    kept_subgoals: list[Subgoal] = []
    for sg in planner.subgoal:
        if not _subgoal_allowed_for_chapter(sg, chapter_rank):
            continue
        kept_subgoals.append(sg)

    planner.subgoal = normalize_subgoals(
        kept_subgoals,
        state,
        goal_criteria=list(planner.success_criteria),
    )

    at_target = required_target is not None and int(state.map_id) == int(required_target)
    min_subgoals = 1 if at_target else MIN_SUBGOALS
    if len(planner.subgoal) < min_subgoals:
        raise PlannerValidationError(
            f"Need {min_subgoals}–{MAX_SUBGOALS} subgoals for the current chapter "
            f"after scoping; got {len(planner.subgoal)}. "
            "Reason over planner_knowledge and planning_context; "
            "emit only unfinished next steps as memory tokens."
        )

    planner.hint = dict(planner.hint or {})
    if required_target is not None:
        planner.hint["target_map_id"] = int(required_target)
    elif "target_map_id" not in planner.hint:
        planner.hint["target_map_id"] = int(state.map_id)

    if "no_progress" not in planner.failure_criteria:
        planner.failure_criteria = list(planner.failure_criteria) + ["no_progress"]
    return planner
