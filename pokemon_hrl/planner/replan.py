"""When to invoke the planner (LLM or rule-based)."""

from __future__ import annotations

from typing import Any

from pokemon_hrl.types import ProgressResult

VALID_CALL_ON = frozenset({"goal_check", "every_step", "never", "mode_end"})


def should_invoke_planner(
    call_on: str,
    progress: ProgressResult,
    *,
    info: dict[str, Any] | None = None,
    initial: bool = False,
) -> bool:
    if initial:
        return True
    mode = (call_on or "goal_check").strip()
    if mode not in VALID_CALL_ON:
        mode = "goal_check"

    if mode == "never":
        return False
    if mode == "every_step":
        return True
    if mode == "mode_end":
        return bool(progress.done or progress.truncated)

    # goal_check — success/failure or mode ended without success (no_progress path)
    if progress.success or progress.failure:
        return True
    if progress.truncated and info:
        reason = str(info.get("truncated_reason", ""))
        if reason == "mode_max_steps":
            return True
    return False
