"""Sync active PlannerOutput into env wrappers."""

from __future__ import annotations

from pokemon_hrl.env.base_hrl_env import _sync_target_map_to_base
from pokemon_hrl.types import PlannerOutput


def _walk_wrappers(env):
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        next_env = getattr(current, "env", None)
        if next_env is current:
            break
        current = next_env


def sync_planner_to_env(env, planner: PlannerOutput) -> None:
    subgoal_index = 0
    for wrapper in _walk_wrappers(env):
        if hasattr(wrapper, "set_planner"):
            wrapper.set_planner(planner)
        if hasattr(wrapper, "_subgoal_index"):
            subgoal_index = int(wrapper._subgoal_index)
    _sync_target_map_to_base(env, planner, subgoal_index=subgoal_index)


def sync_subgoal_index_to_env(env, index: int) -> None:
    for wrapper in _walk_wrappers(env):
        sync_hook = getattr(wrapper, "sync_subgoal_index", None)
        if callable(sync_hook):
            sync_hook(index)


def sync_after_external_state_change(
    env,
    *,
    global_step: int | None = None,
    subgoal_index: int | None = None,
) -> None:
    for wrapper in _walk_wrappers(env):
        sync_hook = getattr(wrapper, "sync_after_external_state_change", None)
        if callable(sync_hook):
            sync_hook(global_step=global_step, subgoal_index=subgoal_index)
            return
