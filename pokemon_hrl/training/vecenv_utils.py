"""Helpers to iterate pufferlib vecenv agents and broadcast HRL sync."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterator

from pokemon_hrl.env.planner_sync import (
    sync_after_external_state_change,
    sync_planner_to_env,
    sync_subgoal_index_to_env,
)
from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.types import PlannerOutput


def iter_vecenv_agents(vecenv: Any) -> Iterator[Any]:
    """Yield each agent's top-level gym wrapper from a pufferlib vecenv."""
    envs = getattr(vecenv, "envs", None)
    if envs is not None:
        yield from envs
        return
    driver = getattr(vecenv, "driver_env", None)
    if driver is not None:
        yield driver
        return
    yield vecenv


def base_env_for_agent(vecenv: Any, agent_index: int = 0) -> Any:
    agents = list(iter_vecenv_agents(vecenv))
    if not agents:
        raise ValueError("vecenv has no agent environments")
    index = max(0, min(int(agent_index), len(agents) - 1))
    return unwrap_hrl_env(agents[index])


def agent_index_for_env_id(vecenv: Any, env_id: int | None) -> int:
    """Map RedGymEnv ``env_id`` to vecenv agent index."""
    if env_id is None:
        return 0
    target = int(env_id)
    for index, agent in enumerate(iter_vecenv_agents(vecenv)):
        base = unwrap_hrl_env(agent)
        if int(getattr(base, "env_id", -1)) == target:
            return index
    return 0


def broadcast_planner(vecenv: Any, planner: PlannerOutput) -> None:
    for agent in iter_vecenv_agents(vecenv):
        sync_planner_to_env(agent, planner)


def broadcast_subgoal_soft_sync(
    vecenv: Any,
    subgoal_index: int,
    *,
    global_step: int | None = None,
) -> None:
    """Advance shared subgoal metadata on every env without loading game state."""
    for agent in iter_vecenv_agents(vecenv):
        sync_subgoal_index_to_env(agent, subgoal_index)
        sync_after_external_state_change(
            agent,
            global_step=global_step,
            subgoal_index=subgoal_index,
        )


def broadcast_hard_reset(
    vecenv: Any,
    save_point_path: Path | None,
    *,
    global_step: int | None = None,
    subgoal_index: int = 0,
) -> None:
    """Load save point on every env and align progress metadata."""
    for agent in iter_vecenv_agents(vecenv):
        base = unwrap_hrl_env(agent)
        if save_point_path is not None and save_point_path.is_file():
            with open(save_point_path, "rb") as state_file:
                base.pyboy.load_state(state_file)
        sync_subgoal_index_to_env(agent, subgoal_index)
        sync_after_external_state_change(
            agent,
            global_step=global_step,
            subgoal_index=subgoal_index,
        )


def pin_save_point_on_all_agents(vecenv: Any, save_point_path: Path) -> None:
    resolved = save_point_path.resolve()
    for agent in iter_vecenv_agents(vecenv):
        base = unwrap_hrl_env(agent)
        base.init_state_path = resolved
        base.init_state_name = resolved.stem


def save_pyboy_state(base_env: Any) -> bytes:
    buffer = io.BytesIO()
    base_env.pyboy.save_state(buffer)
    return buffer.getvalue()
