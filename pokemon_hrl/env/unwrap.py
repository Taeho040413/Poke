"""Environment unwrapping helpers."""

from __future__ import annotations

import gymnasium as gym

from pokemon_hrl.env.interactive_env import HrlInteractiveRewardEnv


def _is_puffer_dtype_namespace(obj: object) -> bool:
    """PufferLib ``GymnasiumPufferEnv.emulated`` holds observation dtype metadata."""
    return type(obj).__name__ == "namespace" and hasattr(obj, "observation_dtype")


def unwrap_hrl_env(env: gym.Env) -> HrlInteractiveRewardEnv:
    current: gym.Env | HrlInteractiveRewardEnv = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, HrlInteractiveRewardEnv):
            return current
        if _is_puffer_dtype_namespace(current):
            break
        if isinstance(current, gym.Wrapper):
            current = current.env
            continue
        inner = getattr(current, "env", None)
        if inner is not None and inner is not current:
            current = inner
            continue
        break
    if isinstance(current, HrlInteractiveRewardEnv):
        return current
    raise TypeError(f"Could not find HrlInteractiveRewardEnv in wrapper chain: {env!r}")
