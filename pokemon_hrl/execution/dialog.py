"""Dialog / animation input helpers shared by tile and low-level execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokemonred_puffer.environment import RedGymEnv

_JOY_IGNORE_CLEAR_LIMIT = 1000


def clear_dialog_if_needed(env: RedGymEnv) -> None:
    """Advance text and battle animations while ``wJoyIgnore`` blocks player input."""
    for _ in range(_JOY_IGNORE_CLEAR_LIMIT):
        if not env.read_m("wJoyIgnore"):
            break
        env.pyboy.button("a", 8)
        env.pyboy.tick(env.action_freq, render=False)
