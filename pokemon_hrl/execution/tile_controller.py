"""Coordinate-based one-tile movement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyboy.utils import WindowEvent

from pokemon_hrl.execution.action_space import HrlAction
from pokemon_hrl.execution.dialog import clear_dialog_if_needed
from pokemon_hrl.types import ExecutionResult

if TYPE_CHECKING:
    from pokemonred_puffer.environment import RedGymEnv

_TILE_TO_PRESS = {
    HrlAction.TILE_UP: WindowEvent.PRESS_ARROW_UP,
    HrlAction.TILE_DOWN: WindowEvent.PRESS_ARROW_DOWN,
    HrlAction.TILE_LEFT: WindowEvent.PRESS_ARROW_LEFT,
    HrlAction.TILE_RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
}

_TILE_TO_RELEASE = {
    HrlAction.TILE_UP: WindowEvent.RELEASE_ARROW_UP,
    HrlAction.TILE_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    HrlAction.TILE_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    HrlAction.TILE_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
}

_EXPECTED_DELTA = {
    HrlAction.TILE_UP: (0, -1),
    HrlAction.TILE_DOWN: (0, 1),
    HrlAction.TILE_LEFT: (-1, 0),
    HrlAction.TILE_RIGHT: (1, 0),
}


def move_one_tile(env: RedGymEnv, action: int, *, max_substeps: int = 120) -> ExecutionResult:
    """Hold a direction until the player moves one tile or timeout."""
    tile_action = HrlAction(int(action))
    if tile_action not in _TILE_TO_PRESS:
        raise ValueError(f"Not a tile action: {action}")

    x0, y0, m0 = env.get_game_coords()
    dx, dy = _EXPECTED_DELTA[tile_action]
    press = _TILE_TO_PRESS[tile_action]
    release = _TILE_TO_RELEASE[tile_action]

    steps_used = 0
    moved_tile = False
    map_changed = False

    if env.disable_ai_actions:
        env.pyboy.tick(env.action_freq, render=True)
        return ExecutionResult(steps_used=1)

    for _ in range(max_substeps):
        env.pyboy.send_input(press)
        env.pyboy.send_input(release, delay=8)
        env.pyboy.tick(env.action_freq - 1, render=False)
        steps_used += 1

        x1, y1, m1 = env.get_game_coords()
        if m1 != m0:
            map_changed = True
            moved_tile = True
            break
        if (x1 - x0, y1 - y0) == (dx, dy):
            moved_tile = True
            break
        if env.read_m("wIsInBattle") != 0:
            break
    clear_dialog_if_needed(env)
    x1, y1, m1 = env.get_game_coords()
    in_battle = env.read_m("wIsInBattle") != 0
    coords_unchanged = (x1, y1, m1) == (x0, y0, m0)
    tile_direction_blocked = coords_unchanged and not in_battle
    return ExecutionResult(
        steps_used=steps_used,
        moved_tile=moved_tile,
        map_changed=map_changed,
        tile_direction_blocked=tile_direction_blocked,
    )
