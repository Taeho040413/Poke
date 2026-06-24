"""HRL action space: tile movement + low-level buttons."""

from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
from pyboy.utils import WindowEvent

ACTION_DIM = 12


class HrlAction(IntEnum):
    TILE_UP = 0
    TILE_DOWN = 1
    TILE_LEFT = 2
    TILE_RIGHT = 3
    LOW_DOWN = 4
    LOW_LEFT = 5
    LOW_RIGHT = 6
    LOW_UP = 7
    LOW_A = 8
    LOW_B = 9
    LOW_START = 10
    LOW_SELECT = 11


TILE_ACTIONS = frozenset(
    {
        HrlAction.TILE_UP,
        HrlAction.TILE_DOWN,
        HrlAction.TILE_LEFT,
        HrlAction.TILE_RIGHT,
    }
)

LOW_LEVEL_ACTIONS = frozenset(
    {
        HrlAction.LOW_DOWN,
        HrlAction.LOW_LEFT,
        HrlAction.LOW_RIGHT,
        HrlAction.LOW_UP,
        HrlAction.LOW_A,
        HrlAction.LOW_B,
        HrlAction.LOW_START,
        HrlAction.LOW_SELECT,
    }
)

_PRESS_BY_HRL_ACTION = {
    HrlAction.TILE_UP: WindowEvent.PRESS_ARROW_UP,
    HrlAction.TILE_DOWN: WindowEvent.PRESS_ARROW_DOWN,
    HrlAction.TILE_LEFT: WindowEvent.PRESS_ARROW_LEFT,
    HrlAction.TILE_RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
    HrlAction.LOW_DOWN: WindowEvent.PRESS_ARROW_DOWN,
    HrlAction.LOW_LEFT: WindowEvent.PRESS_ARROW_LEFT,
    HrlAction.LOW_RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
    HrlAction.LOW_UP: WindowEvent.PRESS_ARROW_UP,
    HrlAction.LOW_A: WindowEvent.PRESS_BUTTON_A,
    HrlAction.LOW_B: WindowEvent.PRESS_BUTTON_B,
    HrlAction.LOW_START: WindowEvent.PRESS_BUTTON_START,
    HrlAction.LOW_SELECT: WindowEvent.PRESS_BUTTON_SELECT,
}


def make_action_space() -> gym.spaces.Discrete:
    return gym.spaces.Discrete(ACTION_DIM)


_TILE_DELTA = {
    HrlAction.TILE_UP: (0, -1),
    HrlAction.TILE_DOWN: (0, 1),
    HrlAction.TILE_LEFT: (-1, 0),
    HrlAction.TILE_RIGHT: (1, 0),
}


def tile_target_coords(x: int, y: int, direction: int) -> tuple[int, int]:
    """Return the tile coordinate an agent at ``(x, y)`` would enter."""
    dx, dy = _TILE_DELTA[HrlAction(int(direction))]
    return int(x) + dx, int(y) + dy


def is_tile_action(action: int) -> bool:
    return int(action) in TILE_ACTIONS


def is_low_level_action(action: int) -> bool:
    return int(action) in LOW_LEVEL_ACTIONS


def action_name(action: int) -> str:
    return HrlAction(int(action)).name


def hrl_action_to_press_event(action: int) -> WindowEvent | None:
    try:
        return _PRESS_BY_HRL_ACTION[HrlAction(int(action))]
    except (ValueError, KeyError):
        return None
