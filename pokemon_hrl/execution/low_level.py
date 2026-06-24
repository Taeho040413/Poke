"""Low-level PyBoy button actions including SELECT."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyboy.utils import WindowEvent

from pokemon_hrl.execution.action_space import HrlAction

if TYPE_CHECKING:
    from pokemonred_puffer.environment import RedGymEnv

_PRESS_BY_ACTION = {
    HrlAction.LOW_DOWN: WindowEvent.PRESS_ARROW_DOWN,
    HrlAction.LOW_LEFT: WindowEvent.PRESS_ARROW_LEFT,
    HrlAction.LOW_RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
    HrlAction.LOW_UP: WindowEvent.PRESS_ARROW_UP,
    HrlAction.LOW_A: WindowEvent.PRESS_BUTTON_A,
    HrlAction.LOW_B: WindowEvent.PRESS_BUTTON_B,
    HrlAction.LOW_START: WindowEvent.PRESS_BUTTON_START,
    HrlAction.LOW_SELECT: WindowEvent.PRESS_BUTTON_SELECT,
}

_RELEASE_BY_ACTION = {
    HrlAction.LOW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    HrlAction.LOW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    HrlAction.LOW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    HrlAction.LOW_UP: WindowEvent.RELEASE_ARROW_UP,
    HrlAction.LOW_A: WindowEvent.RELEASE_BUTTON_A,
    HrlAction.LOW_B: WindowEvent.RELEASE_BUTTON_B,
    HrlAction.LOW_START: WindowEvent.RELEASE_BUTTON_START,
    HrlAction.LOW_SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
}


def press_release_low_level(env: RedGymEnv, action: int) -> None:
    """Press and release one low-level button, mirroring RedGymEnv timing."""
    hrl_action = HrlAction(int(action))
    if hrl_action not in _PRESS_BY_ACTION:
        raise ValueError(f"Not a low-level action: {action}")

    if env.disable_ai_actions:
        env.pyboy.tick(env.action_freq, render=True)
        return

    press = _PRESS_BY_ACTION[hrl_action]
    release = _RELEASE_BY_ACTION[hrl_action]

    if press == WindowEvent.PRESS_BUTTON_A:
        env.update_a_press()

    env.pyboy.send_input(press)
    env.pyboy.send_input(release, delay=8)
    env.pyboy.tick(env.action_freq - 1, render=False)
