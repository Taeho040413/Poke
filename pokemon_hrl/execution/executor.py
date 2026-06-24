"""Dispatch HRL actions to tile controller or low-level buttons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pokemon_hrl.execution import action_space, low_level, tile_controller
from pokemon_hrl.execution.dialog import clear_dialog_if_needed
from pokemon_hrl.types import ExecutionResult

if TYPE_CHECKING:
    from pokemonred_puffer.environment import RedGymEnv


@dataclass
class ActionExecutor:
    max_tile_substeps: int = 120

    def execute(self, env: RedGymEnv, action: int) -> ExecutionResult:
        action = int(action)
        if action_space.is_tile_action(action):
            return tile_controller.move_one_tile(
                env, action, max_substeps=self.max_tile_substeps
            )
        if action_space.is_low_level_action(action):
            low_level.press_release_low_level(env, action)
            clear_dialog_if_needed(env)
            return ExecutionResult(steps_used=1)
        raise ValueError(f"Unknown HRL action: {action}")

    def run_post_action_hooks(self, env: RedGymEnv, pressed_action: int | None = None) -> None:
        """Mirror RedGymEnv.run_action_on_emulator post-action automation."""
        from pokemonred_puffer.data.tm_hm import (
            CUT_SPECIES_IDS,
            STRENGTH_SPECIES_IDS,
            SURF_SPECIES_IDS,
            TmHmMoves,
        )
        from pokemon_hrl.execution.action_space import hrl_action_to_press_event

        env.update_seen_coords()

        if env.events.get_event("EVENT_GOT_HM01"):
            if env.auto_teach_cut and not env.check_if_party_has_hm(TmHmMoves.CUT.value):
                env.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
            if env.auto_use_cut:
                env.cut_if_next()

        if env.events.get_event("EVENT_GOT_HM03"):
            if env.auto_teach_surf and not env.check_if_party_has_hm(TmHmMoves.SURF.value):
                env.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
            if env.auto_use_surf and pressed_action is not None:
                press_event = hrl_action_to_press_event(pressed_action)
                if press_event is not None:
                    env.surf_if_attempt(press_event)

        if env.events.get_event("EVENT_GOT_HM04"):
            if env.auto_teach_strength and not env.check_if_party_has_hm(TmHmMoves.STRENGTH.value):
                env.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
            if env.auto_solve_strength_puzzles:
                env.solve_strength_puzzle()
            if not env.check_if_party_has_hm(TmHmMoves.STRENGTH.value) and env.auto_use_strength:
                env.use_strength()

        if env.events.get_event("EVENT_GOT_POKE_FLUTE") and env.auto_pokeflute:
            env.use_pokeflute()

        if env.get_game_coords() == (18, 4, 7) and env.skip_safari_zone:
            env.skip_safari_zone_atn()

        if env.auto_next_elevator_floor:
            env.next_elevator_floor()

        if env.insert_saffron_guard_drinks:
            env.insert_guard_drinks()

        env.pyboy.tick(1, render=True)
