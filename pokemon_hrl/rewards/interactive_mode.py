"""Interactive Mode reward — interaction + exploration shaping (v1)."""

from __future__ import annotations

from pokemonred_puffer.rewards.baseline import ExplorationInteractionRewardEnv


class InteractiveModeRewardEnv(ExplorationInteractionRewardEnv):
    """Reward env for Interactive mode training."""

    def get_game_state_reward(self) -> dict[str, float]:
        self._seed_reward_state_if_needed()
        return {
            "event": self._reward("event") * self.update_max_event_rew(),
            "item": self._reward("item") * self.item_count,
            "gym_core_npc": self._reward("gym_core_npc") * self.gym_core_npc_count,
            "npc_first_talk": self._reward("npc_first_talk") * self.first_npc_talk_count,
            "object_first_interaction": self._reward("object_first_interaction")
            * self.first_object_interaction_count,
            "new_tile": self._reward("new_tile") * self.new_tile_count,
            "new_building": self._reward("new_building") * self.new_building_count,
            "new_room": self._reward("new_room") * self.new_room_count,
            "new_npc_textbox": self._reward("new_npc_textbox") * self.new_npc_textbox_count,
            "step_penalty": self._reward("step_penalty") * self.step_count,
            "repeat_npc_penalty": self._reward("repeat_npc_penalty")
            * self.repeat_npc_interaction_count,
            "repeat_object_penalty": self._reward("repeat_object_penalty")
            * self.repeat_object_interaction_count,
            "invalid_interaction": self._reward("invalid_interaction")
            * self.invalid_interaction_count,
            "start_menu_penalty": self._reward("start_menu_penalty")
            * self.start_menu_open_count,
            "stuck_penalty": self._reward("stuck_penalty") * self.stuck_penalty_count,
            "blocked_tile_retry": self._reward("blocked_tile_retry")
            * getattr(self, "blocked_tile_retry_count", 0),
            "wild_encounter_penalty": self._reward("wild_encounter_penalty")
            * self.wild_encounter_count,
            "trainer_battle_win": self._reward("trainer_battle_win")
            * self.trainer_battle_win_count,
            "wild_battle_win": self._reward("wild_battle_win")
            * self.wild_battle_win_count,
            "death": self._reward("death") * self.death_count,
            "pokecenter_first_entry": self._reward("pokecenter_first_entry")
            * self.pokecenter_first_entry_count,
            "pokemon_heal_hp": self._reward("pokemon_heal_hp")
            * self.pokecenter_heal_hp_count,
            "new_map": self._reward("new_map") * self.new_map_count,
            "target_map_entry": self._reward("target_map_entry")
            * self.target_map_entry_count,
            "party_level": self._reward("party_level") * self.party_level_count,
        }
