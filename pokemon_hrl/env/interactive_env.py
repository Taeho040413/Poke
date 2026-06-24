"""Interactive-mode HRL environment with tile + low-level actions."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from pokemon_hrl.env.goal_memory import (
    GoalMemoryConfig,
    GoalMemoryTracker,
    front_tile_coords,
    goal_memory_observation_space,
)
from pokemon_hrl.execution.action_masks import apply_action_mask, compute_hrl_action_mask
from pokemon_hrl.execution.action_space import (
    ACTION_DIM,
    HrlAction,
    is_tile_action,
    make_action_space,
    tile_target_coords,
)
from pokemon_hrl.execution.executor import ActionExecutor
from pokemon_hrl.execution.npc_tiles import (
    build_local_npc_map,
    npc_local_observation_space,
    read_npc_coords,
)
from pokemon_hrl.execution.tile_blocked import TileBlockedTracker, blocked_tile_observation_space
from pokemon_hrl.rewards.interactive_mode import InteractiveModeRewardEnv
from pokemonred_puffer.global_map import local_to_global


class HrlInteractiveRewardEnv(InteractiveModeRewardEnv):
    """Interactive-mode reward env with HRL action space (tile + low-level)."""

    def __init__(
        self,
        env_config,
        reward_config,
        *,
        max_tile_substeps: int = 120,
        battle_action_mask: bool = True,
        tile_move_enabled: bool = True,
        tile_blocked_enabled: bool = True,
        tile_blocked_ttl_steps: int = 100,
        tile_blocked_weaken_weight: float = 0.05,
        tile_blocked_retry_window_steps: int = 50,
        tile_blocked_confidence_threshold: int = 1,
        goal_memory_config: GoalMemoryConfig | None = None,
    ):
        super().__init__(env_config, reward_config)
        self.action_space = make_action_space()
        self.action_hist = np.zeros(ACTION_DIM)
        self._executor = ActionExecutor(max_tile_substeps=max_tile_substeps)
        self._trainer_battle_loss_count = 0
        self.blocked_tile_retry_count = 0
        self._battle_action_mask = bool(battle_action_mask)
        self._tile_move_enabled = bool(tile_move_enabled)
        self._tile_blocked = TileBlockedTracker(
            enabled=tile_blocked_enabled,
            ttl_steps=tile_blocked_ttl_steps,
            weaken_weight=tile_blocked_weaken_weight,
            retry_window_steps=tile_blocked_retry_window_steps,
            confidence_threshold=tile_blocked_confidence_threshold,
        )
        self._goal_memory_cfg = goal_memory_config or GoalMemoryConfig(enabled=False)
        self._goal_memory = GoalMemoryTracker(self._goal_memory_cfg)
        self._blocked_tile_local_radius = int(self._goal_memory_cfg.local_radius)
        obs_spaces = {
            **dict(self.observation_space.spaces),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(ACTION_DIM,), dtype=np.float32
            ),
        }
        obs_spaces.update(goal_memory_observation_space(self._goal_memory_cfg))
        obs_spaces.update(npc_local_observation_space(radius=self._blocked_tile_local_radius))
        if tile_blocked_enabled:
            obs_spaces.update(
                blocked_tile_observation_space(radius=self._blocked_tile_local_radius)
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self._goal_memory_new_tile = False

    def set_goal_context(self, goal_context: dict[str, Any] | None) -> None:
        self._goal_memory.set_context(goal_context)
        if goal_context is not None:
            map_id = goal_context.get("target_map_id")
            if map_id is not None:
                super().set_hrl_target_map_id(int(map_id))

    def set_hrl_target_map_id(self, map_id: int | None) -> None:
        super().set_hrl_target_map_id(map_id)
        if self._goal_memory_cfg.enabled:
            ctx = self._goal_memory.context
            self._goal_memory.set_context(
                {
                    "target_map_id": int(map_id) if map_id is not None else None,
                    "target_event_id": ctx.target_event_id,
                    "target_object_id": ctx.target_object_id,
                    "target_x": ctx.target_x,
                    "target_y": ctx.target_y,
                    "goal_key": ctx.goal_key,
                }
            )

    def is_a_press_action(self, action: int) -> bool:
        return int(action) == int(HrlAction.LOW_A)

    def get_action_mask(self) -> np.ndarray:
        x, y, map_id = self.get_game_coords()
        tile_weights = self._tile_blocked.tile_mask_weights(map_id, x, y)
        return compute_hrl_action_mask(
            int(self.read_m("wIsInBattle")),
            enabled=self._battle_action_mask,
            tile_move_enabled=self._tile_move_enabled,
            weakened_tile_weights=tile_weights,
        )

    def _apply_action_mask(self, action: int) -> int:
        return apply_action_mask(int(action), self.get_action_mask())

    def _current_npc_coords(self) -> set[tuple[int, int]]:
        return read_npc_coords(self)

    def _attach_goal_memory_obs(self, obs: dict) -> dict:
        if not self._goal_memory_cfg.enabled:
            return obs
        x, y, map_id = self.get_game_coords()
        npc_coords = self._current_npc_coords()
        merged = dict(obs)
        merged.update(
            self._goal_memory.build_obs(
                map_id=int(map_id),
                player_x=int(x),
                player_y=int(y),
                tile_blocked=self._tile_blocked,
                npc_coords=npc_coords,
            )
        )
        return merged

    def _attach_blocked_tile_obs(self, obs: dict) -> dict:
        x, y, map_id = self.get_game_coords()
        npc_coords = self._current_npc_coords()
        merged = dict(obs)
        merged["npc_local"] = build_local_npc_map(
            self,
            player_x=int(x),
            player_y=int(y),
            radius=self._blocked_tile_local_radius,
        )
        if not self._tile_blocked.enabled:
            return merged
        merged["blocked_tile_local"] = self._tile_blocked.build_local_blocked_map(
            map_id=int(map_id),
            player_x=int(x),
            player_y=int(y),
            radius=self._blocked_tile_local_radius,
            exclude_coords=npc_coords,
        )
        return merged

    def _attach_hrl_obs_extensions(self, obs: dict) -> dict:
        return self._attach_blocked_tile_obs(self._attach_goal_memory_obs(obs))

    def _get_obs(self):
        obs = super()._get_obs()
        obs["action_mask"] = self.get_action_mask()
        return self._attach_hrl_obs_extensions(obs)

    def step(self, action):
        if self._goal_memory_cfg.enabled:
            self._goal_memory.on_step_start(self.events)
        obs, reward, terminated, truncated, info = super().step(self._apply_action_mask(action))
        if self._goal_memory_cfg.enabled:
            reward += self._goal_memory.consume_step_reward()
            x, y, map_id = self.get_game_coords()
            info = dict(info or {})
            info.update(self._goal_memory.info_fields(map_id=int(map_id)))
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.action_hist = np.zeros(ACTION_DIM)
        self._trainer_battle_loss_count = 0
        self.blocked_tile_retry_count = 0
        self._tile_blocked.reset()
        self._goal_memory.reset()
        if "action_mask" not in obs:
            obs = dict(obs)
            obs["action_mask"] = self.get_action_mask()
        obs = self._attach_hrl_obs_extensions(obs)
        if self._goal_memory_cfg.enabled:
            info = dict(info or {})
            x, y, map_id = self.get_game_coords()
            info.update(self._goal_memory.info_fields(map_id=int(map_id)))
        return obs, info

    def _ensure_hrl_action_hist(self) -> None:
        if self.action_hist.shape[0] != ACTION_DIM:
            self.action_hist = np.zeros(ACTION_DIM)

    def run_action_on_emulator(self, action):
        action = int(action)
        self._ensure_hrl_action_hist()
        self.action_hist[action] += 1
        self._seed_reward_state_if_needed()
        map_before = int(self.read_m("wCurMap"))
        ts_before = int(self.read_m("wCurMapTileset"))
        self._interaction_triggered_this_step = False
        pressed_a = action == int(HrlAction.LOW_A)

        hp_sum_before = int(self._read_party_hp_sum())
        prev_pokecenter_heal = int(self.pokecenter_heal)
        prev_blackout_count = int(self.blackout_count)
        prev_is_in_battle = int(self.read_m("wIsInBattle"))
        party_n = max(0, min(int(self.read_m("wPartyCount")), 6))
        hp_before_slots = [
            int(self.read_short(f"wPartyMon{i+1}HP")) for i in range(party_n)
        ]

        x0, y0, m0 = self.get_game_coords()

        exec_result = self._executor.execute(self, action)
        x1, y1, m1 = self.get_game_coords()
        self._goal_memory_new_tile = (int(x1), int(y1), int(m1)) not in self._seen_unique_coords

        if is_tile_action(action):
            if exec_result.tile_direction_blocked:
                move_result = self._tile_blocked.record_failed_move(
                    m0,
                    x0,
                    y0,
                    action,
                    step=int(self.step_count),
                    skip_coords=self._current_npc_coords(),
                )
                if move_result.retry_penalty:
                    self.blocked_tile_retry_count += 1
            elif exec_result.moved_tile:
                tx, ty = tile_target_coords(x0, y0, action)
                self._tile_blocked.clear_target(m0, tx, ty)
        self._tile_blocked.tick()
        self._executor.run_post_action_hooks(self, pressed_action=action)

        map_after = int(self.read_m("wCurMap"))
        ts_after = int(self.read_m("wCurMapTileset"))
        if self._goal_memory_cfg.enabled and map_before != map_after:
            if is_tile_action(action) and exec_result.moved_tile:
                warp_x, warp_y = tile_target_coords(x0, y0, action)
            else:
                warp_x, warp_y = x0, y0
            self._goal_memory.record_warp(m0, warp_x, warp_y)

        if self._reward_state_seeded and map_before != map_after:
            self._apply_map_change_structure_reward(map_before, ts_before, map_after, ts_after)
            self._register_map_transition_rewards(map_before, map_after)

        current_blackout_map_id = int(self.read_m("wLastBlackoutMap"))
        self._last_blackout_map_id = current_blackout_map_id

        did_blackout = int(self.blackout_count) > prev_blackout_count
        if did_blackout:
            self._suppress_pokecenter_shaping_after_blackout = True
            if prev_is_in_battle == 2:
                self._trainer_battle_loss_count += 1

        post_battle = int(self.read_m("wIsInBattle"))
        battle_ctx = post_battle if post_battle in (1, 2) else prev_is_in_battle
        party_n2 = max(0, min(int(self.read_m("wPartyCount")), 6))
        for i in range(min(party_n, party_n2)):
            hp_after_slot = int(self.read_short(f"wPartyMon{i+1}HP"))
            if hp_before_slots[i] > 0 and hp_after_slot == 0 and battle_ctx == 1:
                self.death_count += 1

        self._register_battle_end_rewards(
            prev_is_in_battle=prev_is_in_battle,
            post_battle=post_battle,
            prev_blackout_count=prev_blackout_count,
        )

        if prev_is_in_battle == 0 and post_battle == 1:
            self.wild_encounter_count += 1
            enc_x, enc_y, enc_map = self.get_game_coords()
            _gy, _gx = local_to_global(enc_y, enc_x, enc_map)
            if 0 <= _gy < self.wild_encounter_tile_map.shape[0] and (
                0 <= _gx < self.wild_encounter_tile_map.shape[1]
            ):
                self.wild_encounter_tile_map[_gy, _gx] = min(
                    self.wild_encounter_tile_map[_gy, _gx] + 1.0, 1e4
                )

        if self.pokecenter_heal == 1 and prev_pokecenter_heal == 0:
            hp_sum_after = int(self._read_party_hp_sum())
            if (
                hp_sum_before > 0
                and not did_blackout
                and not self._suppress_pokecenter_shaping_after_blackout
            ):
                healed = max(0, hp_sum_after - hp_sum_before)
                current_map_id = int(self.read_m("wCurMap"))
                current_tileset = int(self.read_m("wCurMapTileset"))
                self._ensure_pokecenter_entry_recorded(current_map_id, current_tileset)
                self.pokecenter_heal_hp_count += healed

        if int(self.pokecenter_heal) == 1:
            self.pokecenter_heal = 0

        self._update_bag_item_tracking()
        self._update_script_and_text_tracking()
        self._update_party_level_tracking()

        interaction_fail = False
        if (
            pressed_a
            and not self._interaction_triggered_this_step
            and self.read_m("wIsInBattle") == 0
            and not self._textbox_active()
        ):
            self.invalid_interaction_count += 1
            interaction_fail = True

        if self._goal_memory_cfg.enabled:
            x_final, y_final, m_final = self.get_game_coords()
            interact_x, interact_y = x_final, y_final
            if pressed_a or self._interaction_triggered_this_step or interaction_fail:
                facing = None
                try:
                    facing = int(self.read_m("wSpritePlayerStateData1FacingDirection"))
                except Exception:
                    facing = None
                front = front_tile_coords(x_final, y_final, facing)
                if front is not None:
                    interact_x, interact_y = front
            if self._interaction_triggered_this_step:
                self._goal_memory.record_interact_success(m_final, interact_x, interact_y)
            if interaction_fail:
                self._goal_memory.record_interact_fail(m_final, interact_x, interact_y)
            self._goal_memory.on_post_step(
                map_id=int(m_final),
                x=int(x_final),
                y=int(y_final),
                map_before=map_before,
                map_after=map_after,
                new_tile_on_map=self._goal_memory_new_tile,
                interaction_success=bool(self._interaction_triggered_this_step),
                interaction_fail=interaction_fail,
                events_reader=self.events,
                interaction_x=interact_x,
                interaction_y=interact_y,
            )
