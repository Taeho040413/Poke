"""Goal-conditioned local memory maps and compact goal vector for RL observations."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from pokemon_hrl.execution.tile_blocked import TileBlockedTracker

Coord = tuple[int, int]

# wSpritePlayerStateData1FacingDirection raw values.
_FACING_FRONT_DELTA: dict[int, tuple[int, int]] = {
    0: (0, 1),  # down
    4: (0, -1),  # up
    8: (-1, 0),  # left
    12: (1, 0),  # right
}

_PLAYER_COORD_NORM = 255.0


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _norm_id(value: int | str | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int):
        return float(value % 256) / 255.0
    # Stable compact encoding for string event/object keys.
    return float(sum(ord(c) for c in str(value)) % 256) / 255.0


def front_tile_coords(
    x: int,
    y: int,
    facing_direction: int | None,
) -> tuple[int, int] | None:
    """Tile in front of the player for raw facing direction (0/4/8/12)."""
    if facing_direction is None:
        return None
    facing = int(facing_direction)
    if facing not in _FACING_FRONT_DELTA and 0 <= facing <= 3:
        facing *= 4
    delta = _FACING_FRONT_DELTA.get(facing)
    if delta is None:
        return None
    dx, dy = delta
    return int(x) + dx, int(y) + dy


@dataclass
class GoalContext:
    target_map_id: int | None = None
    target_event_id: int | str | None = None
    target_object_id: int | str | None = None
    target_x: int | None = None
    target_y: int | None = None
    goal_key: str | None = None


@dataclass
class GoalMemoryConfig:
    enabled: bool = True
    local_radius: int = 5
    include_visited: bool = True
    include_blocked: bool = True
    include_warps: bool = True
    include_interactions: bool = True
    include_goal_vector: bool = True
    target_map_enter_reward: float = 0.0
    target_map_explore_reward: float = 0.02
    target_event_done_reward: float = 2.0
    target_interaction_reward: float = 0.2
    # Unused in GoalMemoryTracker; interactive_mode.reward.blocked_tile_retry shapes retries.
    blocked_repeat_penalty: float = 0.0

    @classmethod
    def from_omega(cls, config: Any, prefix: str = "hrl.goal_memory") -> GoalMemoryConfig:
        from omegaconf import OmegaConf

        kwargs: dict[str, Any] = {}
        for field in fields(cls):
            path = f"{prefix}.{field.name}"
            kwargs[field.name] = OmegaConf.select(config, path, default=field.default)
        return cls(**kwargs)


@dataclass
class GoalMemoryStepRewards:
    map_enter: float = 0.0
    map_explore: float = 0.0
    event_done: float = 0.0
    interaction: float = 0.0
    blocked_penalty: float = 0.0

    def total(self) -> float:
        return (
            self.map_enter
            + self.map_explore
            + self.event_done
            + self.interaction
            + self.blocked_penalty
        )


class GoalMemoryTracker:
    """Per-map local memory and goal-scoped reward shaping."""

    def __init__(self, config: GoalMemoryConfig):
        self.config = config
        self.context = GoalContext()
        self._visited: dict[int, set[Coord]] = {}
        self._warps: dict[int, set[Coord]] = {}
        self._interact_success: dict[int, set[Coord]] = {}
        self._interact_fail: dict[int, set[Coord]] = {}
        self._goal_explored: set[Coord] = set()
        self._entered_target_map_for_goal: bool = False
        self._target_event_was_set: bool | None = None
        self._step_rewards = GoalMemoryStepRewards()
        self._last_local_counts: tuple[int, int] = (0, 0)
        self._last_goal_vector: tuple[float, float] = (0.0, 0.0)
        self._last_map_id: int | None = None

    @property
    def local_size(self) -> int:
        return 2 * int(self.config.local_radius) + 1

    def reset(self) -> None:
        self._visited.clear()
        self._warps.clear()
        self._interact_success.clear()
        self._interact_fail.clear()
        self._reset_goal_scoped_state()
        self._step_rewards = GoalMemoryStepRewards()
        self._target_event_was_set = None
        self._last_map_id = None

    def set_context(self, goal_context: dict[str, Any] | GoalContext | None) -> None:
        if goal_context is None:
            new_ctx = GoalContext()
        elif isinstance(goal_context, GoalContext):
            new_ctx = goal_context
        else:
            new_ctx = GoalContext(
                target_map_id=goal_context.get("target_map_id"),
                target_event_id=goal_context.get("target_event_id"),
                target_object_id=goal_context.get("target_object_id"),
                target_x=goal_context.get("target_x"),
                target_y=goal_context.get("target_y"),
                goal_key=goal_context.get("goal_key"),
            )
        if (
            new_ctx.goal_key != self.context.goal_key
            or new_ctx.target_map_id != self.context.target_map_id
            or new_ctx.target_event_id != self.context.target_event_id
            or new_ctx.target_object_id != self.context.target_object_id
            or new_ctx.target_x != self.context.target_x
            or new_ctx.target_y != self.context.target_y
        ):
            self._reset_goal_scoped_state()
        self.context = new_ctx

    def _reset_goal_scoped_state(self) -> None:
        self._goal_explored.clear()
        self._entered_target_map_for_goal = False
        self._target_event_was_set = None

    def _map_set(self, store: dict[int, set[Coord]], map_id: int, x: int, y: int) -> None:
        m = int(map_id)
        if m not in store:
            store[m] = set()
        store[m].add((int(x), int(y)))

    def record_visit(self, map_id: int, x: int, y: int) -> None:
        self._map_set(self._visited, map_id, x, y)

    def record_warp(self, map_id: int, x: int, y: int) -> None:
        if self.config.include_warps:
            self._map_set(self._warps, map_id, x, y)

    def record_interact_success(self, map_id: int, x: int, y: int) -> None:
        if self.config.include_interactions:
            self._map_set(self._interact_success, map_id, x, y)

    def record_interact_fail(self, map_id: int, x: int, y: int) -> None:
        if self.config.include_interactions:
            self._map_set(self._interact_fail, map_id, x, y)

    def on_step_start(self, events_reader: Any) -> None:
        self._step_rewards = GoalMemoryStepRewards()
        self._target_event_was_set = self._read_target_event(events_reader)

    def on_post_step(
        self,
        *,
        map_id: int,
        x: int,
        y: int,
        map_before: int,
        map_after: int,
        new_tile_on_map: bool,
        interaction_success: bool,
        interaction_fail: bool,
        events_reader: Any,
        interaction_x: int | None = None,
        interaction_y: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return

        self.record_visit(map_id, x, y)

        target_map = self.context.target_map_id
        if (
            target_map is not None
            and int(map_after) == int(target_map)
            and not self._entered_target_map_for_goal
            and map_before != map_after
        ):
            self._entered_target_map_for_goal = True
            self._step_rewards.map_enter += float(self.config.target_map_enter_reward)

        on_target = target_map is not None and int(map_id) == int(target_map)

        if on_target and new_tile_on_map:
            key = (int(map_id), int(x), int(y))
            if key not in self._goal_explored:
                self._goal_explored.add(key)
                self._step_rewards.map_explore += float(self.config.target_map_explore_reward)

        if interaction_success:
            ix = int(interaction_x) if interaction_x is not None else int(x)
            iy = int(interaction_y) if interaction_y is not None else int(y)
            if self._near_goal_target(map_id, ix, iy):
                self._step_rewards.interaction += float(self.config.target_interaction_reward)

        if self._target_event_rising_edge(events_reader):
            self._step_rewards.event_done += float(self.config.target_event_done_reward)

        self._last_map_id = int(map_id)

    def _read_target_event(self, events_reader: Any) -> bool | None:
        flag_name = self._target_event_flag_name()
        if flag_name is None:
            return None
        try:
            return bool(events_reader.get_event(flag_name))
        except Exception:
            return None

    def _target_event_flag_name(self) -> str | None:
        target = self.context.target_event_id
        if target is None:
            return None
        if isinstance(target, str):
            if target.startswith("flag:"):
                return target.split(":", 1)[1]
            return target
        return None

    def _target_event_rising_edge(self, events_reader: Any) -> bool:
        flag_name = self._target_event_flag_name()
        if flag_name is None:
            return False
        try:
            now = bool(events_reader.get_event(flag_name))
        except Exception:
            return False
        if not now:
            return False
        if self._target_event_was_set is None:
            return True
        return not self._target_event_was_set

    def _near_goal_target(self, map_id: int, x: int, y: int) -> bool:
        r = int(self.config.local_radius)
        ctx = self.context
        if ctx.target_map_id is not None and int(map_id) != int(ctx.target_map_id):
            return False
        if ctx.target_x is not None and ctx.target_y is not None:
            return abs(int(x) - int(ctx.target_x)) <= r and abs(int(y) - int(ctx.target_y)) <= r
        # Object-only hints need a coordinate resolver before proximity rewards apply.
        return False

    def build_local_channel(
        self,
        coords: set[Coord],
        *,
        player_x: int,
        player_y: int,
    ) -> np.ndarray:
        r = int(self.config.local_radius)
        size = self.local_size
        out = np.zeros((size, size), dtype=np.float32)
        for tx, ty in coords:
            lx = int(tx) - int(player_x)
            ly = int(ty) - int(player_y)
            if abs(lx) > r or abs(ly) > r:
                continue
            out[ly + r, lx + r] = 1.0
        return out

    def build_local_maps(
        self,
        *,
        map_id: int,
        player_x: int,
        player_y: int,
        tile_blocked: TileBlockedTracker | None,
        npc_coords: set[Coord] | None = None,
    ) -> np.ndarray:
        cfg = self.config
        channels: list[np.ndarray] = []
        m = int(map_id)

        if cfg.include_visited:
            channels.append(
                self.build_local_channel(
                    self._visited.get(m, set()),
                    player_x=player_x,
                    player_y=player_y,
                )
            )
        if cfg.include_blocked:
            blocked: set[Coord] = set()
            if tile_blocked is not None:
                blocked = tile_blocked.blocked_coords_for_map(
                    m,
                    exclude_coords=npc_coords,
                )
            channels.append(
                self.build_local_channel(blocked, player_x=player_x, player_y=player_y)
            )
        if cfg.include_warps:
            channels.append(
                self.build_local_channel(
                    self._warps.get(m, set()),
                    player_x=player_x,
                    player_y=player_y,
                )
            )
        if cfg.include_interactions:
            channels.append(
                self.build_local_channel(
                    self._interact_success.get(m, set()),
                    player_x=player_x,
                    player_y=player_y,
                )
            )
            channels.append(
                self.build_local_channel(
                    self._interact_fail.get(m, set()),
                    player_x=player_x,
                    player_y=player_y,
                )
            )

        if not channels:
            return np.zeros((1, self.local_size, self.local_size), dtype=np.float32)
        return np.stack(channels, axis=0)

    def build_goal_vector(
        self,
        *,
        map_id: int,
        player_x: int,
        player_y: int,
    ) -> np.ndarray:
        cfg = self.config
        ctx = self.context
        r = max(1, int(cfg.local_radius))
        target_map = int(ctx.target_map_id or 0)
        on_target = float(
            ctx.target_map_id is not None and int(map_id) == int(ctx.target_map_id)
        )
        goal_dx = 0.0
        goal_dy = 0.0
        has_coords = 0.0
        if cfg.include_goal_vector and ctx.target_x is not None and ctx.target_y is not None:
            has_coords = 1.0
            goal_dx = _clamp(int(ctx.target_x) - int(player_x), -r, r) / float(r)
            goal_dy = _clamp(int(ctx.target_y) - int(player_y), -r, r) / float(r)
        self._last_goal_vector = (float(goal_dx), float(goal_dy))
        player_x_norm = float(int(player_x) % 256) / _PLAYER_COORD_NORM
        player_y_norm = float(int(player_y) % 256) / _PLAYER_COORD_NORM
        return np.array(
            [
                target_map / 255.0,
                on_target,
                player_x_norm,
                player_y_norm,
                goal_dx,
                goal_dy,
                has_coords,
                _norm_id(ctx.target_event_id),
                _norm_id(ctx.target_object_id),
            ],
            dtype=np.float32,
        )

    def build_obs(
        self,
        *,
        map_id: int,
        player_x: int,
        player_y: int,
        tile_blocked: TileBlockedTracker | None,
        npc_coords: set[Coord] | None = None,
    ) -> dict[str, np.ndarray]:
        local = self.build_local_maps(
            map_id=map_id,
            player_x=player_x,
            player_y=player_y,
            tile_blocked=tile_blocked,
            npc_coords=npc_coords,
        )
        idx = 0
        visited_count = 0
        blocked_count = 0
        if self.config.include_visited:
            visited_count = int(local[idx].sum())
            idx += 1
        if self.config.include_blocked:
            blocked_count = int(local[idx].sum())
        self._last_local_counts = (blocked_count, visited_count)
        return {
            "goal_memory_local": local,
            "goal_memory_goal": self.build_goal_vector(
                map_id=map_id, player_x=player_x, player_y=player_y
            ),
        }

    def info_fields(self, *, map_id: int | None = None) -> dict[str, Any]:
        ctx = self.context
        goal_dx, goal_dy = self._last_goal_vector
        blocked_count, visited_count = self._last_local_counts
        cur_map = int(map_id) if map_id is not None else self._last_map_id
        on_target = (
            ctx.target_map_id is not None
            and cur_map is not None
            and int(ctx.target_map_id) == int(cur_map)
        )
        return {
            "goal_key": ctx.goal_key,
            "target_map_id": ctx.target_map_id,
            "on_target_map": on_target,
            "goal_dx": goal_dx,
            "goal_dy": goal_dy,
            "blocked_local_count": blocked_count,
            "visited_local_count": visited_count,
            "reward_goal_map": self._step_rewards.map_enter + self._step_rewards.map_explore,
            "reward_goal_event": self._step_rewards.event_done,
            "reward_goal_interaction": self._step_rewards.interaction,
            "reward_blocked_penalty": self._step_rewards.blocked_penalty,
        }

    def consume_step_reward(self) -> float:
        return self._step_rewards.total()


def goal_memory_observation_space(config: GoalMemoryConfig) -> dict[str, spaces.Space]:
    if not config.enabled:
        return {}
    size = 2 * int(config.local_radius) + 1
    num_channels = sum(
        [
            config.include_visited,
            config.include_blocked,
            config.include_warps,
            config.include_interactions * 2,
        ]
    )
    num_channels = max(1, num_channels)
    return {
        "goal_memory_local": spaces.Box(
            low=0.0, high=1.0, shape=(num_channels, size, size), dtype=np.float32
        ),
        "goal_memory_goal": spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        ),
    }


def goal_context_from_planner_dict(
    planner: Any,
    *,
    subgoal_index: int = 0,
) -> dict[str, Any]:
    """Build goal_context dict from PlannerOutput without NLP parsing."""
    from pokemon_hrl.mode.subgoal import current_subgoal
    from pokemon_hrl.planner.criteria import planner_goal_key

    ctx: dict[str, Any] = {
        "target_map_id": getattr(planner, "target_map_id", None),
        "goal_key": planner_goal_key(planner),
    }
    hint = getattr(planner, "hint", None) or {}
    for key in ("target_x", "target_y", "target_object_id", "target_event_id"):
        if key in hint:
            ctx[key] = hint[key]
    criteria = list(getattr(planner, "success_criteria", []) or [])
    active = current_subgoal(getattr(planner, "subgoal", []) or [], subgoal_index)
    if active is not None:
        criteria.extend(active.success_criteria)
    for token in criteria:
        raw = str(token).strip()
        if raw.startswith("flag:"):
            ctx.setdefault("target_event_id", raw.split(":", 1)[1])
            break
    return ctx
