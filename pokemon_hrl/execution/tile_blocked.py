"""Track blocked tile targets and escalate to full action-mask blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from pokemon_hrl.execution.action_space import tile_target_coords, TILE_ACTIONS
from pokemon_hrl.types import FailedTileMoveResult

TargetKey = tuple[int, int, int]


@dataclass
class _TargetBlockState:
    ttl: int
    confidence: int
    last_attempt_step: int


class TileBlockedTracker:
    """Remember failed move targets ``(map_id, target_x, target_y)`` with TTL and confidence."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_steps: int = 100,
        weaken_weight: float = 0.05,
        retry_window_steps: int = 50,
        confidence_threshold: int = 1,
    ):
        self.enabled = bool(enabled)
        self.ttl_steps = max(0, int(ttl_steps))
        self.weaken_weight = float(weaken_weight)
        self.retry_window_steps = max(0, int(retry_window_steps))
        self.confidence_threshold = max(1, int(confidence_threshold))
        self._targets: dict[TargetKey, _TargetBlockState] = {}

    def reset(self) -> None:
        self._targets.clear()

    def record_failed_move(
        self,
        map_id: int,
        x: int,
        y: int,
        direction: int,
        *,
        step: int,
        skip_coords: set[tuple[int, int]] | None = None,
    ) -> FailedTileMoveResult:
        """Record a failed tile move toward ``tile_target_coords(x, y, direction)``."""
        tx, ty = tile_target_coords(x, y, direction)
        if skip_coords and (int(tx), int(ty)) in skip_coords:
            return FailedTileMoveResult(
                target_map=int(map_id),
                target_x=tx,
                target_y=ty,
            )

        if not self.enabled or self.ttl_steps <= 0:
            return FailedTileMoveResult(
                target_map=int(map_id),
                target_x=tx,
                target_y=ty,
            )

        direction = int(direction)
        if direction not in {int(a) for a in TILE_ACTIONS}:
            return FailedTileMoveResult(
                target_map=int(map_id),
                target_x=tx,
                target_y=ty,
            )

        key = (int(map_id), tx, ty)
        step = int(step)
        existing = self._targets.get(key)
        retry_penalty = False

        if existing is None:
            self._targets[key] = _TargetBlockState(
                ttl=self.ttl_steps,
                confidence=1,
                last_attempt_step=step,
            )
        else:
            gap = step - existing.last_attempt_step
            if 0 < gap <= self.retry_window_steps:
                existing.confidence += 1
                retry_penalty = True
            existing.ttl = self.ttl_steps
            existing.last_attempt_step = step

        return FailedTileMoveResult(
            target_map=int(map_id),
            target_x=tx,
            target_y=ty,
            retry_penalty=retry_penalty,
        )

    def clear_target(self, map_id: int, target_x: int, target_y: int) -> None:
        self._targets.pop((int(map_id), int(target_x), int(target_y)), None)

    def tick(self) -> None:
        if not self._targets:
            return
        expired: list[TargetKey] = []
        for key, state in self._targets.items():
            if state.ttl <= 1:
                expired.append(key)
            else:
                state.ttl -= 1
        for key in expired:
            del self._targets[key]

    def blocked_coords_for_map(
        self,
        map_id: int,
        *,
        exclude_coords: set[tuple[int, int]] | None = None,
    ) -> set[tuple[int, int]]:
        """Coords with confidence at or above threshold and active TTL on ``map_id``."""
        if not self.enabled or not self._targets:
            return set()
        m = int(map_id)
        out: set[tuple[int, int]] = set()
        for (mid, tx, ty), state in self._targets.items():
            if mid != m or state.ttl <= 0:
                continue
            if state.confidence >= self.confidence_threshold:
                coord = (int(tx), int(ty))
                if exclude_coords and coord in exclude_coords:
                    continue
                out.add(coord)
        return out

    def is_blocked_target(self, map_id: int, target_x: int, target_y: int) -> bool:
        key = (int(map_id), int(target_x), int(target_y))
        state = self._targets.get(key)
        if state is None or state.ttl <= 0:
            return False
        return state.confidence >= self.confidence_threshold

    def tile_mask_weights(self, map_id: int, x: int, y: int) -> dict[int, float]:
        """Mask weights for tile actions from ``(x, y)`` based on known blocked targets."""
        if not self.enabled or not self._targets:
            return {}

        m, x, y = int(map_id), int(x), int(y)
        out: dict[int, float] = {}
        for direction in TILE_ACTIONS:
            tx, ty = tile_target_coords(x, y, int(direction))
            state = self._targets.get((m, tx, ty))
            if state is None or state.ttl <= 0:
                continue
            if state.confidence >= self.confidence_threshold:
                out[int(direction)] = 0.0
            else:
                out[int(direction)] = self.weaken_weight
        return out

    def target_weight(self, map_id: int, target_x: int, target_y: int) -> float:
        """Observation weight for a blocked target tile: 0, weaken_weight, or 1.0."""
        if not self.enabled:
            return 0.0
        state = self._targets.get((int(map_id), int(target_x), int(target_y)))
        if state is None or state.ttl <= 0:
            return 0.0
        if state.confidence >= self.confidence_threshold:
            return 1.0
        return float(self.weaken_weight)

    def build_local_blocked_map(
        self,
        map_id: int,
        player_x: int,
        player_y: int,
        *,
        radius: int,
        exclude_coords: set[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Player-centered local map of blocked targets (weakened or fully blocked)."""
        r = max(0, int(radius))
        size = 2 * r + 1
        out = np.zeros((size, size), dtype=np.float32)
        if not self.enabled or not self._targets:
            return out

        m = int(map_id)
        px, py = int(player_x), int(player_y)
        for (mid, tx, ty), state in self._targets.items():
            if mid != m or state.ttl <= 0:
                continue
            if exclude_coords and (int(tx), int(ty)) in exclude_coords:
                continue
            lx, ly = int(tx) - px, int(ty) - py
            if abs(lx) > r or abs(ly) > r:
                continue
            weight = self.target_weight(m, tx, ty)
            if weight <= 0.0:
                continue
            row, col = ly + r, lx + r
            out[row, col] = max(out[row, col], weight)
        return out


def blocked_tile_observation_space(*, radius: int) -> dict[str, spaces.Space]:
    size = 2 * max(0, int(radius)) + 1
    return {
        "blocked_tile_local": spaces.Box(
            low=0.0, high=1.0, shape=(size, size), dtype=np.float32
        ),
    }
