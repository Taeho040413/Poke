"""On-map NPC/object sprite tile positions for policy observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from pokemonred_puffer.environment import RedGymEnv

Coord = tuple[int, int]

_MAX_SPRITES = 16


class _NpcCoordReader(Protocol):
    def read_m(self, addr: str) -> int: ...


def read_npc_coords(env: _NpcCoordReader) -> set[Coord]:
    """Tile coords of map sprites (ids 1..wNumSprites), excluding the player."""
    coords: set[Coord] = set()
    try:
        sprite_count = max(0, min(int(env.read_m("wNumSprites")), _MAX_SPRITES))
    except Exception:
        return coords

    for sprite_id in range(1, sprite_count + 1):
        prefix = f"wSprite{sprite_id:02}StateData2"
        try:
            map_x = int(env.read_m(f"{prefix}MapX"))
            map_y = int(env.read_m(f"{prefix}MapY"))
        except Exception:
            continue
        coords.add((map_x, map_y))
    return coords


def build_local_coord_map(
    coords: set[Coord],
    *,
    player_x: int,
    player_y: int,
    radius: int,
) -> np.ndarray:
    """Player-centered binary map for the given tile coords."""
    r = max(0, int(radius))
    size = 2 * r + 1
    out = np.zeros((size, size), dtype=np.float32)
    px, py = int(player_x), int(player_y)
    for tx, ty in coords:
        lx, ly = int(tx) - px, int(ty) - py
        if abs(lx) > r or abs(ly) > r:
            continue
        out[ly + r, lx + r] = 1.0
    return out


def build_local_npc_map(
    env: RedGymEnv | _NpcCoordReader,
    *,
    player_x: int,
    player_y: int,
    radius: int,
) -> np.ndarray:
    return build_local_coord_map(
        read_npc_coords(env),
        player_x=player_x,
        player_y=player_y,
        radius=radius,
    )


def npc_local_observation_space(*, radius: int) -> dict[str, spaces.Space]:
    size = 2 * max(0, int(radius)) + 1
    return {
        "npc_local": spaces.Box(
            low=0.0, high=1.0, shape=(size, size), dtype=np.float32
        ),
    }
