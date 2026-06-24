import pytest

from pokemon_hrl.execution.action_space import HrlAction
from pokemon_hrl.execution.npc_tiles import (
    build_local_coord_map,
    build_local_npc_map,
    read_npc_coords,
)
from pokemon_hrl.execution.tile_blocked import TileBlockedTracker


class _MockNpcEnv:
    def __init__(self, sprite_count: int, sprites: dict[int, tuple[int, int]]):
        self._sprite_count = sprite_count
        self._sprites = sprites

    def read_m(self, addr: str) -> int:
        if addr == "wNumSprites":
            return self._sprite_count
        if addr.startswith("wSprite") and addr.endswith("StateData2MapX"):
            sprite_id = int(addr[7:9])
            return self._sprites[sprite_id][0]
        if addr.startswith("wSprite") and addr.endswith("StateData2MapY"):
            sprite_id = int(addr[7:9])
            return self._sprites[sprite_id][1]
        raise KeyError(addr)


def test_read_npc_coords_from_sprite_wram():
    env = _MockNpcEnv(
        sprite_count=2,
        sprites={1: (6, 7), 2: (8, 9)},
    )
    assert read_npc_coords(env) == {(6, 7), (8, 9)}


def test_build_local_npc_map_centers_on_player():
    env = _MockNpcEnv(sprite_count=1, sprites={1: (6, 7)})
    local = build_local_npc_map(env, player_x=5, player_y=7, radius=2)
    assert local.shape == (5, 5)
    assert local[2, 3] == pytest.approx(1.0)


def test_record_failed_move_skips_npc_tile():
    tracker = TileBlockedTracker(
        enabled=True, ttl_steps=10, confidence_threshold=1
    )
    tracker.record_failed_move(
        1,
        5,
        7,
        int(HrlAction.TILE_RIGHT),
        step=0,
        skip_coords={(6, 7)},
    )
    assert tracker.blocked_coords_for_map(1) == set()
    assert tracker.tile_mask_weights(1, 5, 7) == {}


def test_build_local_blocked_map_excludes_npc_coords():
    tracker = TileBlockedTracker(
        enabled=True, ttl_steps=10, confidence_threshold=1
    )
    tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_RIGHT), step=0)
    local = tracker.build_local_blocked_map(
        1,
        5,
        7,
        radius=2,
        exclude_coords={(6, 7)},
    )
    assert local.sum() == pytest.approx(0.0)


def test_npc_and_blocked_maps_are_separate():
    tracker = TileBlockedTracker(
        enabled=True, ttl_steps=10, confidence_threshold=1
    )
    tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_UP), step=0)
    npc = build_local_coord_map({(6, 7)}, player_x=5, player_y=7, radius=2)
    blocked = tracker.build_local_blocked_map(
        1, 5, 7, radius=2, exclude_coords={(6, 7)}
    )
    assert npc[2, 3] == pytest.approx(1.0)
    assert blocked[1, 2] == pytest.approx(1.0)
    assert blocked[2, 3] == pytest.approx(0.0)
