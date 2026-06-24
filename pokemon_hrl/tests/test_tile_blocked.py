import numpy as np
import pytest

from pokemon_hrl.execution.action_masks import compute_hrl_action_mask
from pokemon_hrl.execution.action_space import HrlAction, tile_target_coords
from pokemon_hrl.execution.tile_blocked import TileBlockedTracker


def test_tile_target_coords_right():
    assert tile_target_coords(5, 7, int(HrlAction.TILE_RIGHT)) == (6, 7)


def test_failed_right_records_target_not_source():
    tracker = TileBlockedTracker(
        enabled=True, ttl_steps=5, weaken_weight=0.1, confidence_threshold=1
    )
    tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_RIGHT), step=0)
    tracker.tick()

    weights = tracker.tile_mask_weights(1, 5, 7)
    assert weights[int(HrlAction.TILE_RIGHT)] == 0.0
    assert int(HrlAction.TILE_LEFT) not in weights

    weights_from_east = tracker.tile_mask_weights(1, 7, 7)
    assert weights_from_east[int(HrlAction.TILE_LEFT)] == 0.0


def test_retry_within_window_applies_penalty_and_escalates_to_block():
    tracker = TileBlockedTracker(
        enabled=True,
        ttl_steps=10,
        weaken_weight=0.1,
        retry_window_steps=50,
        confidence_threshold=2,
    )
    first = tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_RIGHT), step=0)
    assert first.retry_penalty is False
    assert first.target_x == 6 and first.target_y == 7

    tracker.tick()
    weights = tracker.tile_mask_weights(1, 5, 7)
    assert weights[int(HrlAction.TILE_RIGHT)] == 0.1

    retry = tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_RIGHT), step=10)
    assert retry.retry_penalty is True
    tracker.tick()
    weights = tracker.tile_mask_weights(1, 5, 7)
    assert weights[int(HrlAction.TILE_RIGHT)] == 0.0


def test_retry_outside_window_does_not_increment_confidence():
    tracker = TileBlockedTracker(
        enabled=True,
        ttl_steps=10,
        weaken_weight=0.1,
        retry_window_steps=5,
        confidence_threshold=2,
    )
    tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_RIGHT), step=0)
    late = tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_RIGHT), step=10)
    assert late.retry_penalty is False
    tracker.tick()
    weights = tracker.tile_mask_weights(1, 5, 7)
    assert weights[int(HrlAction.TILE_RIGHT)] == 0.1


def test_ttl_expires_blocked_target():
    tracker = TileBlockedTracker(
        enabled=True, ttl_steps=2, weaken_weight=0.1, confidence_threshold=1
    )
    tracker.record_failed_move(1, 5, 7, int(HrlAction.TILE_UP), step=0)
    tracker.tick()
    assert tracker.tile_mask_weights(1, 5, 7)[int(HrlAction.TILE_UP)] == 0.0
    tracker.tick()
    assert tracker.tile_mask_weights(1, 5, 7) == {}


def test_compute_hrl_action_mask_applies_weakened_and_blocked_tile_weights():
    weakened = compute_hrl_action_mask(
        0,
        weakened_tile_weights={int(HrlAction.TILE_LEFT): 0.1},
    )
    assert float(weakened[int(HrlAction.TILE_LEFT)]) == pytest.approx(0.1)

    blocked = compute_hrl_action_mask(
        0,
        weakened_tile_weights={int(HrlAction.TILE_RIGHT): 0.0},
    )
    assert blocked[int(HrlAction.TILE_RIGHT)] == 0.0
    assert blocked[int(HrlAction.TILE_LEFT)] == 1.0


def test_battle_mask_overrides_weakened_tiles():
    mask = compute_hrl_action_mask(
        1,
        weakened_tile_weights={int(HrlAction.TILE_UP): 0.1},
    )
    assert mask[int(HrlAction.TILE_UP)] == 0.0
    assert mask[int(HrlAction.LOW_A)] == 1.0


def test_clear_target_removes_mask():
    tracker = TileBlockedTracker(enabled=True, ttl_steps=5, weaken_weight=0.2)
    tracker.record_failed_move(3, 10, 12, int(HrlAction.TILE_RIGHT), step=0)
    tracker.clear_target(3, 11, 12)
    assert tracker.tile_mask_weights(3, 10, 12) == {}


def test_blocked_coords_for_map():
    tracker = TileBlockedTracker(
        enabled=True, ttl_steps=5, confidence_threshold=1
    )
    tracker.record_failed_move(2, 4, 6, int(HrlAction.TILE_UP), step=0)
    assert tracker.blocked_coords_for_map(2) == {(4, 5)}
    assert tracker.is_blocked_target(2, 4, 5)
    tracker.tick()
    tracker.tick()
    tracker.tick()
    tracker.tick()
    tracker.tick()
    assert tracker.blocked_coords_for_map(2) == set()


def test_build_local_blocked_map_marks_fully_blocked_and_weakened():
    tracker = TileBlockedTracker(
        enabled=True,
        ttl_steps=10,
        weaken_weight=0.2,
        confidence_threshold=2,
    )
    tracker.record_failed_move(1, 3, 4, int(HrlAction.TILE_RIGHT), step=0)
    tracker.record_failed_move(1, 3, 4, int(HrlAction.TILE_UP), step=0)
    tracker.record_failed_move(1, 3, 4, int(HrlAction.TILE_UP), step=1)

    local = tracker.build_local_blocked_map(1, 3, 4, radius=2)
    center = 2
    assert local.shape == (5, 5)
    assert local[center, center + 1] == pytest.approx(0.2)
    assert local[center - 1, center] == pytest.approx(1.0)


def test_blocked_tile_observation_space_shape():
    from pokemon_hrl.execution.tile_blocked import blocked_tile_observation_space

    spaces = blocked_tile_observation_space(radius=5)
    assert spaces["blocked_tile_local"].shape == (11, 11)
