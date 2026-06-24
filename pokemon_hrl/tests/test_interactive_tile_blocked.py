"""End-to-end wiring: failed tile move -> target-based action_mask."""

from __future__ import annotations

import numpy as np
import pytest

from pokemon_hrl.execution.action_masks import compute_hrl_action_mask
from pokemon_hrl.execution.action_space import HrlAction, tile_target_coords
from pokemon_hrl.execution.tile_blocked import TileBlockedTracker
from pokemon_hrl.types import ExecutionResult


def _simulate_interactive_tile_step(
    *,
    map_id: int,
    x: int,
    y: int,
    action: int,
    exec_result: ExecutionResult,
    step: int = 0,
    ttl_steps: int = 10,
    confidence_threshold: int = 2,
) -> np.ndarray:
    """Mirror HrlInteractiveRewardEnv.run_action_on_emulator tile-block logic."""
    tracker = TileBlockedTracker(
        enabled=True,
        ttl_steps=ttl_steps,
        weaken_weight=0.1,
        retry_window_steps=50,
        confidence_threshold=confidence_threshold,
    )
    if int(action) in {
        int(HrlAction.TILE_UP),
        int(HrlAction.TILE_DOWN),
        int(HrlAction.TILE_LEFT),
        int(HrlAction.TILE_RIGHT),
    }:
        if exec_result.tile_direction_blocked:
            tracker.record_failed_move(map_id, x, y, action, step=step)
        elif exec_result.moved_tile:
            tx, ty = tile_target_coords(x, y, action)
            tracker.clear_target(map_id, tx, ty)
    tracker.tick()
    tile_weights = tracker.tile_mask_weights(map_id, x, y)
    return compute_hrl_action_mask(0, weakened_tile_weights=tile_weights)


def test_failed_tile_move_blocks_mask_toward_target():
    mask = _simulate_interactive_tile_step(
        map_id=5,
        x=10,
        y=12,
        action=int(HrlAction.TILE_UP),
        exec_result=ExecutionResult(steps_used=1, tile_direction_blocked=True),
        confidence_threshold=1,
    )
    assert float(mask[int(HrlAction.TILE_UP)]) == pytest.approx(0.0)
    assert float(mask[int(HrlAction.TILE_DOWN)]) == pytest.approx(1.0)


def test_successful_tile_move_keeps_full_mask():
    mask = _simulate_interactive_tile_step(
        map_id=5,
        x=10,
        y=12,
        action=int(HrlAction.TILE_LEFT),
        exec_result=ExecutionResult(steps_used=1, moved_tile=True),
    )
    assert float(mask[int(HrlAction.TILE_LEFT)]) == pytest.approx(1.0)


def test_policy_log_mask_reduces_weakened_logit():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 12)
    mask = torch.ones(1, 12)
    mask[0, int(HrlAction.TILE_UP)] = 0.1
    adjusted = logits + torch.log(mask.clamp(min=1e-8))
    assert float(adjusted[0, int(HrlAction.TILE_UP)]) == pytest.approx(np.log(0.1), rel=1e-5)
    assert float(adjusted[0, int(HrlAction.TILE_DOWN)]) == pytest.approx(0.0, abs=1e-5)


def test_confidence_threshold_fully_blocks_after_repeat():
    tracker = TileBlockedTracker(
        enabled=True,
        ttl_steps=10,
        weaken_weight=0.1,
        retry_window_steps=50,
        confidence_threshold=2,
    )
    tracker.record_failed_move(1, 4, 6, int(HrlAction.TILE_RIGHT), step=0)
    tracker.tick()
    weights = tracker.tile_mask_weights(1, 4, 6)
    assert weights[int(HrlAction.TILE_RIGHT)] == 0.1
    tracker.record_failed_move(1, 4, 6, int(HrlAction.TILE_RIGHT), step=1)
    tracker.tick()
    weights = tracker.tile_mask_weights(1, 4, 6)
    assert weights[int(HrlAction.TILE_RIGHT)] == 0.0
