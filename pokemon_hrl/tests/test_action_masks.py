import numpy as np

from pokemon_hrl.execution.action_masks import apply_action_mask, compute_hrl_action_mask
from pokemon_hrl.execution.action_space import ACTION_DIM, HrlAction


def test_field_mask_allows_all_actions():
    mask = compute_hrl_action_mask(0)
    assert mask.shape == (ACTION_DIM,)
    assert mask.sum() == ACTION_DIM


def test_battle_mask_blocks_tile_actions_only():
    mask = compute_hrl_action_mask(1)
    assert mask[int(HrlAction.TILE_UP)] == 0
    assert mask[int(HrlAction.TILE_DOWN)] == 0
    assert mask[int(HrlAction.TILE_LEFT)] == 0
    assert mask[int(HrlAction.TILE_RIGHT)] == 0
    assert mask[int(HrlAction.LOW_A)] == 1
    assert mask[int(HrlAction.LOW_B)] == 1


def test_battle_mask_disabled_allows_tiles():
    mask = compute_hrl_action_mask(2, enabled=False)
    assert mask.sum() == ACTION_DIM


def test_tile_move_disabled_blocks_tile_actions():
    mask = compute_hrl_action_mask(0, tile_move_enabled=False)
    assert mask[int(HrlAction.TILE_UP)] == 0
    assert mask[int(HrlAction.LOW_A)] == 1


def test_apply_action_mask_remaps_blocked_tile_in_battle():
    mask = compute_hrl_action_mask(1)
    remapped = apply_action_mask(int(HrlAction.TILE_UP), mask)
    assert float(mask[remapped]) > 0.0
    assert remapped == int(HrlAction.LOW_A)


def test_apply_action_mask_remaps_fully_blocked_tile():
    mask = compute_hrl_action_mask(
        0,
        weakened_tile_weights={int(HrlAction.TILE_UP): 0.0},
    )
    remapped = apply_action_mask(int(HrlAction.TILE_UP), mask)
    assert float(mask[remapped]) > 0.0
    assert remapped != int(HrlAction.TILE_UP)


def test_apply_action_mask_blocks_weakened_tile_execution():
    mask = compute_hrl_action_mask(
        0,
        weakened_tile_weights={int(HrlAction.TILE_UP): 0.1},
    )
    remapped = apply_action_mask(int(HrlAction.TILE_UP), mask)
    assert remapped != int(HrlAction.TILE_UP)
    assert float(mask[remapped]) >= 1.0 - 1e-6


def test_apply_action_mask_prefers_other_tile_over_button():
    mask = compute_hrl_action_mask(
        0,
        weakened_tile_weights={int(HrlAction.TILE_UP): 0.0},
    )
    remapped = apply_action_mask(int(HrlAction.TILE_UP), mask)
    assert remapped in {
        int(HrlAction.TILE_DOWN),
        int(HrlAction.TILE_LEFT),
        int(HrlAction.TILE_RIGHT),
    }
