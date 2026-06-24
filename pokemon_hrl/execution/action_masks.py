"""Context-dependent HRL action masks."""

from __future__ import annotations

import numpy as np

from pokemon_hrl.execution.action_space import ACTION_DIM, HrlAction, TILE_ACTIONS, is_tile_action

_MASK_ALL = np.ones(ACTION_DIM, dtype=np.float32)
_MASK_BATTLE = np.zeros(ACTION_DIM, dtype=np.float32)
_TILE_ACTION_IDS = tuple(int(a) for a in TILE_ACTIONS)
_FULL_MASK_WEIGHT = 1.0 - 1e-6
for _action in (
    HrlAction.LOW_DOWN,
    HrlAction.LOW_LEFT,
    HrlAction.LOW_RIGHT,
    HrlAction.LOW_UP,
    HrlAction.LOW_A,
    HrlAction.LOW_B,
    HrlAction.LOW_START,
    HrlAction.LOW_SELECT,
):
    _MASK_BATTLE[int(_action)] = 1


def compute_hrl_action_mask(
    is_in_battle: int,
    *,
    enabled: bool = True,
    tile_move_enabled: bool = True,
    weakened_tile_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Return a length-``ACTION_DIM`` float mask in ``[0, 1]``.

    ``1`` = fully allowed, values in ``(0, 1)`` weaken logits, ``0`` = blocked.
    In battle, tile movement (0-3) is blocked; low-level buttons remain available.
    When ``tile_move_enabled`` is false, tile actions are always blocked.
    """
    if not enabled:
        mask = _MASK_ALL.copy()
    elif int(is_in_battle) != 0:
        mask = _MASK_BATTLE.copy()
    else:
        mask = _MASK_ALL.copy()

    if not tile_move_enabled:
        for tile_action in (
            HrlAction.TILE_UP,
            HrlAction.TILE_DOWN,
            HrlAction.TILE_LEFT,
            HrlAction.TILE_RIGHT,
        ):
            mask[int(tile_action)] = 0.0

    if weakened_tile_weights:
        for action_id, weight in weakened_tile_weights.items():
            idx = int(action_id)
            if not (0 <= idx < ACTION_DIM):
                continue
            weight_f = float(weight)
            if weight_f <= 0.0:
                mask[idx] = 0.0
            elif mask[idx] > 0.0:
                mask[idx] = min(mask[idx], weight_f)
    return mask


def _mask_allows_execution(action: int, mask: np.ndarray) -> bool:
    idx = int(action)
    if not (0 <= idx < mask.shape[0]):
        return False
    weight = float(mask[idx])
    if weight <= 0.0:
        return False
    # Weakened tile logits guide learning; the emulator must not re-try blocked dirs.
    if is_tile_action(idx) and weight < _FULL_MASK_WEIGHT:
        return False
    return True


def _pick_fallback_action(mask: np.ndarray) -> int:
    for tile_id in _TILE_ACTION_IDS:
        if float(mask[tile_id]) >= _FULL_MASK_WEIGHT:
            return tile_id
    low_a = int(HrlAction.LOW_A)
    if 0 <= low_a < mask.shape[0] and float(mask[low_a]) > 0.0:
        return low_a
    valid = np.flatnonzero(mask > 0.0)
    if valid.size == 0:
        return int(HrlAction.LOW_A)
    return int(valid[0])


def apply_action_mask(action: int, mask: np.ndarray) -> int:
    """Keep ``action`` when allowed; otherwise pick another valid tile or button."""
    action = int(action)
    if _mask_allows_execution(action, mask):
        return action
    return _pick_fallback_action(mask)
