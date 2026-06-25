from pathlib import Path

# 1) curriculum success_criteria fix
path = Path("pokemon_hrl/training/curriculum.yaml")
text = path.read_text(encoding="utf-8")
old = """    success_criteria:
      - stat_on_target_map:first_npc_talk_count
    failure_criteria:
      - no_progress
"""
new = """    success_criteria:
      - stat_on_target_map:first_object_interaction_count
    failure_criteria:
      - no_progress
"""
if old not in text:
    raise SystemExit("pewter success_criteria block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 2) target_map_entry one-shot
path = Path("pokemon_hrl/pokemonred_puffer/rewards/baseline.py")
text = path.read_text(encoding="utf-8")

old = """        self._seen_map_ids: set[int] = set()
"""
new = """        self._seen_map_ids: set[int] = set()
        self._seen_target_map_entries: set[int] = set()
"""
if old not in text:
    raise SystemExit("_seen_map_ids block not found")
text = text.replace(old, new, 1)

old = """        target = self.hrl_target_map_id
        if target is not None and map_after == int(target):
            self.target_map_entry_count += 1
"""
new = """        target = self.hrl_target_map_id
        if (
            target is not None
            and map_after == int(target)
            and int(target) not in self._seen_target_map_entries
        ):
            self._seen_target_map_entries.add(int(target))
            self.target_map_entry_count += 1
"""
if old not in text:
    raise SystemExit("target_map_entry block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")


# 3) sync goal context immediately after subgoal advance
path = Path("pokemon_hrl/env/progress_wrapper.py")
text = path.read_text(encoding="utf-8")
old = """            reward += self.subgoal_success_reward
            self._subgoal_index += 1
"""
new = """            reward += self.subgoal_success_reward
            self._subgoal_index += 1
            self._sync_goal_context()
"""
if old not in text:
    raise SystemExit("subgoal advance block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 4,5) mask low-level movement outside battle + remove UP-biased fallback
path = Path("pokemon_hrl/execution/action_masks.py")
text = path.read_text(encoding="utf-8")

old = """_TILE_ACTION_IDS = tuple(int(a) for a in TILE_ACTIONS)
_FULL_MASK_WEIGHT = 1.0 - 1e-6
"""
new = """_TILE_ACTION_IDS = tuple(sorted(int(a) for a in TILE_ACTIONS))
_LOW_OVERWORLD_BLOCKED_IDS = tuple(
    int(a)
    for a in (
        HrlAction.LOW_DOWN,
        HrlAction.LOW_LEFT,
        HrlAction.LOW_RIGHT,
        HrlAction.LOW_UP,
        HrlAction.LOW_START,
        HrlAction.LOW_SELECT,
    )
)
_FULL_MASK_WEIGHT = 1.0 - 1e-6
"""
if old not in text:
    raise SystemExit("mask constants block not found")
text = text.replace(old, new, 1)

old = """    else:
        mask = _MASK_ALL.copy()
"""
new = """    else:
        mask = _MASK_ALL.copy()
        # Outside battle, movement must go through TILE_* macros so collision
        # tracking, blocked-tile memory, and tile action masks stay consistent.
        # Raw arrow keys can bypass TileBlockedTracker and cause wall/ledge headbutting.
        for action_id in _LOW_OVERWORLD_BLOCKED_IDS:
            mask[action_id] = 0.0
"""
if old not in text:
    raise SystemExit("outside battle mask block not found")
text = text.replace(old, new, 1)

old = """def _pick_fallback_action(mask: np.ndarray) -> int:
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
"""
new = """def _pick_fallback_action(mask: np.ndarray) -> int:
    valid_tiles = [
        tile_id
        for tile_id in _TILE_ACTION_IDS
        if float(mask[tile_id]) >= _FULL_MASK_WEIGHT
    ]
    if valid_tiles:
        return int(np.random.choice(valid_tiles))
    low_a = int(HrlAction.LOW_A)
    if 0 <= low_a < mask.shape[0] and float(mask[low_a]) > 0.0:
        return low_a
    valid = np.flatnonzero(mask > 0.0)
    if valid.size == 0:
        return int(HrlAction.LOW_A)
    return int(np.random.choice(valid))
"""
if old not in text:
    raise SystemExit("fallback block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")


# 6) stronger blocked memory config
path = Path("pokemon_hrl/config/hrl_config.yaml")
text = path.read_text(encoding="utf-8")
text = text.replace("      ttl_steps: 100\n", "      ttl_steps: 1000\n", 1)
text = text.replace("      retry_window_steps: 50\n", "      retry_window_steps: 200\n", 1)
text = text.replace("      confidence_threshold: 1\n", "      confidence_threshold: 1\n", 1)
path.write_text(text, encoding="utf-8")

print("patched curriculum/reward/progress/action-mask/blocked config")
