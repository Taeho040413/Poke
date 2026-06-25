from pathlib import Path

path = Path("pokemon_hrl/pokemonred_puffer/environment.py")
text = path.read_text(encoding="utf-8")

def remove_all_methods(src: str, method_name: str) -> tuple[str, int]:
    marker = f"\n    def {method_name}("
    removed = 0

    while True:
        start = src.find(marker)
        if start == -1:
            break

        # 다음 같은 class-indent method를 찾는다.
        next_def = src.find("\n    def ", start + len(marker))
        if next_def == -1:
            raise RuntimeError(f"Could not find end of method {method_name}")

        src = src[:start] + src[next_def:]
        removed += 1

    return src, removed

text, removed = remove_all_methods(text, "build_pokemon_exploration_rgb")
print(f"[patched] removed {removed} build_pokemon_exploration_rgb definition(s)")

# high_reward_tile_map 잔여 라인 제거
lines = []
removed_high_reward_lines = 0
for line in text.splitlines(True):
    if "high_reward_tile_map" in line:
        removed_high_reward_lines += 1
        continue
    lines.append(line)
text = "".join(lines)
print(f"[patched] removed {removed_high_reward_lines} high_reward_tile_map line(s)")

raw_grid_func = '''
    def build_pokemon_exploration_rgb(self):
        """Full global agent-memory grid for W&B.

        This is not the Kanto background overlay. It visualizes the same global
        memory source that visited_mask is cropped from.

        Colors:
          dark/black = unseen
          green      = visited memory from explore_map
          red        = blocked target tile from blocked_tile_map
        """
        h, w = GLOBAL_MAP_SHAPE
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        max_visit = max(float(getattr(self, "exploration_max", 1.0)), 1e-6)
        visited = np.clip(self.explore_map.astype(np.float32) / max_visit, 0.0, 1.0)

        rgb[..., 0] = 0.08 * visited
        rgb[..., 1] = 0.85 * visited
        rgb[..., 2] = 0.08 * visited

        blocked = getattr(self, "blocked_tile_map", None)
        if blocked is not None:
            rgb[np.asarray(blocked) > 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        return rgb

'''

anchor = "\n    def agent_stats(self, action):"
idx = text.find(anchor)
if idx == -1:
    raise RuntimeError("Could not find agent_stats anchor")

text = text[:idx] + "\n" + raw_grid_func + text[idx:]

path.write_text(text, encoding="utf-8")
print("[patched] inserted one raw-grid build_pokemon_exploration_rgb before agent_stats")