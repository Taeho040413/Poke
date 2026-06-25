from pathlib import Path
import re

def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        print(f"[patched] {label}")
        return text.replace(old, new, 1)
    if new in text:
        print(f"[skip] {label} already patched")
        return text
    print(f"[skip] {label} anchor not found")
    return text

# ============================================================
# 1. environment.py
#    - high_reward_tile_map 완전 제거
#    - fake blue marker 제거
#    - build_pokemon_exploration_rgb를 raw memory grid로 고정
# ============================================================
env_path = "pokemon_hrl/pokemonred_puffer/environment.py"
text = read(env_path)

# high_reward_tile_map init/reset 제거
text = re.sub(
    r"^\s*self\.high_reward_tile_map\s*=\s*np\.zeros\(GLOBAL_MAP_SHAPE,\s*dtype=np\.float32\)\n",
    "",
    text,
    flags=re.MULTILINE,
)
text = re.sub(
    r"^\s*self\.high_reward_tile_map\s*\*=\s*0\n",
    "",
    text,
    flags=re.MULTILINE,
)

# new_reward >= 1.5 marker block 제거
text = re.sub(
    r"""        new_reward = self\.update_reward\(\)
        if new_reward >= 1\.5:
            _x, _y, _map_n = self\.get_game_coords\(\)
            _gy, _gx = local_to_global\(_y, _x, _map_n\)
            self\.high_reward_tile_map\[_gy, _gx\] = 1\.0
        self\.update_map_progress\(\)
""",
    """        new_reward = self.update_reward()
        self.update_map_progress()
""",
    text,
)

# raw memory grid 함수
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

        # visited grid
        rgb[..., 0] = 0.08 * visited
        rgb[..., 1] = 0.85 * visited
        rgb[..., 2] = 0.08 * visited

        # blocked tile has priority over visited
        blocked = getattr(self, "blocked_tile_map", None)
        if blocked is not None:
            rgb[np.asarray(blocked) > 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        return rgb
'''

start = text.find("\n    def build_pokemon_exploration_rgb(self):")
agent_stats = text.find("\n    def agent_stats(self, action):")
if start != -1 and agent_stats != -1 and start < agent_stats:
    text = text[:start] + "\n" + raw_grid_func + text[agent_stats:]
    print("[patched] replace build_pokemon_exploration_rgb")
elif agent_stats != -1:
    text = text[:agent_stats] + "\n" + raw_grid_func + text[agent_stats:]
    print("[patched] insert build_pokemon_exploration_rgb")
else:
    raise RuntimeError("Could not find agent_stats anchor in environment.py")

write(env_path, text)

# ============================================================
# 2. eval.py
#    - Kanto 배경 없는 raw grid renderer 추가
# ============================================================
eval_path = "pokemon_hrl/pokemonred_puffer/eval.py"
text = read(eval_path)

memory_grid_func = '''
def make_agent_memory_grid(counts: np.ndarray, scale: int = 4) -> np.ndarray:
    """Render full global agent-memory grid without Kanto background.

    Input:
      - (N, H, W, 3): RGB memory grids from env infos
      - (N, H, W): scalar visit maps

    Output:
      - upscaled uint8 RGB image for wandb.Image
    """
    counts = np.asarray(counts)
    if counts.ndim == 4 and counts.shape[-1] == 3:
        rgb = np.max(counts.astype(np.float32), axis=0)
    elif counts.ndim == 3:
        visit = np.max(counts.astype(np.float32), axis=0)
        if np.max(visit) > 0:
            visit = visit / np.max(visit)
        rgb = np.zeros((*visit.shape, 3), dtype=np.float32)
        rgb[..., 1] = visit
    else:
        raise ValueError(f"Unsupported grid shape for make_agent_memory_grid: {counts.shape}")

    rgb = np.clip(rgb, 0.0, 1.0)
    image = (255.0 * rgb).astype(np.uint8)

    scale = max(1, int(scale))
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return image

'''

if "def make_agent_memory_grid(" not in text:
    text = text.replace(
        "\ndef make_pokemon_red_overlay(counts: np.ndarray):",
        "\n" + memory_grid_func + "\ndef make_pokemon_red_overlay(counts: np.ndarray):",
        1,
    )
    print("[patched] add make_agent_memory_grid")
else:
    print("[skip] make_agent_memory_grid already exists")

write(eval_path, text)

# ============================================================
# 3. cleanrl_puffer.py
#    - W&B에서 Kanto overlay 대신 raw memory grid 로깅
# ============================================================
cleanrl_path = "pokemon_hrl/pokemonred_puffer/cleanrl_puffer.py"
text = read(cleanrl_path)

text = text.replace(
    "from pokemonred_puffer.eval import make_pokemon_red_overlay\n",
    "from pokemonred_puffer.eval import make_agent_memory_grid\n",
)
text = text.replace(
    "from pokemonred_puffer.eval import make_agent_memory_grid\n",
    "from pokemonred_puffer.eval import make_agent_memory_grid\n",
)

text = replace_once(
    text,
    """                if "pokemon_exploration_map" in k and self.config.save_overlay is True:
                    if self.epoch % self.config.overlay_interval == 0:
                        overlay = make_pokemon_red_overlay(np.stack(self.infos[k], axis=0))
                        if self.wandb_client is not None:
                            self.stats["Media/aggregate_exploration_map"] = wandb.Image(overlay)
""",
    """                if "pokemon_exploration_map" in k and self.config.save_overlay is True:
                    if self.epoch % self.config.overlay_interval == 0:
                        grid = make_agent_memory_grid(
                            np.stack(self.infos[k], axis=0),
                            scale=getattr(self.config, "grid_map_scale", 4),
                        )
                        if self.wandb_client is not None:
                            self.stats["Media/agent_memory_grid"] = wandb.Image(grid)
""",
    "log raw agent memory grid",
)

write(cleanrl_path, text)

# ============================================================
# 4. engine.py
#    - grid_map_scale 기본값 추가
# ============================================================
engine_path = "pokemon_hrl/training/engine.py"
text = read(engine_path)

if '"grid_map_scale":' not in text:
    text = text.replace(
        '    "overlay_interval": 10,\n',
        '    "overlay_interval": 10,\n    "grid_map_scale": 4,\n',
        1,
    )
    print("[patched] add grid_map_scale default")
else:
    print("[skip] grid_map_scale already exists")

write(engine_path, text)

# ============================================================
# 5. checkpoint.py
#    - auto-resume이 model_interrupt.pt도 찾도록 수정
# ============================================================
ckpt_path = "pokemon_hrl/training/checkpoint.py"
text = read(ckpt_path)

# helper 추가
helper = '''
def _iter_resume_model_candidates(dir_path: Path) -> list[Path]:
    """Return resume candidates including interrupt/latest/numeric checkpoints."""
    candidates: list[Path] = []
    latest = dir_path / "model_latest.pt"
    interrupt = dir_path / "model_interrupt.pt"
    if latest.is_file():
        candidates.append(latest)
    candidates.extend(_iter_numeric_model_pts(dir_path))
    if interrupt.is_file():
        candidates.append(interrupt)

    # dedupe while preserving paths
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.expanduser().resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


'''

if "def _iter_resume_model_candidates(" not in text:
    insert_at = text.find("\ndef _pick_latest_checkpoint_by_mtime")
    if insert_at == -1:
        raise RuntimeError("Could not find _pick_latest_checkpoint_by_mtime anchor")
    text = text[:insert_at] + "\n" + helper + text[insert_at:]
    print("[patched] add _iter_resume_model_candidates")
else:
    print("[skip] _iter_resume_model_candidates already exists")

# find_latest_saved_model 내부 run_dir 로직 교체
text = re.sub(
    r"""        run_dir = root / exp_id
        if run_dir\.is_dir\(\):
            latest = run_dir / "model_latest\.pt"
            if latest\.is_file\(\):
                return latest
            picked = _pick_latest_checkpoint_by_mtime\(_iter_numeric_model_pts\(run_dir\)\)
            if picked is not None:
                return picked
""",
    """        run_dir = root / exp_id
        if run_dir.is_dir():
            picked = _pick_latest_checkpoint_by_mtime(_iter_resume_model_candidates(run_dir))
            if picked is not None:
                return picked
""",
    text,
)

# global fallback 로직 교체
text = re.sub(
    r"""        latest_candidates = list\(root\.rglob\("model_latest\.pt"\)\)
        if latest_candidates:
            return _pick_latest_checkpoint_by_mtime\(latest_candidates\)
        all_pts = \[
            p
            for p in root\.rglob\("model_\*\.pt"\)
            if _model_checkpoint_sort_key\(p\) >= 0
        \]
        if all_pts:
            return _pick_latest_checkpoint_by_mtime\(all_pts\)
""",
    """        all_pts: list[Path] = []
        for run_dir in root.rglob("*"):
            if run_dir.is_dir():
                all_pts.extend(_iter_resume_model_candidates(run_dir))
        if all_pts:
            return _pick_latest_checkpoint_by_mtime(all_pts)
""",
    text,
)

# resolve_resume_checkpoint(path dir)에서도 interrupt/latest 후보 허용
text = re.sub(
    r"""    models = _iter_numeric_model_pts\(path\)
    if not models:
        return None, None
    model = _pick_latest_checkpoint_by_mtime\(models\)
""",
    """    models = _iter_resume_model_candidates(path)
    if not models:
        return None, None
    model = _pick_latest_checkpoint_by_mtime(models)
""",
    text,
)

write(ckpt_path, text)

# ============================================================
# 6. config: 정책만 resume하려면 optimizer state false 권장
# ============================================================
cfg_path = Path("pokemon_hrl/config/hrl_config.yaml")
if cfg_path.exists():
    text = cfg_path.read_text(encoding="utf-8")
    text = text.replace("    load_optimizer_state: true\n", "    load_optimizer_state: false\n")
    if "    save_overlay: true\n" not in text and "    resume_checkpoint: auto\n" in text:
        text = text.replace(
            "    resume_checkpoint: auto\n",
            "    resume_checkpoint: auto\n    save_overlay: true\n    overlay_interval: 10\n    grid_map_scale: 4\n",
            1,
        )
    cfg_path.write_text(text, encoding="utf-8")
    print("[patched] hrl_config.yaml policy-only resume + grid logging")
else:
    print("[skip] hrl_config.yaml not found")

print("Done.")