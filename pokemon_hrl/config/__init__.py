"""HRL configuration loading."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from pokemon_hrl.paths import BASE_CONFIG_PATH, DEFAULT_START_SAV, HRL_CONFIG_PATH, project_root


def _resolve_optional_path(root: Path, raw: str | None) -> str | None:
    if raw is None or str(raw).strip() in ("", "~", "null", "None"):
        return None
    candidate = Path(str(raw)).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    under_root = (root / candidate).resolve()
    if under_root.is_file():
        return str(under_root)
    return str(candidate.resolve())


def load_hrl_config(
    hrl_config_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
) -> DictConfig:
    hrl_path = Path(hrl_config_path or HRL_CONFIG_PATH).expanduser().resolve()
    base_path = Path(base_config_path or BASE_CONFIG_PATH).expanduser().resolve()
    hrl_cfg = OmegaConf.load(hrl_path)
    if base_path.is_file():
        base_cfg = OmegaConf.load(base_path)
        merged = OmegaConf.merge(base_cfg, hrl_cfg)
    else:
        merged = hrl_cfg

    root = project_root()
    merged.env.gb_path = str(root / "assets" / "red.gb")
    merged.env.state_dir = str(root / "assets" / "pyboy_states")
    merged.hrl.checkpoint.directory = str(root / "checkpoints")

    init_path = _resolve_optional_path(root, merged.env.get("init_state_path"))
    if init_path is None and DEFAULT_START_SAV.is_file():
        init_path = str(DEFAULT_START_SAV.resolve())
        merged.env.init_state = "red"
    if init_path is not None:
        merged.env.init_state_path = init_path

    video_dir = merged.env.get("video_dir")
    if video_dir:
        merged.env.video_dir = str((root / str(video_dir)).resolve())

    prompt_rel = OmegaConf.select(merged, "hrl.planner.prompt_path")
    if prompt_rel:
        merged.hrl.planner.prompt_path = str((root / str(prompt_rel)).resolve())

    autotune_storage = OmegaConf.select(merged, "hrl.autotune.storage")
    if autotune_storage and str(autotune_storage).startswith("sqlite:///"):
        rel = str(autotune_storage).replace("sqlite:///", "", 1)
        merged.hrl.autotune.storage = f"sqlite:///{(root / rel).resolve().as_posix()}"

    merged.train = OmegaConf.merge(
        OmegaConf.create({"data_dir": str(root / "runs")}),
        merged.get("train", OmegaConf.create({})),
        merged.hrl.get("training", OmegaConf.create({})),
    )
    return merged


def clone_hrl_config(config: DictConfig) -> DictConfig:
    """Deep-copy an HRL config while keeping OmegaConf node types."""
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))
