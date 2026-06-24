"""Canonical paths for the self-contained pokemon_hrl package."""

from __future__ import annotations

from pathlib import Path

# pokemon_hrl/ package root
HRL_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = HRL_ROOT / "assets"
ROM_PATH = ASSETS_DIR / "red.gb"
RAM_PATH = ASSETS_DIR / "red.gb.ram"
STATE_DIR = ASSETS_DIR / "pyboy_states"
DEFAULT_START_SAV = STATE_DIR / "red.sav"

CONFIG_DIR = HRL_ROOT / "config"
HRL_CONFIG_PATH = CONFIG_DIR / "hrl_config.yaml"
BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"

CHECKPOINTS_DIR = HRL_ROOT / "checkpoints"
RUNS_DIR = HRL_ROOT / "runs"


def project_root() -> Path:
    """Runtime root for assets and config (equals HRL_ROOT)."""
    return HRL_ROOT
