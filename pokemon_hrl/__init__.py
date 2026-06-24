"""Pokemon HRL — self-contained package root."""

from __future__ import annotations

import sys
from pathlib import Path

from pokemon_hrl.paths import HRL_ROOT, ROM_PATH, project_root

__all__ = ["HRL_ROOT", "ROM_PATH", "project_root"]


def _ensure_pokemonred_puffer_importable() -> None:
    """Allow ``import pokemonred_puffer`` from ``pokemon_hrl/pokemonred_puffer``."""
    root = Path(__file__).resolve().parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_pokemonred_puffer_importable()
