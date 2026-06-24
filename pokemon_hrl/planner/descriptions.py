"""Map name helpers for planner logging."""

from __future__ import annotations

from pokemonred_puffer.data.map import MapIds


def format_map_id(map_id: int | None) -> str:
    if map_id is None:
        return "미지정"
    try:
        name = MapIds(int(map_id)).name
    except ValueError:
        name = "UNKNOWN"
    return f"{name} (map_id={int(map_id)})"
