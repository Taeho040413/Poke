"""Map ID helpers — re-exports MapIds and provides lookup utilities."""

from __future__ import annotations

from pokemonred_puffer.data.map import MapIds

# Commonly referenced early-game maps (values match pokemonred_puffer MapIds).
EARLY_GAME_MAPS = (
    MapIds.PALLET_TOWN,
    MapIds.VIRIDIAN_CITY,
    MapIds.PEWTER_CITY,
    MapIds.LAVENDER_TOWN,
    MapIds.VERMILION_CITY,
    MapIds.ROUTE_1,
    MapIds.ROUTE_2,
    MapIds.OAKS_LAB,
    MapIds.VIRIDIAN_POKECENTER,
    MapIds.VIRIDIAN_MART,
    MapIds.VIRIDIAN_FOREST,
    MapIds.PEWTER_GYM,
    MapIds.PEWTER_POKECENTER,
    MapIds.VERMILION_POKECENTER,
    MapIds.VERMILION_GYM,
)

_NAME_TO_ID: dict[str, int] = {member.name: int(member.value) for member in MapIds}
_ID_TO_NAME: dict[int, str] = {int(member.value): member.name for member in MapIds}


def map_id_to_name(map_id: int) -> str | None:
    return _ID_TO_NAME.get(int(map_id))


def map_name_to_id(name: str) -> int | None:
    normalized = str(name).strip().upper()
    if not normalized:
        return None
    return _NAME_TO_ID.get(normalized)


def get_maps_for_names(names: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in names:
        map_id = map_name_to_id(name)
        if map_id is not None:
            result[name.upper()] = map_id
    return result


__all__ = [
    "EARLY_GAME_MAPS",
    "MapIds",
    "get_maps_for_names",
    "map_id_to_name",
    "map_name_to_id",
]
