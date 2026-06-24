"""Pokémon Red English story facts keyed by chapter success-criteria token."""

from __future__ import annotations

from typing import Any

CHAPTER_FACTS: dict[str, dict[str, Any]] = {
    "flag:EVENT_GOT_OAKS_PARCEL": {
        "label": "receive Oak's Parcel",
        "truth": [
            "Oak's Parcel is obtained at Viridian Mart.",
            "The giver is the Viridian Mart clerk.",
            "Professor Oak does not give Oak's Parcel to the player.",
            "Professor Oak is relevant for the later parcel delivery event, not for receiving the parcel.",
        ],
        "required_route": [
            "OAKS_LAB",
            "PALLET_TOWN",
            "ROUTE_1",
            "VIRIDIAN_CITY",
            "VIRIDIAN_MART",
        ],
        "required_target_map_id": 0x2A,
        "required_interaction": {
            "map": "VIRIDIAN_MART",
            "target": "mart_clerk",
            "expected_result": "flag:EVENT_GOT_OAKS_PARCEL",
        },
        "forbidden_final_maps": [0x28],
        "forbidden_targets": ["Professor Oak", "오크 박사"],
    },
    "flag:EVENT_OAK_GOT_PARCEL": {
        "label": "deliver Oak's Parcel to Professor Oak",
        "truth": [
            "After receiving Oak's Parcel, return to Oak's Lab.",
            "Professor Oak receives the parcel in Oak's Lab.",
        ],
        "required_route": [
            "VIRIDIAN_MART",
            "VIRIDIAN_CITY",
            "ROUTE_1",
            "PALLET_TOWN",
            "OAKS_LAB",
        ],
        "required_target_map_id": 0x28,
        "required_interaction": {
            "map": "OAKS_LAB",
            "target": "Professor Oak",
            "expected_result": "flag:EVENT_OAK_GOT_PARCEL",
        },
    },
    "flag:EVENT_BEAT_BROCK": {
        "label": "defeat Brock",
        "truth": [
            "Brock is the Pewter Gym leader.",
            "The player must reach Pewter Gym and defeat Brock.",
        ],
        "required_route": [
            "VIRIDIAN_CITY",
            "ROUTE_2",
            "VIRIDIAN_FOREST",
            "PEWTER_CITY",
            "PEWTER_GYM",
        ],
        "required_target_map_id": 0x36,
        "required_interaction": {
            "map": "PEWTER_GYM",
            "target": "Brock",
            "expected_result": "flag:EVENT_BEAT_BROCK",
        },
    },
}
