"""Extract symbolic WorldState from RedGymEnv."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokemon_hrl.types import WorldState

if TYPE_CHECKING:
    from pokemonred_puffer.environment import RedGymEnv


def extract_world_state(env: RedGymEnv, *, global_step: int = 0) -> WorldState:
    x, y, map_id = env.get_game_coords()
    party = []
    party_size = max(0, min(int(env.read_m("wPartyCount")), 6))
    for i in range(party_size):
        party.append(
            {
                "species": int(env.party[i].Species),
                "level": int(env.party[i].Level),
                "hp": int(env.party[i].HP),
                "max_hp": int(env.party[i].MaxHP),
            }
        )

    bag = []
    from pokemonred_puffer.data.items import MAX_ITEM_CAPACITY

    num_items = max(0, min(int(env.read_m("wNumBagItems")), MAX_ITEM_CAPACITY))
    _, bag_addr = env.pyboy.symbol_lookup("wBagItems")
    bag_start = int(bag_addr)
    bag_end = bag_start + num_items * 2
    if num_items > 0 and bag_end > bag_start:
        raw = env.pyboy.memory[bag_start:bag_end]
        for idx in range(0, len(raw), 2):
            item_id = int(raw[idx])
            qty = int(raw[idx + 1]) if idx + 1 < len(raw) else 0
            if item_id:
                bag.append({"item_id": item_id, "quantity": qty})

    from pokemonred_puffer.data.events import EVENTS

    flags: dict[str, bool] = {}
    for name in EVENTS:
        try:
            if env.events.get_event(name):
                flags[name] = True
        except (KeyError, ValueError, IndexError, AttributeError):
            continue

    money_addr = env.pyboy.symbol_lookup("wPlayerMoney")[1]
    return WorldState(
        map_id=int(map_id),
        x=int(x),
        y=int(y),
        badges=int(env.read_m("wObtainedBadges")),
        flags=flags,
        party=party,
        bag=bag,
        resources={
            "money": int.from_bytes(
                env.pyboy.memory[money_addr : money_addr + 3],
                "little",
            ),
            "first_npc_talk_count": int(getattr(env, "first_npc_talk_count", 0)),
            "first_object_interaction_count": int(
                getattr(env, "first_object_interaction_count", 0)
            ),
            "new_npc_textbox_count": int(getattr(env, "new_npc_textbox_count", 0)),
            "item_count": int(getattr(env, "item_count", 0)),
            "trainer_battle_win_count": int(getattr(env, "trainer_battle_win_count", 0)),
            "wild_battle_win_count": int(getattr(env, "wild_battle_win_count", 0)),
            "pokecenter_heal_hp_count": int(getattr(env, "pokecenter_heal_hp_count", 0)),
        },
        map_visited=sorted(int(m) for m, seen in enumerate(env.seen_map_ids) if seen > 0),
        global_step=global_step,
    )
