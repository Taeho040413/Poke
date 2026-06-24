"""End-to-end verification: summarizer, planner, DB update, LLM (optional).

Usage:
    python -m pokemon_hrl.scripts.verify_pipeline
    OPENROUTER_API_KEY=sk-... python -m pokemon_hrl.scripts.verify_pipeline --llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.planner.factory import build_planner
from pokemon_hrl.planner.openrouter import OpenRouterPlanner
from pokemon_hrl.planner.criteria import planner_goal_key, subgoal_label
from pokemon_hrl.planner.validation import parse_planner_dict
from pokemon_hrl.summarizer.rule_based import RuleBasedSummarizer
from pokemon_hrl.types import PlannerOutput, ProgressResult, Subgoal, WorldState
from pokemon_hrl.update.information import UpdateInformation
from pokemon_hrl.world_state.merge import merge_extracted_state
from pokemon_hrl.world_state.serialization import world_state_from_dict
from pokemon_hrl.world_state.store import WorldStateStore


def _sample_state(**kwargs) -> WorldState:
    base = dict(
        map_id=2,
        x=10,
        y=12,
        badges=1,
        flags={"EVENT_GOT_POKEDEX": True},
        party=[{"species": 1, "level": 8, "hp": 20, "max_hp": 20}],
        bag=[{"item_id": 4, "quantity": 1}],
        resources={"money": 1200, "first_npc_talk_count": 3},
        success_memory=[],
        failure_memory=[],
        map_visited=[0, 1, 2],
        global_step=42,
    )
    base.update(kwargs)
    return WorldState(**base)


def verify_summarizer() -> None:
    state = _sample_state()
    summary = RuleBasedSummarizer().summarize(state)
    assert "map_id=2" in summary.semantic_progression, summary.semantic_progression
    assert summary.evidence["money"] == 1200
    assert summary.exploration_coverage
    print("[PASS] summarizer: semantic_progression, evidence, exploration")


def verify_planner_output(out: PlannerOutput, label: str) -> None:
    assert out.success_criteria, f"{label}: empty goal success_criteria"
    assert out.hint.get("target_map_id") is not None, f"{label}: missing hint.target_map_id"
    assert isinstance(out.subgoal, list), f"{label}: subgoal not list"
    for sg in out.subgoal:
        assert sg.success_criteria, f"{label}: subgoal missing success_criteria"
    print(
        f"[PASS] {label}: goal_key={planner_goal_key(out)!r}, "
        f"subgoals={len(out.subgoal)}, hint.target_map_id={out.target_map_id}"
    )


def verify_rule_based_planner(cfg) -> PlannerOutput:
    planner = build_planner(cfg)
    state = _sample_state()
    summary = RuleBasedSummarizer().summarize(state)
    out = planner.plan(summary, state)
    verify_planner_output(out, "rule-based planner")
    return out


def verify_db_update(planner: PlannerOutput) -> None:
    store = WorldStateStore(Path("checkpoints") / "_verify_tmp")
    store.set_planner_output(planner)
    state = _sample_state(planner_output=planner, goal_stack=store.state.goal_stack)

    mock_env = MagicMock()
    mock_env.get_game_coords.return_value = (11, 13, 2)
    mock_env.read_m.side_effect = lambda name: {
        "wPartyCount": 1,
        "wNumBagItems": 1,
        "wObtainedBadges": 1,
    }.get(name, 0)
    mock_env.party = [MagicMock(Species=1, Level=8, HP=20, MaxHP=20)]
    mock_env.pyboy.symbol_lookup.return_value = ("wBagItems", 0xD31E)
    mock_env.pyboy.memory = bytearray(0x10000)
    mock_env.events.get_event.return_value = False
    mock_env.seen_map_ids = [1, 1, 1]
    mock_env.first_npc_talk_count = 4
    mock_env.first_object_interaction_count = 0
    mock_env.new_npc_textbox_count = 0
    mock_env.item_count = 0
    mock_env.trainer_battle_win_count = 0
    mock_env.pokecenter_heal_hp_count = 0

    store.replace(state)
    updater = UpdateInformation()

    # subgoal success
    progress = ProgressResult(
        subgoal_success=True,
        subgoal=subgoal_label(planner.subgoal[0]),
        subgoal_index=0,
    )
    updater.apply(store, mock_env, progress=progress, global_step=43)
    assert len(store.state.success_memory) == 1
    assert store.state.goal_stack["current_index"] == 1
    assert store.state.map_id == 2
    print("[PASS] db update: subgoal success -> success_memory + goal_stack index")

    # goal success
    progress = ProgressResult(success=True, reason="success_criteria_met")
    updater.apply(store, mock_env, progress=progress, global_step=44)
    assert any(m.get("kind") == "goal_complete" for m in store.state.success_memory)
    print("[PASS] db update: goal success -> goal_complete in success_memory")

    # failure
    progress = ProgressResult(failure=True, reason="no_progress")
    updater.apply(store, mock_env, progress=progress, global_step=45)
    assert any(f.get("cause") == "no_progress" for f in store.state.failure_memory)
    print("[PASS] db update: failure -> failure_memory")

    # JSON roundtrip
    raw = store.to_json()
    restored = world_state_from_dict(raw)
    assert restored.planner_output is not None
    assert planner_goal_key(restored.planner_output) == planner_goal_key(planner)
    print("[PASS] db update: JSON serialize/deserialize roundtrip")


def verify_orchestrator_flow(cfg) -> None:
    from pokemon_hrl.loop.orchestrator import HrlOrchestrator
    from pokemon_hrl.mode.progress import ProgressCheck
    from pokemon_hrl.mode.selector import ModeSelector
    from pokemon_hrl.types import Mode

    planner_obj = build_planner(cfg)
    base_env = MagicMock()
    base_env.events = None
    base_env.get_game_coords.return_value = (0, 0, 2)

    env = MagicMock()
    env.env = None
    env.step.return_value = (
        {"screen": [0]},
        0.5,
        False,
        False,
        {
            "hrl_progress_success": 0,
            "hrl_subgoal_success": 1,
            "hrl_subgoal_completed": "talk",
            "hrl_subgoal_index": 0,
            "hrl_progress_reason": "",
        },
    )
    env.reset.return_value = ({"screen": [0]}, {})

    store = WorldStateStore(Path("checkpoints") / "_verify_tmp")
    extract_fn = lambda _env, global_step=0: _sample_state(global_step=global_step)
    with patch("pokemon_hrl.loop.orchestrator.unwrap_hrl_env", return_value=base_env):
        with patch("pokemon_hrl.loop.orchestrator.extract_world_state", side_effect=extract_fn):
            with patch("pokemon_hrl.update.information.extract_world_state", side_effect=extract_fn):
                orch = HrlOrchestrator(
                    config=cfg,
                    env=env,
                    store=store,
                    selector=ModeSelector(
                        enabled=bool(cfg.hrl.mode_selector.enabled),
                        forced_mode=Mode(cfg.hrl.mode_selector.forced_mode),
                    ),
                    summarizer=RuleBasedSummarizer(),
                    progress=ProgressCheck(),
                    updater=UpdateInformation(),
                    checkpoints=MagicMock(),
                    planner=planner_obj,
                )
                progress, obs, _info = orch.step_once(0)

    assert orch.store.state.planner_output is not None
    assert orch.store.state.recent_summary is not None
    assert isinstance(progress, ProgressResult)
    print("[PASS] orchestrator: step_once with real summarizer/updater/planner")


def verify_llm_mock() -> None:
    """Verify OpenRouterPlanner parses a realistic API response (no network)."""
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "subgoal": [
                                {
                                    "success_criteria": [
                                        "stat_on_target_map:first_npc_talk_count"
                                    ],
                                }
                            ],
                            "hint": {"target_map_id": 2},
                            "success_criteria": ["flag:EVENT_BEAT_BROCK"],
                            "failure_criteria": ["no_progress"],
                        }
                    )
                }
            }
        ]
    }

    cfg = load_hrl_config()
    planner = OpenRouterPlanner(
        cfg.hrl.planner,
        curriculum_path=str(cfg.hrl.curriculum.path),
    )
    state = _sample_state()
    summary = RuleBasedSummarizer().summarize(state)

    def _fake_urlopen(request, timeout=120):
        del request, timeout
        resp = MagicMock()
        resp.read.return_value = json.dumps(fake_response).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            out = planner.plan(summary, state)
    verify_planner_output(out, "LLM planner (mocked HTTP)")


def verify_llm_live(cfg) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[SKIP] LLM live: OPENROUTER_API_KEY not set")
        return

    planner = OpenRouterPlanner(
        cfg.hrl.planner,
        curriculum_path=str(cfg.hrl.curriculum.path),
    )
    state = _sample_state()
    summary = RuleBasedSummarizer().summarize(state)
    print("\n--- Live OpenRouter call ---")
    out = planner.plan(summary, state)
    verify_planner_output(out, "LLM planner (live OpenRouter)")
    print(f"[PASS] LLM live: model={cfg.hrl.planner.model}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Also call live OpenRouter API")
    args = parser.parse_args()

    print("=== HRL Pipeline Verification ===\n")
    cfg = load_hrl_config()

    verify_summarizer()
    planner_out = verify_rule_based_planner(cfg)
    verify_db_update(planner_out)
    verify_orchestrator_flow(cfg)
    verify_llm_mock()
    if args.llm:
        verify_llm_live(cfg)

    print("\n=== ALL CHECKS PASSED ===")
    if not args.llm and not os.environ.get("OPENROUTER_API_KEY"):
        print("(Live LLM skipped - set OPENROUTER_API_KEY and pass --llm to verify)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
