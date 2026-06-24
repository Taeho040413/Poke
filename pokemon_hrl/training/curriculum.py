"""Curriculum scenarios for LLM-free Interactive training."""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

from omegaconf import OmegaConf

from pokemon_hrl.planner.validation import parse_planner_dict
from pokemon_hrl.types import PlannerOutput

from pokemon_hrl.paths import HRL_ROOT

_DEFAULT_CURRICULUM = HRL_ROOT / "training" / "curriculum.yaml"


@dataclass
class CurriculumScenario:
    planner: PlannerOutput
    init_state: str | None = None


def _resolve_curriculum_path(path: str | Path | None) -> Path:
    if path is None:
        return _DEFAULT_CURRICULUM
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    under_root = (HRL_ROOT / candidate).resolve()
    if under_root.is_file():
        return under_root
    return candidate.resolve()


def load_curriculum(path: str | Path | None = None) -> list[CurriculumScenario]:
    curriculum_path = _resolve_curriculum_path(path)
    raw = OmegaConf.load(curriculum_path)
    scenarios = []
    for item in raw.get("scenarios", []):
        planner_data = {
            "subgoal": OmegaConf.to_container(item.get("subgoal", []), resolve=True),
            "hint": dict(OmegaConf.to_container(item.get("hint", {}), resolve=True)),
            "success_criteria": [
                str(x) for x in item.get("success_criteria", [])
            ],
            "failure_criteria": [
                str(x) for x in item.get("failure_criteria", [])
            ],
        }
        scenarios.append(
            CurriculumScenario(
                planner=parse_planner_dict(planner_data),
                init_state=str(item.get("init_state")) if item.get("init_state") else None,
            )
        )
    return scenarios


def pick_scenario(index: int = 0, path: str | Path | None = None) -> CurriculumScenario:
    scenarios = load_curriculum(path)
    if not scenarios:
        return CurriculumScenario(
            planner=parse_planner_dict(
                {
                    "subgoal": [
                        {
                            "success_criteria": [
                                "stat_on_target_map:first_npc_talk_count"
                            ],
                        }
                    ],
                    "hint": {"target_map_id": 0},
                    "success_criteria": ["map_reached:0"],
                    "failure_criteria": [],
                }
            )
        )
    return scenarios[index % len(scenarios)]
