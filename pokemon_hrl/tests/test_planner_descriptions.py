import pytest

from pokemon_hrl.planner.descriptions import format_map_id
from pokemon_hrl.planner.validation import PlannerValidationError, validate_subgoal_descriptions
from pokemon_hrl.types import Subgoal


def test_format_map_id_pewter():
    assert format_map_id(2) == "PEWTER_CITY (map_id=2)"


def test_validate_subgoal_descriptions_requires_all_fields():
    with pytest.raises(PlannerValidationError, match="where"):
        validate_subgoal_descriptions(
            [Subgoal(success_criteria=["flag:EVENT_GOT_STARTER"], what="x", how="y")]
        )


def test_validate_subgoal_descriptions_accepts_complete_subgoal():
    validate_subgoal_descriptions(
        [
            Subgoal(
                success_criteria=["flag:EVENT_GOT_STARTER"],
                where="오크 연구소",
                what="스타터 선택",
                how="공 앞에서 A",
            )
        ]
    )
