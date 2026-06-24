from pokemon_hrl.execution.action_space import ACTION_DIM, HrlAction, is_tile_action


def test_action_dim():
    assert ACTION_DIM == 12


def test_tile_actions():
    assert is_tile_action(HrlAction.TILE_UP)
    assert is_tile_action(HrlAction.TILE_RIGHT)
    assert not is_tile_action(HrlAction.LOW_A)
