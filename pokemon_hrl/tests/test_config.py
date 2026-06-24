from omegaconf import OmegaConf

from pokemon_hrl.config import clone_hrl_config, load_hrl_config
from pokemon_hrl.rewards.interactive_mode import InteractiveModeRewardEnv
from pokemon_hrl.training.env_factory import INTERACTIVE_REWARD_KEY


def test_load_hrl_config_paths():
    cfg = load_hrl_config()
    assert cfg.env.gb_path.endswith("red.gb")
    assert "pyboy_states" in cfg.env.state_dir
    assert cfg.env.init_state_path.endswith("red.state")
    assert cfg.env.end_episode_on_first_gym is False
    assert "video" in cfg.env.video_dir
    assert cfg.env.auto_remove_all_nonuseful_items is False
    assert cfg.hrl.mode_selector.forced_mode == "interactive"
    assert cfg.hrl.training.mode == "interactive"
    assert cfg.hrl.training.use_curriculum_init_state is False
    assert cfg.train.sqlite_wrapper is False


def test_interactive_reward_key():
    cfg = load_hrl_config()
    assert INTERACTIVE_REWARD_KEY in cfg.rewards
    reward = OmegaConf.to_object(cfg.rewards[INTERACTIVE_REWARD_KEY].reward)
    assert reward["new_tile"] > 0
    assert reward["new_building"] > 0
    assert reward["new_room"] > 0
    assert reward["pokecenter_first_entry"] > 0
    assert reward["target_map_entry"] > 0
    assert reward["new_map"] > 0
    assert reward["party_level"] > 0
    assert reward["pokemon_heal_hp"] > 0
    assert reward["death"] < 0
    assert reward["npc_first_talk"] > 0


def test_interactive_reward_env_mro():
    assert InteractiveModeRewardEnv.__name__ in [
        c.__name__ for c in InteractiveModeRewardEnv.__mro__
    ]


def test_clone_hrl_config_preserves_omegaconf_and_env_keys():
    cfg = load_hrl_config()
    cloned = clone_hrl_config(cfg)
    assert OmegaConf.is_config(cloned)
    assert "video_dir" in cloned.env
    reward = OmegaConf.to_object(cloned.rewards[INTERACTIVE_REWARD_KEY].reward)
    assert reward["npc_first_talk"] > 0


def test_merge_train_config_includes_cleanrl_defaults():
    from pokemon_hrl.training.engine import merge_train_config

    cfg = load_hrl_config()
    train_cfg = merge_train_config(cfg)
    assert hasattr(train_cfg, "archive_states")
    assert train_cfg.archive_states is False
    assert hasattr(train_cfg, "compile_mode")


def test_merge_train_config_validates_batch_layout():
    from pokemon_hrl.training.engine import merge_train_config

    cfg = load_hrl_config()
    cfg.hrl.training.minibatch_size = 100
    try:
        merge_train_config(cfg)
        raise AssertionError("expected ValueError for invalid minibatch_size")
    except ValueError as exc:
        assert "divisible" in str(exc)
