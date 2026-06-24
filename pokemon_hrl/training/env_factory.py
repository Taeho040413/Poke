"""Build Interactive HRL environments for training."""

from __future__ import annotations

import importlib

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from pokemon_hrl.env.base_hrl_env import CurriculumWrapper
from pokemon_hrl.env.goal_memory import GoalMemoryConfig
from pokemon_hrl.env.interactive_env import HrlInteractiveRewardEnv
from pokemon_hrl.env.mode_steps import ModeMaxStepsWrapper
from pokemon_hrl.env.progress_wrapper import ProgressCheckWrapper
from pokemon_hrl.planner.validation import parse_planner_dict, planner_output_to_dict
from pokemon_hrl.training.curriculum import pick_scenario
from pokemon_hrl.training.shared_plan import SharedPlanStore, get_shared_plan_store

INTERACTIVE_REWARD_KEY = "hrl.rewards.interactive_mode.InteractiveModeRewardEnv"
INTERACTIVE_WRAPPER_PROFILE = "interactive"
SUPPORTED_TRAINING_MODES = frozenset({"interactive"})


def _import_wrapper(module_name: str, class_name: str):
    if module_name.startswith("hrl."):
        dotted = f"pokemon_hrl{module_name[3:]}"
    elif module_name.startswith("pokemon_hrl."):
        dotted = module_name
    else:
        dotted = f"pokemonred_puffer.wrappers.{module_name}"
    return getattr(importlib.import_module(dotted), class_name)


def _apply_wrappers(env: gym.Env, config: DictConfig, profile: str) -> gym.Env:
    for wrapper_dict in config.wrappers.get(profile, []):
        for key, args in wrapper_dict.items():
            module_name, class_name = key.rsplit(".", 1)
            if class_name == "StreamWrapper":
                continue
            wrapper_cls = _import_wrapper(module_name, class_name)
            env = wrapper_cls(env, OmegaConf.create(args))
    return env


def bootstrap_shared_planner(
    config: DictConfig,
    *,
    scenario_index: int = 0,
    shared_plan: SharedPlanStore | None = None,
) -> SharedPlanStore:
    """Load curriculum planner into the process-wide shared store once."""
    store = shared_plan or get_shared_plan_store()
    if store.planner is not None:
        return store
    scenario = pick_scenario(scenario_index, config.hrl.curriculum.path)
    store.set_planner(parse_planner_dict(planner_output_to_dict(scenario.planner)))
    return store


def make_interactive_env(
    config: DictConfig,
    *,
    scenario_index: int = 0,
    puffer_wrapper: bool = True,
    shared_plan: SharedPlanStore | None = None,
) -> gym.Env:
    hrl_cfg = config.hrl
    training_mode = str(OmegaConf.select(config, "hrl.training.mode", default="interactive"))
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f"Unsupported training mode {training_mode!r}. "
            f"Supported: {sorted(SUPPORTED_TRAINING_MODES)}"
        )

    plan_store = bootstrap_shared_planner(
        config,
        scenario_index=scenario_index,
        shared_plan=shared_plan,
    )
    planner = plan_store.planner
    if planner is None:
        raise RuntimeError("Shared planner was not initialized")

    env_config = OmegaConf.create(OmegaConf.to_container(config.env, resolve=True))
    if bool(OmegaConf.select(config, "hrl.training.use_curriculum_init_state", default=False)):
        scenario = pick_scenario(scenario_index, hrl_cfg.curriculum.path)
        if scenario.init_state:
            env_config.init_state = scenario.init_state
            env_config.init_state_path = None

    reward_cfg = OmegaConf.create(
        OmegaConf.to_container(config.rewards[INTERACTIVE_REWARD_KEY].reward, resolve=True)
    )

    goal_memory_cfg = GoalMemoryConfig.from_omega(config)

    tile_blocked_prefix = "hrl.execution.tile_blocked"
    env = HrlInteractiveRewardEnv(
        env_config,
        reward_cfg,
        max_tile_substeps=int(hrl_cfg.execution.max_tile_substeps),
        battle_action_mask=bool(
            OmegaConf.select(config, "hrl.execution.battle_action_mask", default=True)
        ),
        tile_move_enabled=bool(
            OmegaConf.select(config, "hrl.execution.tile_move_enabled", default=True)
        ),
        tile_blocked_enabled=bool(
            OmegaConf.select(config, f"{tile_blocked_prefix}.enabled", default=True)
        ),
        tile_blocked_ttl_steps=int(
            OmegaConf.select(config, f"{tile_blocked_prefix}.ttl_steps", default=100)
        ),
        tile_blocked_weaken_weight=float(
            OmegaConf.select(config, f"{tile_blocked_prefix}.weaken_weight", default=0.05)
        ),
        tile_blocked_retry_window_steps=int(
            OmegaConf.select(config, f"{tile_blocked_prefix}.retry_window_steps", default=50)
        ),
        tile_blocked_confidence_threshold=int(
            OmegaConf.select(config, f"{tile_blocked_prefix}.confidence_threshold", default=1)
        ),
        goal_memory_config=goal_memory_cfg,
    )

    env = _apply_wrappers(env, config, INTERACTIVE_WRAPPER_PROFILE)
    env = CurriculumWrapper(env, planner)
    env = ModeMaxStepsWrapper(env, int(hrl_cfg.interactive.max_steps))
    log_goals = bool(OmegaConf.select(config, "hrl.logging.goal_events", default=True))
    subgoal_reward = float(
        OmegaConf.select(config, "hrl.progress.subgoal_success_reward", default=3.0)
    )
    goal_reward = float(
        OmegaConf.select(config, "hrl.progress.goal_success_reward", default=5.0)
    )
    reward_floor = OmegaConf.select(config, "hrl.checkpoint.reward_floor", default=-10.0)
    if reward_floor is not None:
        reward_floor = float(reward_floor)
    rollback_penalty = float(
        OmegaConf.select(
            config,
            "hrl.checkpoint.reward_floor_rollback_penalty",
            default=-1.0,
        )
    )
    env = ProgressCheckWrapper(
        env,
        planner,
        log_goal_events=log_goals,
        subgoal_success_reward=subgoal_reward,
        goal_success_reward=goal_reward,
        reward_floor=reward_floor,
        reward_floor_rollback_penalty=rollback_penalty,
    )

    if puffer_wrapper:
        try:
            from pufferlib import emulation
        except ImportError as exc:
            raise RuntimeError(
                "pufferlib is required for puffer_wrapper=True. "
                "Install project dependencies (e.g. pip install -e '.[dev]')."
            ) from exc
        env = emulation.GymnasiumPufferEnv(env=env)
    return env


def env_creator(config: DictConfig, scenario_index: int = 0) -> gym.Env:
    return make_interactive_env(config, scenario_index=scenario_index)
