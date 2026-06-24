"""Closed-loop HRL with on-policy PPO updates (planner + orchestrator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pokemon_hrl  # noqa: F401 — bootstrap pokemonred_puffer import path

from pokemon_hrl.config import clone_hrl_config
from pokemon_hrl.loop.orchestrator import HrlOrchestrator
from pokemon_hrl.loop.policy_loader import _env_for_policy
from pokemon_hrl.loop.vecenv_shim import OrchestratorVecShim
from pokemon_hrl.mode.agents.interactive import InteractiveModeAgent
from pokemon_hrl.training.engine import (
    _load_resume_checkpoint,
    _log_policy_update,
    _reward_from_stats,
    merge_train_config,
)
from pokemon_hrl.training.env_factory import bootstrap_shared_planner, make_interactive_env
from pokemon_hrl.training.shared_plan import SharedPlanStore, get_shared_plan_store
from pokemon_hrl.training.hrl_checkpoint import (
    TrainingCheckpointCoordinator,
    apply_save_point_to_env_config,
)
from pokemon_hrl.training.checkpoint import peek_resume_model_path
from pokemon_hrl.training.policy import make_policy


def run_hrl_loop(
    config,
    *,
    scenario_index: int = 0,
    max_steps: int | None = None,
    checkpoint_path: Path | None = None,
    fresh: bool = False,
    headless: bool | None = None,
) -> dict[str, Any]:
    """Run planner-orchestrated loop with PPO training (same trainer as train_interactive)."""
    try:
        from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL
    except ImportError as exc:
        raise RuntimeError(
            "HRL loop training requires pufferlib. "
            "Install project dependencies (e.g. pip install -e '.[dev]')."
        ) from exc

    cfg = clone_hrl_config(config)
    if headless is not None:
        cfg.env.headless = bool(headless)
    train_cfg = merge_train_config(cfg)
    cfg.train.device = train_cfg.device
    cfg.hrl.training.scenario_index = int(scenario_index)

    if max_steps is not None:
        train_cfg.total_timesteps = int(max_steps)

    resume_model_pt = peek_resume_model_path(checkpoint_path, cfg, fresh_cli=fresh)
    apply_save_point_to_env_config(cfg, fresh=fresh, resume_model_pt=resume_model_pt)

    shared_plan = get_shared_plan_store()
    bootstrap_shared_planner(cfg, scenario_index=scenario_index, shared_plan=shared_plan)

    env = make_interactive_env(
        cfg,
        scenario_index=scenario_index,
        puffer_wrapper=True,
        shared_plan=shared_plan,
    )
    obs, _ = env.reset()

    policy_env = _env_for_policy(env)
    policy = make_policy(
        policy_env,
        "multi_convolutional.MultiConvolutionalPolicy",
        cfg,
    )

    resume_trainer_state, resume_load_log_lines = _load_resume_checkpoint(
        cfg,
        train_cfg,
        policy,
        checkpoint_path=checkpoint_path,
        fresh=fresh,
    )

    agent = InteractiveModeAgent(policy)
    orchestrator = HrlOrchestrator.from_config(cfg, env, agent=agent)
    vecenv = OrchestratorVecShim(env, orchestrator)
    vecenv._obs = obs

    hrl_checkpoint = TrainingCheckpointCoordinator.from_config(
        cfg,
        store=orchestrator.store,
        shared_plan=shared_plan,
    )

    last_stats: dict[str, Any] = {}
    global_steps = 0
    epochs = 0

    with CleanPuffeRL(
        exp_name=str(train_cfg.exp_id),
        config=train_cfg,
        vecenv=vecenv,
        policy=policy,
        env_send_queues=[],
        env_recv_queues=[],
        sqlite_db=None,
        wandb_client=None,
        resume_trainer_state=resume_trainer_state,
        resume_load_log_lines=resume_load_log_lines,
    ) as trainer:
        eval_interval = max(1, int(getattr(train_cfg, "eval_interval", 1)))
        ppo_step = 0
        step_limit = int(train_cfg.total_timesteps)
        while trainer.global_step < step_limit and not trainer.done_training():
            stats, _ = trainer.evaluate()
            hrl_checkpoint.process_rollout(trainer)
            if ppo_step % eval_interval == 0:
                last_stats = dict(stats or {})
            trainer.train()
            if not last_stats and trainer.stats:
                last_stats = dict(trainer.stats)
            global_steps = int(trainer.global_step)
            epochs = int(trainer.epoch)
            _log_policy_update(trainer, reward=_reward_from_stats(last_stats))
            ppo_step += 1

    return {
        "global_steps": global_steps,
        "epochs": epochs,
        "exp_id": str(train_cfg.exp_id),
        "last_stats": last_stats,
    }
