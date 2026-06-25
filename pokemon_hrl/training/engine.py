"""Interactive-mode PPO training engine (shared by train + autotune)."""

from __future__ import annotations

import pokemon_hrl  # noqa: F401 — bootstrap pokemonred_puffer import path

from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pokemon_hrl.config import clone_hrl_config

from pokemon_hrl.planner.logging import log_planner_output
from pokemon_hrl.training.curriculum import pick_scenario
from pokemon_hrl.training.env_factory import bootstrap_shared_planner, make_interactive_env
from pokemon_hrl.training.shared_plan import get_shared_plan_store
from pokemon_hrl.training.checkpoint import (
    effective_resume_path,
    load_policy_checkpoint,
    log_resume_intent,
    peek_resume_model_path,
    resolve_resume_checkpoint,
)
from pokemon_hrl.training.hrl_checkpoint import (
    TrainingCheckpointCoordinator,
    apply_save_point_to_env_config,
)
from pokemon_hrl.training.policy import make_policy
from pokemon_hrl.training.wandb_logging import init_hrl_wandb


def resolve_device(raw: str) -> str:
    import torch

    if raw and raw != "auto":
        return raw
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_interactive_env_for_vector(
    config_container: dict[str, Any],
    scenario_index: int,
    *_args: Any,
    **_kwargs: Any,
):
    """Pickle-safe env creator for Windows multiprocessing spawn."""
    cfg = OmegaConf.create(config_container)
    shared_plan = get_shared_plan_store()
    bootstrap_shared_planner(cfg, scenario_index=scenario_index, shared_plan=shared_plan)
    return make_interactive_env(
        cfg,
        scenario_index=scenario_index,
        puffer_wrapper=True,
        shared_plan=shared_plan,
    )


def resolve_vector_backend(train_cfg: Namespace):
    """Pick pufferlib vecenv backend from config (serial vs multiprocessing)."""
    from pufferlib import vector

    raw = str(getattr(train_cfg, "vectorization", "auto")).lower()
    num_envs = int(getattr(train_cfg, "num_envs", 1))

    if raw in ("serial", "none"):
        return vector.Serial
    if raw in ("multiprocessing", "mp", "parallel"):
        return vector.Multiprocessing
    if raw == "ray":
        return vector.Ray
    # auto: parallel only when multiple envs are requested
    if num_envs > 1:
        return vector.Multiprocessing
    return vector.Serial


# Keys required by CleanPuffeRL but not always present in hrl_config train sections.
_CLEANRL_PPO_DEFAULTS: dict[str, object] = {
    "archive_states": False,
    "compile_mode": "default",
    "early_stop": None,
    "required_rate": None,
    "swarm": None,
    "swarm_keep_pct": 0.5,
    "save_overlay": True,
    "overlay_interval": 10,
    "grid_map_scale": 4,
    "target_kl": None,
    "anneal_lr": False,
    "one_epoch": None,
    "save_on_first_gym": False,
    "save_on_exit": False,
    "save_policy_on_interrupt": True,
    "wandb_environment_metrics": "reward_only",
    "eval_interval": 1,
    "load_optimizer_state": False,
    "vectorization": "auto",
}


def merge_train_config(config: DictConfig) -> Namespace:
    merged = OmegaConf.merge(
        OmegaConf.create(_CLEANRL_PPO_DEFAULTS),
        config.get("train", OmegaConf.create({})),
        config.hrl.get("training", OmegaConf.create({})),
    )
    merged.device = resolve_device(str(merged.get("device", "auto")))
    merged.env = "Pokemon HRL Interactive"
    merged.save_game_on_checkpoint = bool(
        OmegaConf.select(config, "hrl.checkpoint.save_game_state", default=True)
        and merged.get("save_checkpoint", True)
    )
    _validate_batch_layout(merged)
    return Namespace(**OmegaConf.to_container(merged, resolve=True))


def _validate_batch_layout(merged: DictConfig) -> None:
    batch_size = int(merged.batch_size)
    minibatch_size = int(merged.minibatch_size)
    bptt_horizon = int(merged.bptt_horizon)
    if batch_size % minibatch_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by minibatch_size ({minibatch_size})"
        )
    if minibatch_size % bptt_horizon != 0:
        raise ValueError(
            f"minibatch_size ({minibatch_size}) must be divisible by bptt_horizon ({bptt_horizon})"
        )


def _reward_from_stats(stats: dict[str, Any]) -> float | None:
    for key in ("reward_sum", "stats/reward_sum", "episode_return"):
        if key in stats:
            try:
                return float(stats[key])
            except (TypeError, ValueError):
                continue
    for key, value in stats.items():
        if "reward_sum" in key:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _log_policy_update(trainer: Any, *, reward: float | None) -> None:
    losses = trainer.losses
    sps = float(getattr(trainer.profile, "SPS", 0.0) or 0.0)
    reward_str = f"{reward:.3f}" if reward is not None else "n/a"
    print(
        f"[ppo] update={int(trainer.epoch)} "
        f"steps={int(trainer.global_step)} "
        f"reward={reward_str} "
        f"pg={float(losses.policy_loss):.4f} "
        f"vf={float(losses.value_loss):.4f} "
        f"kl={float(losses.approx_kl):.4f} "
        f"sps={sps:.0f}",
        flush=True,
    )


def _load_resume_checkpoint(
    cfg: DictConfig,
    train_cfg: Namespace,
    policy: Any,
    *,
    checkpoint_path: Path | None,
    fresh: bool,
) -> tuple[dict[str, Any] | None, list[str] | None]:
    resume_src, resume_auto_attempted, resume_fresh = effective_resume_path(
        checkpoint_path,
        cfg,
        resume_latest_cli=False,
        fresh_cli=fresh,
    )
    model_pt, trainer_pt = resolve_resume_checkpoint(resume_src)
    log_resume_intent(
        resume_src,
        resume_auto_attempted,
        model_pt,
        resume_fresh=resume_fresh,
    )
    if model_pt is None:
        return None, None

    load_policy_checkpoint(policy, model_pt, str(train_cfg.device))
    lines = [
        f"[resume] 정책 가중치 로드 완료: {model_pt.expanduser().resolve()}",
    ]

    resume_trainer_state = None
    if trainer_pt is not None and bool(getattr(train_cfg, "load_optimizer_state", False)):
        import torch

        resume_trainer_state = torch.load(
            trainer_pt,
            map_location=str(train_cfg.device),
            weights_only=False,
        )
        lines.append(
            f"[resume] 옵티마이저 상태 로드 완료: {trainer_pt.expanduser().resolve()}"
        )
    return resume_trainer_state, lines


def run_interactive_training(
    config: DictConfig,
    *,
    scenario_index: int = 0,
    timesteps: int | None = None,
    exp_suffix: str = "",
    collect_rewards: bool = True,
    checkpoint_path: Path | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Train Interactive mode policy; return summary metrics."""
    try:
        from pufferlib import vector
        from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL
    except ImportError as exc:
        raise RuntimeError(
            "Interactive training requires pufferlib. "
            "Install project dependencies (e.g. pip install -e '.[dev]')."
        ) from exc

    cfg = clone_hrl_config(config)
    train_cfg = merge_train_config(cfg)
    cfg.train.device = train_cfg.device

    if timesteps is not None:
        train_cfg.total_timesteps = int(timesteps)

    if exp_suffix:
        train_cfg.exp_id = f"{train_cfg.exp_id}-{exp_suffix}"

    resume_model_pt = peek_resume_model_path(checkpoint_path, cfg, fresh_cli=fresh)
    apply_save_point_to_env_config(cfg, fresh=fresh, resume_model_pt=resume_model_pt)
    shared_plan = get_shared_plan_store()
    bootstrap_shared_planner(cfg, scenario_index=scenario_index, shared_plan=shared_plan)
    hrl_checkpoint = TrainingCheckpointCoordinator.from_config(
        cfg,
        shared_plan=shared_plan,
    )

    if bool(OmegaConf.select(cfg, "hrl.logging.goal_events", default=True)):
        scenario = pick_scenario(scenario_index, cfg.hrl.curriculum.path)
        log_planner_output(
            scenario.planner,
            source="training",
            scenario_index=scenario_index,
        )

    env_creator = partial(
        _make_interactive_env_for_vector,
        OmegaConf.to_container(cfg, resolve=True),
        int(scenario_index),
    )

    num_envs = int(train_cfg.num_envs)
    num_workers = int(train_cfg.num_workers)
    env_batch_size = int(train_cfg.env_batch_size)
    if num_workers > num_envs:
        raise ValueError(f"num_workers ({num_workers}) must be <= num_envs ({num_envs})")
    if num_envs % max(1, num_workers) != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by num_workers ({num_workers})"
        )
    if env_batch_size > num_envs:
        raise ValueError(
            f"env_batch_size ({env_batch_size}) must be <= num_envs ({num_envs})"
        )
    if num_envs % env_batch_size != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by env_batch_size ({env_batch_size})"
        )

    backend = resolve_vector_backend(train_cfg)
    vecenv_kwargs: dict[str, Any] = {
        "num_envs": num_envs,
        "num_workers": num_workers,
        "batch_size": env_batch_size,
        "backend": backend,
    }
    if bool(getattr(train_cfg, "zero_copy", False)):
        vecenv_kwargs["zero_copy"] = True

    vecenv = vector.make(env_creator, **vecenv_kwargs)

    policy = make_policy(
        vecenv.driver_env,
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

    reward_samples: list[float] = []
    last_stats: dict[str, Any] = {}
    global_steps = 0
    epochs = 0

    with init_hrl_wandb(cfg, exp_name=str(train_cfg.exp_id)) as wandb_client:
        with CleanPuffeRL(
            exp_name=str(train_cfg.exp_id),
            config=train_cfg,
            vecenv=vecenv,
            policy=policy,
            env_send_queues=[],
            env_recv_queues=[],
            sqlite_db=None,
            wandb_client=wandb_client,
            resume_trainer_state=resume_trainer_state,
            resume_load_log_lines=resume_load_log_lines,
        ) as trainer:
            eval_interval = max(1, int(getattr(train_cfg, "eval_interval", 1)))
            ppo_step = 0
            while not trainer.done_training():
                if timesteps is not None and trainer.global_step >= timesteps:
                    break
                stats, _ = trainer.evaluate()
                hrl_checkpoint.process_rollout(trainer)
                if ppo_step % eval_interval == 0:
                    last_stats = dict(stats or {})
                trainer.train()
                if not last_stats and trainer.stats:
                    last_stats = dict(trainer.stats)
                global_steps = int(trainer.global_step)
                epochs = int(trainer.epoch)
                rollout_reward = _reward_from_stats(last_stats)
                _log_policy_update(trainer, reward=rollout_reward)
                if collect_rewards:
                    if rollout_reward is not None:
                        reward_samples.append(rollout_reward)
                ppo_step += 1

    mean_reward = float(np.mean(reward_samples)) if reward_samples else 0.0
    return {
        "mean_reward": mean_reward,
        "last_reward": reward_samples[-1] if reward_samples else 0.0,
        "global_steps": global_steps,
        "epochs": epochs,
        "exp_id": str(train_cfg.exp_id),
        "last_stats": last_stats,
        "reward_samples": reward_samples,
    }
