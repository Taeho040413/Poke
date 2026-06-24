"""Load trained Interactive-mode policy for closed-loop runtime."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from pokemon_hrl.mode.agents.interactive import InteractiveModeAgent
from pokemon_hrl.training.checkpoint import (
    effective_resume_path,
    load_policy_checkpoint,
    resolve_resume_checkpoint,
)
from pokemon_hrl.training.policy import make_policy


def _env_for_policy(env):
    """Policy construction expects a Puffer ``GymnasiumPufferEnv`` (``env.emulated``)."""
    if hasattr(env, "emulated"):
        return env
    from pufferlib import emulation

    return emulation.GymnasiumPufferEnv(env=env)


def resolve_policy_checkpoint(
    config: DictConfig,
    *,
    checkpoint_path: Path | None = None,
    fresh: bool = False,
) -> Path | None:
    """Resolve policy ``model_*.pt`` from CLI path or training config."""
    exp_id = str(
        OmegaConf.select(config, "hrl.training.exp_id")
        or OmegaConf.select(config, "train.exp_id")
        or ""
    )
    if not OmegaConf.select(config, "train.exp_id"):
        config.train.exp_id = exp_id
    resume_src, _, _ = effective_resume_path(
        checkpoint_path,
        config,
        resume_latest_cli=False,
        fresh_cli=fresh,
    )
    model_pt, _ = resolve_resume_checkpoint(resume_src)
    return model_pt


def load_interactive_agent(
    config: DictConfig,
    env,
    *,
    checkpoint_path: Path | None = None,
    fresh: bool = False,
) -> InteractiveModeAgent | None:
    """Build policy + ``InteractiveModeAgent`` when a checkpoint exists."""
    model_pt = resolve_policy_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        fresh=fresh,
    )
    if model_pt is None:
        return None

    policy = make_policy(
        env,
        "multi_convolutional.MultiConvolutionalPolicy",
        config,
    )
    device = str(config.train.get("device", "cpu"))
    load_policy_checkpoint(policy, model_pt, device)
    return InteractiveModeAgent(policy)
