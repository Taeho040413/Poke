"""Weights & Biases helpers for HRL interactive training."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import wandb
from omegaconf import DictConfig, OmegaConf


def _wandb_config_payload(config: DictConfig) -> dict[str, Any]:
    reward_key = next(iter(config.rewards.keys()), None)
    policy_key = "multi_convolutional.MultiConvolutionalPolicy"
    return {
        "train": OmegaConf.to_container(config.train, resolve=True),
        "env": OmegaConf.to_container(config.env, resolve=True),
        "hrl": OmegaConf.to_container(config.hrl, resolve=True),
        "reward_module": reward_key,
        "policy_module": policy_key,
        "reward": OmegaConf.to_container(
            config.rewards[reward_key], resolve=True
        )
        if reward_key
        else None,
        "policy": OmegaConf.to_container(
            config.policies[policy_key], resolve=True
        )
        if policy_key in config.policies
        else None,
        "wrappers": OmegaConf.to_container(config.wrappers, resolve=True),
        "rnn": bool(config.train.get("use_rnn", True)),
    }


def _wandb_group(config: DictConfig, exp_name: str) -> str:
    group = OmegaConf.select(config, "wandb.group")
    if group is None or str(group).strip() in ("", "~", "null", "None"):
        return exp_name
    return str(group)


def _wandb_run_name(exp_name: str, run_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = run_id[:8] if run_id else timestamp
    return f"{exp_name}-{timestamp}-{suffix}"


@contextmanager
def init_hrl_wandb(
    config: DictConfig,
    *,
    exp_name: str,
) -> Iterator[wandb.wandb_sdk.wandb_run.Run | None]:
    """Initialize a fresh W&B run when ``config.train.track`` is enabled.

    Each invocation creates a new run so checkpoint resume steps never collide
    with a prior run's logged step counter. ``exp_name`` is used for grouping
    and display only; local checkpoint paths are unchanged.
    """
    track = bool(OmegaConf.select(config, "train.track", default=False))
    if not track:
        yield None
        return

    project = OmegaConf.select(config, "wandb.project")
    entity = OmegaConf.select(config, "wandb.entity")
    if not project:
        raise ValueError("wandb.project is required when train.track=true")
    if not entity:
        raise ValueError("wandb.entity is required when train.track=true")

    run_id = wandb.util.generate_id()
    wandb_kwargs: dict[str, Any] = {
        "id": run_id,
        "project": project,
        "entity": entity,
        "group": _wandb_group(config, exp_name),
        "config": _wandb_config_payload(config),
        "name": _wandb_run_name(exp_name, run_id),
        "monitor_gym": True,
        "save_code": True,
        "resume": False,
    }
    base_url = OmegaConf.select(config, "wandb.base_url")
    if base_url is not None and str(base_url).strip():
        wandb_kwargs["settings"] = wandb.Settings(base_url=str(base_url).strip())

    client = wandb.init(**wandb_kwargs)
    try:
        yield client
    finally:
        client.finish()
