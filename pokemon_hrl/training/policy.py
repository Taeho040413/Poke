"""Policy factory for HRL training (no dependency on pokemonred_puffer.train CLI)."""

from __future__ import annotations

import importlib

from omegaconf import DictConfig
from torch import nn


def make_policy(env, policy_name: str, config: DictConfig) -> nn.Module:
    try:
        from pufferlib.frameworks import cleanrl
    except ImportError as exc:
        raise RuntimeError(
            "Policy construction requires pufferlib. "
            "Install project dependencies (e.g. pip install -e '.[dev]')."
        ) from exc

    if policy_name not in config.policies:
        raise KeyError(f"Unknown policy '{policy_name}'. Available: {list(config.policies.keys())}")

    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(f"pokemonred_puffer.policies.{policy_module_name}")
    policy_class = getattr(policy_module, policy_class_name)

    policy = policy_class(env, **config.policies[policy_name].policy)
    if config.train.get("use_rnn", True):
        rnn_config = config.policies[policy_name].rnn
        rnn_class = getattr(policy_module, rnn_config.name)
        policy = rnn_class(env, policy, **rnn_config.args)
        policy = cleanrl.RecurrentPolicy(policy)
    else:
        policy = cleanrl.Policy(policy)

    device = str(config.train.get("device", "cpu"))
    return policy.to(device)
