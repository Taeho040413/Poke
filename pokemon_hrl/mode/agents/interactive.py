"""Interactive mode agent wrapper around a torch policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class InteractiveModeAgent:
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy

    def act(self, obs: Any, *, deterministic: bool = False) -> int:
        self.policy.eval()
        with torch.no_grad():
            obs_tensor = self._to_tensor(obs, device=self._policy_device())
            output = self.policy(obs_tensor)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())

    def to_policy(self) -> torch.nn.Module:
        return self.policy

    def _policy_device(self) -> torch.device:
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _to_tensor(
        obs: Any, *, device: torch.device
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(obs, dict):
            out = {}
            for key, value in obs.items():
                arr = np.asarray(value)
                tensor = torch.as_tensor(arr, device=device)
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() > 1 and tensor.shape[0] != 1:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                out[key] = tensor
            return out
        tensor = torch.as_tensor(np.asarray(obs), device=device)
        return tensor.unsqueeze(0)
