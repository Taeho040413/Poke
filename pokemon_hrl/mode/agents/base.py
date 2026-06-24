"""Mode agent protocol."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class ModeAgent(Protocol):
    def act(self, obs: Any, *, deterministic: bool = False) -> int: ...

    def to_policy(self) -> torch.nn.Module: ...
