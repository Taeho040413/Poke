"""Tests for CleanPuffeRL trainer_state LSTM persistence."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("pufferlib")

from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL


def test_restore_lstm_state_copies_tensors():
    trainer = CleanPuffeRL.__new__(CleanPuffeRL)
    trainer.config = Namespace(device="cpu")
    trainer.experience = MagicMock()
    trainer.experience.lstm_h = torch.zeros(1, 2, 4)
    trainer.experience.lstm_c = torch.zeros(1, 2, 4)

    saved_h = torch.ones(1, 2, 4)
    saved_c = torch.full((1, 2, 4), 2.0)
    trainer._restore_lstm_state({"lstm_h": saved_h, "lstm_c": saved_c})

    assert torch.equal(trainer.experience.lstm_h, saved_h)
    assert torch.equal(trainer.experience.lstm_c, saved_c)


def test_restore_lstm_state_skips_when_missing_keys():
    trainer = CleanPuffeRL.__new__(CleanPuffeRL)
    trainer.config = Namespace(device="cpu")
    trainer.experience = MagicMock()
    trainer.experience.lstm_h = torch.zeros(1, 2, 4)
    trainer.experience.lstm_c = torch.zeros(1, 2, 4)

    trainer._restore_lstm_state({})
    assert torch.equal(trainer.experience.lstm_h, torch.zeros(1, 2, 4))
