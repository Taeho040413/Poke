from pathlib import Path

import torch
from omegaconf import OmegaConf

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.training.engine import _load_resume_checkpoint, merge_train_config
from pokemon_hrl.training.checkpoint import find_latest_saved_model


def test_find_latest_saved_model_picks_newest_mtime(tmp_path: Path):
    run_dir = tmp_path / "hrl-interactive-v1"
    run_dir.mkdir()
    older = run_dir / "model_000010.pt"
    newer = run_dir / "model_000020.pt"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.touch()
    newer.touch()

    picked = find_latest_saved_model(tmp_path, "hrl-interactive-v1")
    assert picked == newer


def test_load_resume_checkpoint_skips_when_fresh(tmp_path: Path, monkeypatch):
    cfg = load_hrl_config()
    train_cfg = merge_train_config(cfg)

    class _Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

    policy = _Policy()
    before = {k: v.clone() for k, v in policy.state_dict().items()}

    state, lines = _load_resume_checkpoint(
        cfg,
        train_cfg,
        policy,
        checkpoint_path=None,
        fresh=True,
    )
    assert state is None
    assert lines is None
    after = policy.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_load_resume_checkpoint_loads_policy_weights(tmp_path: Path):
    cfg = load_hrl_config()
    cfg.train.data_dir = str(tmp_path)
    cfg.train.exp_id = "resume-test"
    cfg.hrl.training.exp_id = "resume-test"
    train_cfg = merge_train_config(cfg)

    class _Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    run_dir = tmp_path / "resume-test"
    run_dir.mkdir()
    saved = _Policy()
    torch.nn.init.constant_(saved.linear.weight, 3.14)
    model_path = run_dir / "model_000001.pt"
    torch.save(saved.state_dict(), model_path)

    policy = _Policy()
    state, lines = _load_resume_checkpoint(
        cfg,
        train_cfg,
        policy,
        checkpoint_path=None,
        fresh=False,
    )
    assert state is None
    assert lines is not None
    assert any("정책 가중치 로드 완료" in line for line in lines)
    assert torch.allclose(policy.linear.weight, torch.full((2, 2), 3.14))
