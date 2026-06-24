"""Checkpoint discovery and policy resume for HRL / pokemonred_puffer training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from omegaconf import DictConfig


def find_latest_saved_model(
    data_dir: Path,
    exp_id: str | None,
    *,
    global_fallback: bool = False,
) -> Path | None:
    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        return None
    exp_id = (exp_id or "").strip()

    if exp_id:
        run_dir = root / exp_id
        if run_dir.is_dir():
            latest = run_dir / "model_latest.pt"
            if latest.is_file():
                return latest
            picked = _pick_latest_checkpoint_by_mtime(_iter_numeric_model_pts(run_dir))
            if picked is not None:
                return picked
    if global_fallback:
        latest_candidates = list(root.rglob("model_latest.pt"))
        if latest_candidates:
            return _pick_latest_checkpoint_by_mtime(latest_candidates)
        all_pts = [
            p
            for p in root.rglob("model_*.pt")
            if _model_checkpoint_sort_key(p) >= 0
        ]
        if all_pts:
            return _pick_latest_checkpoint_by_mtime(all_pts)
    return None


def _find_latest_from_config(cfg: DictConfig) -> Path | None:
    data_dir = Path(cfg.train.get("data_dir", "runs"))
    exp_id = str(getattr(cfg.train, "exp_id", "") or "").strip()
    global_fb = bool(cfg.train.get("resume_latest_global", False))
    return find_latest_saved_model(data_dir, exp_id, global_fallback=global_fb)


def _find_latest_default_auto(cfg: DictConfig) -> Path | None:
    data_dir = Path(cfg.train.get("data_dir", "runs"))
    exp_id = str(getattr(cfg.train, "exp_id", "") or "").strip()
    found = find_latest_saved_model(data_dir, exp_id, global_fallback=False)
    if found is None:
        found = find_latest_saved_model(data_dir, exp_id, global_fallback=True)
    return found


def effective_resume_path(
    checkpoint_cli: Path | None,
    cfg: DictConfig,
    *,
    resume_latest_cli: bool = False,
    fresh_cli: bool = False,
) -> tuple[Path | None, bool, bool]:
    if fresh_cli or bool(cfg.train.get("resume_fresh", False)):
        return None, False, True
    if checkpoint_cli is not None:
        return checkpoint_cli.expanduser(), False, False
    rc = cfg.train.get("resume_checkpoint")
    rc_str = str(rc).strip() if rc is not None else ""
    rc_lower = rc_str.lower()
    if rc_lower in ("latest", "auto"):
        found = _find_latest_from_config(cfg)
        return found, True, False
    if rc is not None and rc_str not in ("", "~", "null", "None"):
        return Path(rc_str).expanduser(), False, False
    if resume_latest_cli or bool(cfg.train.get("resume_latest", False)):
        found = _find_latest_from_config(cfg)
        return found, True, False
    found = _find_latest_default_auto(cfg)
    return found, True, False


def log_resume_intent(
    resume_src: Path | None,
    resume_auto_attempted: bool,
    model_pt: Path | None,
    *,
    resume_fresh: bool,
) -> None:
    if resume_fresh:
        print("[resume] 처음부터 학습합니다 (--fresh 또는 train.resume_fresh).", flush=True)
        return
    if resume_auto_attempted and resume_src is None:
        print("[resume] runs 아래 model_*.pt 를 찾지 못했습니다. 처음부터 학습합니다.", flush=True)
        return
    if resume_src is not None:
        resolved = resume_src.expanduser().resolve()
        if resume_auto_attempted:
            print(f"[resume] 최신 체크포인트 자동 선택: {resolved}", flush=True)
        else:
            print(f"[resume] 체크포인트 요청 경로: {resolved}", flush=True)
        if model_pt is None:
            print(
                "[resume] 이전 정책을 찾을 수 없습니다. 처음부터 학습을 시작합니다.",
                flush=True,
            )
        else:
            print(
                f"[resume] 로드할 모델 파일: {model_pt.expanduser().resolve()}",
                flush=True,
            )


def _model_checkpoint_sort_key(p: Path) -> int:
    if not p.stem.startswith("model_"):
        return -1
    try:
        return int(p.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _iter_numeric_model_pts(dir_path: Path) -> list[Path]:
    return [
        p
        for p in dir_path.glob("model_*.pt")
        if _model_checkpoint_sort_key(p) >= 0
    ]


def _pick_latest_checkpoint_by_mtime(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: (p.stat().st_mtime, _model_checkpoint_sort_key(p)),
    )


def peek_resume_model_path(
    checkpoint_cli: Path | None,
    cfg: "DictConfig",
    *,
    fresh_cli: bool = False,
) -> Path | None:
    resume_src, _, resume_fresh = effective_resume_path(
        checkpoint_cli,
        cfg,
        resume_latest_cli=False,
        fresh_cli=fresh_cli,
    )
    if resume_fresh:
        return None
    model_pt, _ = resolve_resume_checkpoint(resume_src)
    return model_pt


def resolve_resume_checkpoint(path: Path | None) -> tuple[Path | None, Path | None]:
    if path is None:
        return None, None
    path = path.expanduser()
    if not path.exists():
        return None, None
    if path.is_file():
        if path.suffix != ".pt":
            return None, None
        trainer = path.parent / "trainer_state.pt"
        return path, trainer if trainer.exists() else None
    models = _iter_numeric_model_pts(path)
    if not models:
        return None, None
    model = _pick_latest_checkpoint_by_mtime(models)
    trainer = path / "trainer_state.pt"
    return model, trainer if trainer.exists() else None


def load_policy_checkpoint(policy: nn.Module, path: Path, device: str) -> None:
    resolved = path.expanduser().resolve()
    obj = torch.load(resolved, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        policy.load_state_dict(obj.state_dict(), strict=False)
    elif isinstance(obj, dict) and "state_dict" in obj:
        policy.load_state_dict(obj["state_dict"], strict=False)
    else:
        policy.load_state_dict(obj, strict=False)
