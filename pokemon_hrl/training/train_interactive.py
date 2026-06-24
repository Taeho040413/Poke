"""Train Interactive mode only (pure RL). LLM planner is not used during training."""

from __future__ import annotations

from pathlib import Path

import typer

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.paths import BASE_CONFIG_PATH, HRL_CONFIG_PATH
from pokemon_hrl.training.engine import run_interactive_training

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    config_path: Path = typer.Option(HRL_CONFIG_PATH),
    base_config_path: Path = typer.Option(BASE_CONFIG_PATH),
    scenario_index: int = typer.Option(0, help="Curriculum scenario index (goals only)"),
    headless: bool = typer.Option(
        False,
        "--headless/--no-headless",
        help="PyBoy 창 없이 실행 (기본: 게임 화면 표시)",
    ),
    timesteps: int | None = typer.Option(None, help="Override total_timesteps"),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint-path",
        "-c",
        help="체크포인트 디렉터리 또는 model_*.pt (미지정 시 exp_id 아래 최신 모델 자동 로드)",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="체크포인트 자동 로드를 끄고 처음부터 학습",
    ),
    track: bool | None = typer.Option(
        None,
        "--track/--no-track",
        help="Weights & Biases 로깅 (미지정 시 config train.track 사용)",
    ),
) -> None:
    config = load_hrl_config(config_path, base_config_path)
    config.env.headless = headless
    if track is not None:
        config.train.track = track

    mode = str(config.hrl.training.get("mode", "interactive"))
    if mode != "interactive":
        raise typer.BadParameter(f"Only interactive mode training is supported (got {mode!r})")

    result = run_interactive_training(
        config,
        scenario_index=scenario_index,
        timesteps=timesteps,
        checkpoint_path=checkpoint_path,
        fresh=fresh,
    )
    print(
        f"Training done exp={result['exp_id']} "
        f"steps={result['global_steps']} mean_reward={result['mean_reward']:.4f}"
    )


if __name__ == "__main__":
    app()
