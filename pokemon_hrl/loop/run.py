"""Run closed-loop HRL with optional LLM planner, orchestrator, and on-policy training."""

from __future__ import annotations

from pathlib import Path

import typer

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.loop.trainer import run_hrl_loop
from pokemon_hrl.paths import BASE_CONFIG_PATH, HRL_CONFIG_PATH

app = typer.Typer(pretty_exceptions_enable=False)


def _parse_bool_option(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise typer.BadParameter(
        "--planner.enabled must be one of: true/false, 1/0, yes/no, on/off"
    )


@app.command()
def main(
    config_path: Path = typer.Option(HRL_CONFIG_PATH),
    base_config_path: Path = typer.Option(BASE_CONFIG_PATH),
    planner_enabled: str | None = typer.Option(None, "--planner.enabled"),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Policy checkpoint file or run directory (default: auto from config)",
    ),
    fresh: bool = typer.Option(
        False,
        help="Ignore saved policy checkpoints and train from scratch",
    ),
    headless: bool = typer.Option(
        False,
        help="Run PyBoy headless (no game window)",
    ),
    scenario_index: int = typer.Option(0, help="Curriculum scenario index"),
    max_steps: int = typer.Option(
        1_000_000,
        help="Total environment steps (PPO rollouts until this budget)",
    ),
) -> None:
    config = load_hrl_config(config_path, base_config_path)
    if planner_enabled is not None:
        config.hrl.planner.enabled = _parse_bool_option(planner_enabled)
    config.hrl.training.scenario_index = int(scenario_index)

    result = run_hrl_loop(
        config,
        scenario_index=scenario_index,
        max_steps=max_steps,
        checkpoint_path=checkpoint,
        fresh=fresh,
        headless=headless,
    )
    typer.echo(
        f"[loop] done exp={result['exp_id']} "
        f"steps={result['global_steps']} ppo_updates={result['epochs']}"
    )


if __name__ == "__main__":
    app()
