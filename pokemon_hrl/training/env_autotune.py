"""Benchmark PyBoy env throughput and recommend parallel env settings."""

from __future__ import annotations

import pokemon_hrl  # noqa: F401 — bootstrap pokemonred_puffer import path

import json
from pathlib import Path

import typer
from omegaconf import OmegaConf

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.paths import BASE_CONFIG_PATH, HRL_CONFIG_PATH, project_root
from pokemon_hrl.training.env_factory import bootstrap_shared_planner, make_interactive_env
from pokemon_hrl.training.shared_plan import get_shared_plan_store

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    config_path: Path = typer.Option(HRL_CONFIG_PATH),
    base_config_path: Path = typer.Option(BASE_CONFIG_PATH),
    scenario_index: int = typer.Option(0),
    batch_size: int | None = typer.Option(
        None,
        help="Starting env batch size (default: hrl.env_autotune.batch_size)",
    ),
    max_envs: int | None = typer.Option(None),
    max_env_ram_gb: float | None = typer.Option(None),
    time_per_test: int | None = typer.Option(None),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help="PyBoy 창 없이 실행 (서버 기본: headless)",
    ),
    write_config: bool = typer.Option(
        True,
        "--write-config/--no-write-config",
        help="추천 설정을 runs/autotune_env.json에 저장",
    ),
) -> None:
    try:
        import pufferlib.vector
    except ImportError as exc:
        raise typer.BadParameter(
            "Install project dependencies first: pip install -e '.[dev]'"
        ) from exc

    config = load_hrl_config(config_path, base_config_path)
    config.env.headless = headless

    auto = OmegaConf.to_container(config.hrl.get("env_autotune", {}), resolve=True) or {}
    start_batch = int(batch_size or auto.get("batch_size", 1))
    env_cap = int(max_envs or auto.get("max_envs", 16))
    ram_cap = float(max_env_ram_gb or auto.get("max_env_ram_gb", 16))
    test_secs = int(time_per_test or auto.get("time_per_test", 5))

    shared_plan = get_shared_plan_store()
    bootstrap_shared_planner(config, scenario_index=scenario_index, shared_plan=shared_plan)

    def env_creator():
        return make_interactive_env(
            config,
            scenario_index=scenario_index,
            puffer_wrapper=True,
            shared_plan=shared_plan,
        )

    print(
        f"[env-autotune] batch_size={start_batch} max_envs={env_cap} "
        f"max_env_ram_gb={ram_cap} time_per_test={test_secs}s headless={headless}",
        flush=True,
    )
    pufferlib.vector.autotune(
        env_creator,
        batch_size=start_batch,
        max_envs=env_cap,
        max_env_ram_gb=ram_cap,
        time_per_test=test_secs,
    )

    import psutil

    cores = psutil.cpu_count(logical=False) or 1
    recommended = {
        "vectorization": "multiprocessing",
        "num_envs": min(env_cap, cores),
        "num_workers": min(env_cap, cores),
        "env_batch_size": min(env_cap, cores),
        "note": "위 autotune 출력에서 SPS가 가장 높은 설정을 hrl.training에 반영하세요.",
    }
    print("[env-autotune] suggested starting point:", json.dumps(recommended, indent=2), flush=True)

    if write_config:
        out_path = project_root() / "runs" / "autotune_env.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(recommended, indent=2) + "\n", encoding="utf-8")
        print(f"[env-autotune] wrote {out_path}", flush=True)


if __name__ == "__main__":
    app()
