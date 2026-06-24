"""Optuna hyperparameter search for Interactive mode training."""

from __future__ import annotations

import pokemon_hrl  # noqa: F401 — bootstrap pokemonred_puffer import path

from pathlib import Path

import typer
from omegaconf import OmegaConf

from pokemon_hrl.config import clone_hrl_config, load_hrl_config
from pokemon_hrl.paths import BASE_CONFIG_PATH, HRL_CONFIG_PATH
from pokemon_hrl.training.engine import run_interactive_training

app = typer.Typer(pretty_exceptions_enable=False)


def _sample_param(trial, name: str, spec: dict):
    kind = str(spec.get("type", "float")).lower()
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown autotune param type: {kind}")


def _apply_trial_params(config, trial, param_specs: dict) -> None:
    for name, spec in param_specs.items():
        value = _sample_param(trial, name, spec)
        if name in ("learning_rate", "ent_coef", "gamma", "gae_lambda", "clip_coef", "vf_coef"):
            config.hrl.training[name] = value
            config.train[name] = value
        elif name in ("minibatch_size", "update_epochs", "batch_size", "bptt_horizon"):
            config.hrl.training[name] = value
        elif name == "num_envs":
            num_envs = int(value)
            config.hrl.training.num_envs = num_envs
            config.hrl.training.num_workers = num_envs
            config.hrl.training.env_batch_size = num_envs
            config.hrl.training.vectorization = "multiprocessing" if num_envs > 1 else "serial"
        elif name.startswith("reward."):
            reward_key = "hrl.rewards.interactive_mode.InteractiveModeRewardEnv"
            field = name.split(".", 1)[1]
            config.rewards[reward_key].reward[field] = value
        else:
            config.hrl.training[name] = value


@app.command()
def main(
    config_path: Path = typer.Option(HRL_CONFIG_PATH),
    base_config_path: Path = typer.Option(BASE_CONFIG_PATH),
    scenario_index: int = typer.Option(0),
    n_trials: int | None = typer.Option(None, help="Override hrl.autotune.n_trials"),
    timesteps: int | None = typer.Option(None, help="Override timesteps per trial"),
    study_name: str | None = typer.Option(None),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help="PyBoy 창 없이 실행 (서버 기본: headless)",
    ),
) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise typer.BadParameter("Install optuna: pip install -e '.[autotune]'") from exc

    config = load_hrl_config(config_path, base_config_path)
    config.env.headless = headless

    auto = config.hrl.autotune
    trials = int(n_trials or auto.get("n_trials", 20))
    steps_per_trial = int(timesteps or auto.get("timesteps_per_trial", 200_000))
    name = study_name or str(auto.get("study_name", "interactive-ppo"))
    storage = auto.get("storage")
    direction = str(auto.get("direction", "maximize"))
    param_specs = OmegaConf.to_container(auto.get("params", {}), resolve=True)

    if not param_specs:
        raise typer.BadParameter("hrl.autotune.params is empty in config")

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = clone_hrl_config(config)
        _apply_trial_params(trial_cfg, trial, param_specs)
        result = run_interactive_training(
            trial_cfg,
            scenario_index=scenario_index,
            timesteps=steps_per_trial,
            exp_suffix=f"trial{trial.number:04d}",
        )
        trial.set_user_attr("exp_id", result["exp_id"])
        trial.set_user_attr("global_steps", result["global_steps"])
        return float(result["mean_reward"])

    if storage:
        db_path = str(storage).replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
        )
    else:
        study = optuna.create_study(study_name=name, direction=direction)

    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    print("Best trial:", study.best_trial.number)
    print("Best mean_reward:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    app()
