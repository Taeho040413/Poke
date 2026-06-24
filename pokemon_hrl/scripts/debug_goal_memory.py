"""Smoke-test goal memory observations and info fields."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.env.unwrap import unwrap_hrl_env
from pokemon_hrl.training.env_factory import make_interactive_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to HRL config (defaults to package hrl_config.yaml)",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable-goal-memory", action="store_true")
    args = parser.parse_args()

    config = load_hrl_config() if args.config is None else OmegaConf.load(args.config)
    if args.disable_goal_memory:
        config.hrl.goal_memory.enabled = False

    env = make_interactive_env(config, puffer_wrapper=False)
    obs, info = env.reset(seed=args.seed)
    base = unwrap_hrl_env(env)
    if hasattr(base, "set_goal_context"):
        base.set_goal_context(
            {
                "target_map_id": 3,
                "target_x": 12,
                "target_y": 8,
                "target_event_id": "EVENT_GOT_PARCEL",
                "goal_key": "debug_goal",
            }
        )
        obs, _ = env.reset(seed=args.seed)

    print("=== observation keys / shapes ===")
    if isinstance(obs, dict):
        for key, value in sorted(obs.items()):
            shape = getattr(value, "shape", None)
            print(f"  {key}: shape={shape}")
    else:
        print(f"  obs type: {type(obs)}")

    goal_channels = [k for k in obs if k.startswith("goal_memory_")] if isinstance(obs, dict) else []
    print(f"\n=== goal memory channels present: {bool(goal_channels)} ===")
    for key in goal_channels:
        print(f"  {key}: {obs[key].shape}")

    print("\n=== stepping ===")
    for step in range(args.steps):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        print(
            f"step={step} reward={reward:.4f} "
            f"goal_key={info.get('goal_key')} "
            f"on_target_map={info.get('on_target_map')} "
            f"goal_dx={info.get('goal_dx')} goal_dy={info.get('goal_dy')} "
            f"blocked_local_count={info.get('blocked_local_count')} "
            f"visited_local_count={info.get('visited_local_count')} "
            f"reward_goal_map={info.get('reward_goal_map')} "
            f"reward_goal_event={info.get('reward_goal_event')} "
            f"reward_goal_interaction={info.get('reward_goal_interaction')}"
        )
        if term or trunc:
            break

    env.close()


if __name__ == "__main__":
    main()
