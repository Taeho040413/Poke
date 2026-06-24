import argparse
import os
from itertools import islice

import mediapy
from omegaconf import OmegaConf
from tqdm import tqdm

from pokemon_hrl.config import load_hrl_config
from pokemon_hrl.pokemonred_puffer.rewards.baseline import ExplorationInteractionRewardEnv

BASELINE_REWARD_KEY = "pokemonred_puffer.rewards.baseline.ExplorationInteractionRewardEnv"
INTERACTIVE_REWARD_KEY = "hrl.rewards.interactive_mode.InteractiveModeRewardEnv"


def _resolve_reward_cfg(config):
    rewards = config.get("rewards", {})
    if BASELINE_REWARD_KEY in rewards:
        return rewards[BASELINE_REWARD_KEY].reward
    if INTERACTIVE_REWARD_KEY in rewards:
        return rewards[INTERACTIVE_REWARD_KEY].reward
    return OmegaConf.create({})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rom_path")
    parser.add_argument("actions_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of steps to render.")
    parser.add_argument("--actions-file", default="", help="Only render this file in actions_dir")
    args = parser.parse_args()

    # steps is a list of lists where the index maps to the step count
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_hrl_config()
    env_config = OmegaConf.create(OmegaConf.to_container(config.env, resolve=True))
    env_config.gb_path = args.rom_path
    env_config.log_frequency = None
    reward_cfg = OmegaConf.create(
        OmegaConf.to_container(_resolve_reward_cfg(config), resolve=True)
    )
    for path in os.listdir(str(args.actions_dir)):
        if args.actions_file and args.actions_file != path:
            continue
        if not path.endswith("actions.csv"):
            continue
        env_id, _ = path.split("-")
        # The config must match what was used for training
        env = ExplorationInteractionRewardEnv(env_config, reward_cfg)
        with (
            open(os.path.join(args.actions_dir, path)) as f,
            mediapy.VideoWriter(
                os.path.join(args.output_dir, f"actions-{env_id}.mp4"), (144, 160), fps=60
            ) as writer,
        ):
            env.reset()
            writer.add_image(env.render()[:, :])
            # Read lines so we can get an estimate of the line count
            actions = f.readlines()
            for action in tqdm(islice(actions, args.n_steps), total=args.n_steps or len(actions)):
                env.step(int(action.strip()))
                writer.add_image(env.render()[:, :])
        env.close()
    if hasattr(os, "sync"):
        os.sync()


if __name__ == "__main__":
    main()
