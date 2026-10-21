import argparse

import numpy as np
import torch

from arm_rl import PixelReachEnv, PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a pixel PPO arm reacher.")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/ppo_pixel_arm.pt")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = PixelReachEnv(render=args.gui, seed=args.seed)
    try:
        trainer = PPOTrainer(env, device=args.device)
        trainer.train(args.steps, args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
