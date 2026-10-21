import argparse

import numpy as np
import torch

from arm_rl import PixelReachEnv, PPOTrainer, load_policy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pixel PPO policy under visual shifts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="runs/adapted_pixel_arm.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = PixelReachEnv(
        randomize_target=True,
        randomize_light=True,
        seed=args.seed,
    )
    try:
        model = load_policy(args.checkpoint, device=args.device)
        trainer = PPOTrainer(env, model=model, device=args.device, learning_rate=1e-4)
        trainer.train(args.steps, args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
