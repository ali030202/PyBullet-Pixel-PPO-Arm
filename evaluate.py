import argparse

import numpy as np
import torch

from arm_rl import PixelReachEnv, load_policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pixel PPO reaching policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = PixelReachEnv(
        render=args.gui,
        randomize_target=args.randomize,
        randomize_light=args.randomize,
    )
    model = load_policy(args.checkpoint, device=args.device)
    device = torch.device(args.device)
    successes, distances = [], []
    try:
        for episode in range(args.episodes):
            observation = env.reset(seed=episode)
            done = False
            info = {"distance": float("inf"), "success": False}
            while not done:
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    mu, _ = model(tensor)
                    action = torch.tanh(mu).squeeze(0).cpu().numpy()
                observation, _, done, info = env.step(action)
            successes.append(float(info["success"]))
            distances.append(float(info["distance"]))
        print(f"success_rate={np.mean(successes):.3f}")
        print(f"mean_final_distance={np.mean(distances):.4f} m")
    finally:
        env.close()


if __name__ == "__main__":
    main()
