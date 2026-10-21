from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from .models import PixelActorCritic


def _squashed_log_prob(distribution, latent):
    action = torch.tanh(latent)
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    return (distribution.log_prob(latent) - correction).sum(dim=-1)


class PPOTrainer:
    def __init__(self, env, model=None, device="cpu", learning_rate=3e-4):
        self.env = env
        self.device = torch.device(device)
        self.model = (model or PixelActorCritic()).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _act(self, observation, deterministic=False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, value = self.model(obs)
            std = self.model.log_std.exp().expand_as(mu)
            distribution = Normal(mu, std)
            latent = mu if deterministic else distribution.sample()
            action = torch.tanh(latent)
            log_prob = _squashed_log_prob(distribution, latent)
        return (
            action.squeeze(0).cpu().numpy(),
            latent.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def _update(self, observations, latents, old_log_probs, returns, advantages,
                epochs=4, batch_size=64, clip_ratio=0.2):
        obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        latent = torch.as_tensor(np.asarray(latents), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        count = obs.shape[0]
        for _ in range(epochs):
            indices = np.random.permutation(count)
            for start in range(0, count, batch_size):
                batch = indices[start:start + batch_size]
                mu, value = self.model(obs[batch])
                distribution = Normal(mu, self.model.log_std.exp().expand_as(mu))
                log_prob = _squashed_log_prob(distribution, latent[batch])
                ratio = torch.exp(log_prob - old_logp[batch])
                unclipped = ratio * adv[batch]
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (ret[batch] - value).pow(2).mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = policy_loss + value_loss - 0.01 * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def train(
        self,
        total_steps,
        checkpoint_path,
        rollout_steps=512,
        update_epochs=4,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        observation = self.env.reset()
        steps_done = 0
        episode_return = 0.0
        episode_count = 0
        success_window = []
        while steps_done < total_steps:
            observations, latents = [], []
            rewards, dones, values, log_probs = [], [], [], []
            count = min(rollout_steps, total_steps - steps_done)
            for _ in range(count):
                action, latent, log_prob, value = self._act(observation)
                next_observation, reward, done, info = self.env.step(action)
                observations.append(observation)
                latents.append(latent)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                log_probs.append(log_prob)
                episode_return += reward
                steps_done += 1
                if done:
                    episode_count += 1
                    success_window.append(float(info["success"]))
                    if episode_count % 10 == 0:
                        rate = float(np.mean(success_window[-50:])) if success_window else 0.0
                        print(
                            f"steps={steps_done} episodes={episode_count} "
                            f"success_rate_50={rate:.2f} return={episode_return:.2f}"
                        )
                    episode_return = 0.0
                    observation = self.env.reset()
                else:
                    observation = next_observation

            with torch.no_grad():
                if dones[-1]:
                    next_value = 0.0
                else:
                    _, value_tensor = self.model(
                        torch.as_tensor(
                            observation, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                    )
                    next_value = float(value_tensor.item())

            advantages = np.zeros(len(rewards), dtype=np.float32)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                nonterminal = 1.0 - float(dones[t])
                following_value = next_value if t == len(rewards) - 1 else values[t + 1]
                delta = rewards[t] + gamma * following_value * nonterminal - values[t]
                gae = delta + gamma * gae_lambda * nonterminal * gae
                advantages[t] = gae
            returns = advantages + np.asarray(values, dtype=np.float32)
            self._update(
                observations,
                latents,
                log_probs,
                returns,
                advantages,
                epochs=update_epochs,
                batch_size=batch_size,
            )
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": self.model.state_dict(), "image_size": self.env.image_size},
                path,
            )
        print(f"training finished: {steps_done} steps, {episode_count} episodes")
        return self.model


def load_policy(checkpoint_path, device="cpu"):
    payload = torch.load(checkpoint_path, map_location=device)
    model = PixelActorCritic()
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model
