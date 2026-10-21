import torch
from torch import nn


class PixelActorCritic(nn.Module):
    """CNN actor-critic for 84x84 RGB observations."""

    def __init__(self, action_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observation):
        features = self.encoder(observation)
        return self.policy(features), self.value(features).squeeze(-1)
