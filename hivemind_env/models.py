"""Custom SB3 feature extractor for the implementation-plan observation."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


OBS_DIM = 177
LIDAR_START = 57
LIDAR_END = 129
LIDAR_DIM = LIDAR_END - LIDAR_START
STATE_DIM = OBS_DIM - LIDAR_DIM
FEATURES_DIM = 256


class HiveMindExtractor(BaseFeaturesExtractor):
    """Encode the 105-value state branch and 72-ray LiDAR branch separately."""

    def __init__(self, observation_space):
        if observation_space.shape != (OBS_DIM,):
            raise ValueError(
                "HiveMindExtractor expects one robot observation with shape "
                f"({OBS_DIM},), got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=FEATURES_DIM)

        self.state_branch = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
        )
        self.lidar_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Conv1d(72, k=5, s=2) followed by Conv1d(k=3, s=2) produces 16
        # positions, so the exact concatenation width is 64 + 32 * 16.
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 * 16, FEATURES_DIM),
            nn.ReLU(),
        )

    def forward(self, observations):
        observations = observations.float()
        if observations.ndim != 2 or observations.shape[-1] != OBS_DIM:
            raise ValueError(
                f"HiveMindExtractor expects a 2D tensor ending in {OBS_DIM}; "
                f"got {tuple(observations.shape)}"
            )
        lidar = observations[:, LIDAR_START:LIDAR_END].unsqueeze(1)
        state = torch.cat(
            (observations[:, :LIDAR_START], observations[:, LIDAR_END:]), dim=1
        )
        return self.fusion(torch.cat((self.state_branch(state), self.lidar_branch(lidar)), dim=1))
