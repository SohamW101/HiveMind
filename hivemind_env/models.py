"""
Feature extractor for the HiveMind observation: three branches over one flat vector.

    world    [0:57]     poses, velocities, carrying flags, carton status and positions,
                        depot direction, elapsed time         -> MLP
    lidar    [57:129]   72 range returns, angularly ordered   -> 1-D CNN
    messages [129:177]  3 speakers x 16 one-hot tokens        -> MLP

Slices are read from OBS_SLICES, never from literals, so a layout change moves the data
and the network together. Every dimension is derived from the observation space handed
in - the CNN flatten width in particular, which is baked into the saved weights.

WHY NOT THE DEFAULT MlpPolicy

It would accept the flat Box directly and treat all 177 numbers as an unordered bag,
throwing away the one piece of real structure: the LiDAR returns are a *sequence*. Rays
k and k+1 point 3.75 degrees apart, so a wall spans a run of adjacent slots. A 1-D
convolution learns "obstacle to my left" once and applies it at every bearing; a dense
layer has to learn it separately for all 72 inputs.

THE MESSAGE BRANCH IS DELIBERATELY UNCHANGED BY STEP 7

It was built to run on all-zero input while the slots were reserved, and step 7 filled
them without touching a layer here. Two things depend on that: the silent and
communicating runs share one architecture, so a difference between them is the channel
and not capacity; and every pre-step-7 checkpoint still loads, because msg_net is
Linear(48, 64) in those files already.

The tempting change is to embed each speaker's 16-token block with ONE shared layer
rather than a separate 48 -> 64 map - speaker-symmetric, 3x fewer parameters to learn,
very likely better. It also reshapes the state dict and invalidates every checkpoint, so
it is a deliberate V2 of this extractor with its own class. Do not make it here.

With one-hot tokens msg_net's first Linear is an embedding table read three times:
column (16b + k) is the vector for "speaker b said token k", so the learned protocol is
inspectable in those weights and two synonymous tokens have near-identical columns.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from hivemind_env.env import OBS_SLICES


class HiveMindExtractor(BaseFeaturesExtractor):
    """
    Three-branch extractor over the flat observation. Emits `features_dim` features.

    `lidar_channels` and `hidden` are exposed because they are the two knobs worth
    turning if step 6 underfits; everything else follows from the observation space.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 lidar_channels: int = 32, hidden: int = 128):
        super().__init__(observation_space, features_dim)

        obs_dim = int(observation_space.shape[0])
        expected = OBS_SLICES["messages"].stop
        if obs_dim != expected:
            raise ValueError(
                f"Observation is {obs_dim} wide but the layout in env.py describes "
                f"{expected}. The extractor reads its slices from OBS_SLICES, so these "
                f"must agree - see the observation layout block at the top of env.py."
            )

        # Slices are stored as plain ints so the module can be pickled and reloaded
        # without carrying a reference to the env module's globals.
        # The world block runs from the start of the vector to the end of elapsed_time,
        # i.e. everything before the LiDAR sweep.
        self.world_slice = (0, OBS_SLICES["elapsed_time"].stop)
        self.lidar_slice = (OBS_SLICES["lidar"].start, OBS_SLICES["lidar"].stop)
        self.msg_slice = (OBS_SLICES["messages"].start, OBS_SLICES["messages"].stop)

        world_dim = self.world_slice[1] - self.world_slice[0]
        n_rays = self.lidar_slice[1] - self.lidar_slice[0]
        msg_dim = self.msg_slice[1] - self.msg_slice[0]

        self.world_net = nn.Sequential(
            nn.Linear(world_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # Stride-2 convolutions rather than pooling: the sweep is short enough that
        # halving twice is plenty, and stride keeps the angular ordering intact.
        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, lidar_channels, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(lidar_channels, lidar_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            lidar_out = self.lidar_net(torch.zeros(1, 1, n_rays)).shape[1]

        self.msg_net = nn.Sequential(nn.Linear(msg_dim, 64), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(hidden + lidar_out + 64, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w0, w1 = self.world_slice
        l0, l1 = self.lidar_slice
        m0, m1 = self.msg_slice

        world = self.world_net(observations[:, w0:w1])
        # Conv1d wants (N, channels, length); the sweep is a single channel.
        lidar = self.lidar_net(observations[:, l0:l1].unsqueeze(1))
        msg = self.msg_net(observations[:, m0:m1])

        return self.head(torch.cat((world, lidar, msg), dim=1))


# Default policy_kwargs for PPO. Kept here so train.py and any evaluation script agree
# on the architecture without restating it - a mismatch loads as a shape error.
DEFAULT_POLICY_KWARGS = dict(
    features_extractor_class=HiveMindExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    # Actor and critic each get their own small head on top of the shared features.
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
)
