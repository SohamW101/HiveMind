"""
Neural network policy models and custom feature extractors for HiveMind MARL.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from . import env
except (ImportError, ValueError):
    import env


class HiveMindExtractor(BaseFeaturesExtractor):
    """
    Modular, semantic feature extractor for the HiveMind multi-agent warehouse environment.
    
    Observation Layout (177 floats per agent):
      - Self State (9 dims):
          * own_pose (3): [0:3]
          * own_velocity (2): [3:5]
          * own_carrying (1): [5:6]
          * depot_direction (2): [54:56]
          * elapsed_time (1): [56:57]
      - Teammates State (12 dims):
          * other_poses (9): [6:15]
          * other_carrying (3): [15:18]
      - Cartons State (36 dims):
          * carton_status (12): [18:30]
          * carton_positions (24): [30:54]
      - Planar LiDAR Sweep (72 dims):
          * lidar rays (72): [57:129]
      - Communications Buffer (48 dims):
          * 3 other agents x 16 one-hot tokens: [129:177]

    Architecture:
      1. 1D CNN Branch: Processes 72 LiDAR returns to identify walls, shelves, and corridor geometries.
      2. Self Encoder: Processes own kinematics, carrying status, and depot orientation.
      3. Teammates Encoder: Disentangles teammate poses and carrying statuses for collision avoidance.
      4. Cartons Encoder: Relates carton statuses (available/claimed/done) and coordinates.
      5. Comms Encoder: Decodes 48-dim message buffer received from teammates.
      6. Fusion Network: Multi-layer fusion projecting all semantic representations to features_dim (256).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Slice references from environment definition
        self.slices = env.OBS_SLICES
        obs_dim = observation_space.shape[0]
        if obs_dim != env.OBS_DIM_V3:
            raise ValueError(
                f"HiveMindExtractor expects observation space with dimension {env.OBS_DIM_V3}, "
                f"got {obs_dim}."
            )

        # 1. Self State Branch (9 dims)
        self.self_dim = 3 + 2 + 1 + 2 + 1  # 9
        self.self_net = nn.Sequential(
            nn.Linear(self.self_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 2. Teammates Branch (12 dims)
        self.teammates_dim = 9 + 3  # 12
        self.teammates_net = nn.Sequential(
            nn.Linear(self.teammates_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 3. Cartons Branch (36 dims)
        self.cartons_dim = 12 + 24  # 36
        self.cartons_net = nn.Sequential(
            nn.Linear(self.cartons_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 4. Communications Branch (48 dims)
        self.comms_dim = 48
        self.comms_net = nn.Sequential(
            nn.Linear(self.comms_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

        # 5. Planar LiDAR 1D-CNN Branch (72 dims)
        self.lidar_dim = 72
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),  # -> 36
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # -> 18
            nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # -> 18
            nn.GELU(),
            nn.Flatten(),
        )

        # Pre-compute flattened CNN dimension
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, self.lidar_dim)
            cnn_flat_dim = self.lidar_cnn(dummy_lidar).shape[1]  # 64 * 18 = 1152

        self.lidar_proj = nn.Sequential(
            nn.Linear(cnn_flat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 6. Fusion Network
        # 64 (self) + 64 (teammates) + 64 (cartons) + 32 (comms) + 128 (lidar) = 352
        fusion_in_dim = 64 + 64 + 64 + 32 + 128
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass decomposing observation vector into semantic streams.

        :param observations: Tensor of shape (batch_size, 177)
        :return: Fused feature representations of shape (batch_size, features_dim)
        """
        # Extract Self State: [own_pose (0:3), own_velocity (3:5), own_carrying (5:6),
        #                      depot_direction (54:56), elapsed_time (56:57)]
        self_state = torch.cat([
            observations[:, self.slices["own_pose"]],
            observations[:, self.slices["own_velocity"]],
            observations[:, self.slices["own_carrying"]],
            observations[:, self.slices["depot_direction"]],
            observations[:, self.slices["elapsed_time"]],
        ], dim=1)

        # Extract Teammates State: [other_poses (6:15), other_carrying (15:18)]
        teammates_state = torch.cat([
            observations[:, self.slices["other_poses"]],
            observations[:, self.slices["other_carrying"]],
        ], dim=1)

        # Extract Cartons State: [carton_status (18:30), carton_positions (30:54)]
        cartons_state = torch.cat([
            observations[:, self.slices["carton_status"]],
            observations[:, self.slices["carton_positions"]],
        ], dim=1)

        # Extract Comms State: [messages (129:177)]
        comms_state = observations[:, self.slices["messages"]]

        # Extract LiDAR Sweep: [lidar (57:129)] -> shape (B, 1, 72)
        lidar_data = observations[:, self.slices["lidar"]].unsqueeze(1)

        # Compute branch embeddings
        self_features = self.self_net(self_state)
        teammates_features = self.teammates_net(teammates_state)
        cartons_features = self.cartons_net(cartons_state)
        comms_features = self.comms_net(comms_state)

        lidar_conv = self.lidar_cnn(lidar_data)
        lidar_features = self.lidar_proj(lidar_conv)

        # Combine all features and fuse
        fused_input = torch.cat([
            self_features,
            teammates_features,
            cartons_features,
            comms_features,
            lidar_features,
        ], dim=1)

        return self.fusion_net(fused_input)


def get_policy_kwargs(features_dim: int = 256, net_arch: list | dict | None = None) -> dict:
    """
    Helper returning standard policy_kwargs for Stable-Baselines3 PPO or MaskablePPO.
    """
    if net_arch is None:
        net_arch = dict(pi=[128, 64], vf=[128, 64])

    return dict(
        features_extractor_class=HiveMindExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=net_arch,
    )
