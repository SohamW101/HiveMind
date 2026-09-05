import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import env

class HiveMindExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the HiveMind multi-agent warehouse environment.
    
    The environment's observation is a 177-dimensional vector:
    - 72 dimensions are LiDAR rays.
    - 105 dimensions are state vectors (poses, carrying status, messages, etc.).
    
    This extractor splits the observation, processes the LiDAR sweep through a 1D CNN
    to extract spatial features, passes the remaining state through an MLP, and
    concatenates the results before feeding them to the policy and value networks.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # We assume the observation_space has shape (177,) when it reaches the extractor
        super().__init__(observation_space, features_dim)
        
        self.lidar_slice = env.OBS_SLICES["lidar"]
        
        # Calculate sizes
        obs_dim = observation_space.shape[0]
        self.lidar_dim = self.lidar_slice.stop - self.lidar_slice.start
        self.state_dim = obs_dim - self.lidar_dim
        
        # 1. State MLP (Processes the 105-dim non-LiDAR state)
        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU()
        )
        
        # 2. LiDAR 1D CNN (Processes the 72-dim LiDAR sweep)
        # Input shape expected by Conv1d: (batch_size, channels=1, length=72)
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened size of the CNN output
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, self.lidar_dim)
            cnn_out_dim = self.lidar_cnn(dummy_lidar).shape[1]
            
        # 3. Fusion Layer
        fusion_input_dim = 64 + cnn_out_dim
        
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom feature extractor.
        
        :param observations: Tensor of shape (batch_size, 177)
        :return: Extracted features tensor of shape (batch_size, features_dim)
        """
        # Slice the observations
        # state1: everything before lidar [0:57]
        # state2: everything after lidar [129:]
        state1 = observations[:, :self.lidar_slice.start]
        state2 = observations[:, self.lidar_slice.stop:]
        state = torch.cat([state1, state2], dim=1)
        
        lidar = observations[:, self.lidar_slice.start:self.lidar_slice.stop]
        
        # Add channel dimension for Conv1d: (batch_size, 72) -> (batch_size, 1, 72)
        lidar = lidar.unsqueeze(1)
        
        # Process branches
        state_features = self.state_net(state)
        lidar_features = self.lidar_cnn(lidar)
        
        # Concatenate and fuse
        combined = torch.cat([state_features, lidar_features], dim=1)
        extracted_features = self.fusion_net(combined)
        
        return extracted_features
