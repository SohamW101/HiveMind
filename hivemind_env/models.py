import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that splits the Dict observation space.
    - Passes the 15x15x5 grid through a CNN.
    - Passes the `is_carrying` scalar through directly.
    - Concatenates the features into a single 1D tensor for the PPO MLP.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # We do not pass features_dim to super init immediately because we need to calculate it.
        # Actually, BaseFeaturesExtractor requires features_dim in super().__init__.
        super().__init__(observation_space, features_dim)

        # CNN for the grid (15x15x5)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 15x15 -> 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 7x7 -> 3x3
            nn.Flatten()
        )
        
        # Compute shape by doing one forward pass with dummy data
        grid_shape = observation_space.spaces["grid"].shape
        obs_size = grid_shape[0]
        channels = grid_shape[2]
        with torch.no_grad():
            dummy_grid = torch.zeros(1, channels, obs_size, obs_size)
            cnn_output_dim = self.cnn(dummy_grid).shape[1]
            
        # Total features = CNN features + 2 (is_carrying flag one-hot encoded by SB3)
        total_concat_dim = cnn_output_dim + 2
        
        # Final linear layer to project to requested `features_dim`
        self.linear = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # 1. Process the Grid
        grid = observations["grid"]
        # grid shape from env is (N, 15, 15, 5). PyTorch Conv2D expects (N, 5, 15, 15).
        # We permute the dimensions to match PyTorch format.
        if grid.shape[-1] == 5:
            grid = grid.permute(0, 3, 1, 2)
            
        cnn_features = self.cnn(grid)
        
        # 2. Process is_carrying flag
        is_carrying = observations["is_carrying"].float()
        # Flatten to ensure it's 2D (N, features) regardless of how SB3 one-hots it
        is_carrying = is_carrying.view(is_carrying.shape[0], -1)
            
        # 3. Concatenate
        combined = torch.cat((cnn_features, is_carrying), dim=1)
        
        # 4. Project
        return self.linear(combined)
