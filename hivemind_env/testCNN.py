import torch
import gymnasium as gym
# from models import HiveMindExtractor
import models

# 177 is the default obs dim
obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(177,), dtype=float)
extractor = models.HiveMindExtractor(observation_space=obs_space)

dummy_obs = torch.randn(4, 177)
out = extractor(dummy_obs)
print("Output shape:", out.shape)
