"""
Validation test for HiveMindExtractor neural network architecture.
Run from repo root with:
    python -m hivemind_env.testCNN
"""
import sys
import torch
import gymnasium as gym

try:
    from hivemind_env.models import HiveMindExtractor, get_policy_kwargs
    from hivemind_env.env import OBS_DIM_V3
except ImportError:
    import models
    from models import HiveMindExtractor, get_policy_kwargs
    from env import OBS_DIM_V3


def test_extractor():
    print("Testing HiveMindExtractor...")
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM_V3,), dtype=float)
    features_dim = 256
    extractor = HiveMindExtractor(observation_space=obs_space, features_dim=features_dim)

    # 1. Forward pass test with batch_size = 4
    batch_size = 4
    dummy_obs = torch.randn(batch_size, OBS_DIM_V3)
    out = extractor(dummy_obs)
    assert out.shape == (batch_size, features_dim), (
        f"Expected output shape ({batch_size}, {features_dim}), got {out.shape}"
    )
    print(f"PASS Forward pass: input shape {dummy_obs.shape} -> output shape {out.shape}")

    # 2. Gradient / Backward pass test
    loss = out.sum()
    loss.backward()
    for name, param in extractor.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
    print("PASS Backward pass: all parameters have valid non-null gradients")

    # 3. Policy kwargs test
    kwargs = get_policy_kwargs(features_dim=features_dim)
    assert kwargs["features_extractor_class"] is HiveMindExtractor
    assert kwargs["features_extractor_kwargs"]["features_dim"] == features_dim
    print("PASS Policy kwargs configuration verified for SB3 PPO")

    # 4. Device portability test (CUDA if available)
    if torch.cuda.is_available():
        extractor = extractor.to("cuda")
        cuda_obs = dummy_obs.to("cuda")
        cuda_out = extractor(cuda_obs)
        assert cuda_out.is_cuda
        print("PASS CUDA forward pass successful")

    print("\nAll HiveMindExtractor architecture tests PASSED successfully!")


if __name__ == "__main__":
    test_extractor()
