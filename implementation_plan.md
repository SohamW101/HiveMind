# Implementation Plan: Custom CNN Feature Extractor

This document provides a detailed explanation and implementation plan for the **CNN stack** portion of the HiveMind multi-agent reinforcement learning (MARL) system. 

## Understanding the Problem: Why, Where, and What?

Before writing any code, it's crucial to understand the environment's state and why we need a Convolutional Neural Network (CNN).

### 1. The Need for a CNN (Why?)
The current PyBullet environment returns a **177-dimensional observation vector** for each robot (`obs_dim = 177`). This vector isn't just one type of data; it's a mix:
- **Indices 0-57:** Global state (robot pose, velocities, carton positions, depot direction).
- **Indices 57-129:** **72-ray planar LiDAR sweep** (distance to obstacles at 3.75-degree intervals over a 270-degree arc).
- **Indices 129-177:** Communication message slots.

If we feed this raw 177-dimensional vector into a standard Multi-Layer Perceptron (MLP) (the default in Stable-Baselines3), the neural network treats every LiDAR ray as completely independent of its neighbors. It doesn't inherently understand that ray 14 is spatially next to ray 15. 
**A 1D CNN solves this.** By sliding a small kernel over the 72 LiDAR rays, a 1D CNN can recognize structural shapes—like corners, flat walls, or gaps between shelves—regardless of exactly which angle (which ray) they appear at.

### 2. Where it fits in the architecture (Where?)
We are using Stable-Baselines3 (SB3) to train the policy (specifically PPO). SB3 allows us to replace the default neural network entry point with a **Custom Features Extractor**.
This extractor will sit right at the start of the neural network:
1. It receives the raw `177` observation vector.
2. It mathematically separates the vector into `lidar` (72 dims) and `state` (105 dims).
3. The `state` goes through a simple MLP.
4. The `lidar` goes through our 1D CNN stack.
5. The outputs are concatenated together and passed on to SB3's Actor and Critic networks to decide actions and estimate values.

### 3. What it needs (Inputs)
- **Input shape:** The input to the extractor will be `(batch_size, 177)`.
- We need to slice it in PyTorch: `lidar_obs = obs[:, 57:129]`.
- We must reshape the LiDAR data for a PyTorch `Conv1d` layer. PyTorch expects `(batch_size, channels, length)`, so we reshape `(batch_size, 72)` into `(batch_size, 1, 72)` (1 channel).

### 4. What it gives (Outputs)
- The CNN layers output a set of feature maps which are flattened.
- The concatenated result (CNN features + MLP state features) forms a single flat tensor (e.g., shape `(batch_size, 512)`).
- We tell SB3 that our `features_dim = 512`, and SB3 takes over from there.

---

## Proposed Implementation Plan

We will create a new file `hivemind_env/models.py` (which is mentioned in the `README.md` but currently doesn't exist) to house our custom extractor. We will also update `hivemind_env/training.py` or `train.py` (your teammate's integration part) to use this model.

### 1. `hivemind_env/models.py` (New File)
We will define `HiveMindExtractor(BaseFeaturesExtractor)`.

**Network Architecture Design:**
- **State branch (105 dims):**
  - Linear(105, 64) -> ReLU
- **LiDAR CNN branch (1 channel x 72 length):**
  - Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2) -> ReLU
  - Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2) -> ReLU
  - Flatten() -> Output is `32 * 17 = 544` dimensions.
- **Fusion:**
  - Concatenate State (64) + CNN (544) = 608 dimensions.
  - Linear(608, 256) -> ReLU -> Output `features_dim = 256`.

### 2. [NEW] [models.py](file:///home/het/HiveMind/hivemind_env/models.py)
We will write the PyTorch class that implements the forward pass.

## Open Questions

> [!IMPORTANT]
> **To the User:** 
> 1. Does the proposed 1D CNN structure (Conv1d layers) make sense to you? We can adjust the kernel sizes or channel counts, but a 2-layer CNN is usually a solid baseline for 72-ray LiDAR.
> 2. Are you ready for me to proceed with creating `hivemind_env/models.py` and writing the code? Let me know if you have any questions about the explanation above.
