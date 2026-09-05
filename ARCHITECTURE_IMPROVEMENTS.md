# HiveMind Architecture & Incentive Analysis: Why the Bots Froze and How We Fixed It

This document outlines the diagnosis of our MARL policy's early failure modes and details the architectural improvements implemented on the `multi-agent-v2` branch.

---

## 1. The Core Issue: Why the Robots Were "Frozen in Fear"

When we looked at the training curves, the bots were barely moving—either standing completely still or spinning in place, even on a single carton. This wasn't bad exploration luck; it was simple arithmetic that the policy learned to exploit:

1. **The Collision Risk Tax**:
   - Every collision costs **$-5.0$**, weighted at $90\%$ shared reward = **$-4.50$ deducted from all four robots simultaneously**.
   - Standing still (`stay`) only costs a tiny time penalty of **$-0.047$** per step.
   - When a robot is exploring randomly in narrow $1$-meter aisles, moving forward has roughly a $15\%\text{--}25\%$ chance of bumping a post, wall, or teammate.
   - Mathematically, taking a step had an expected return of **$-0.95$ per step**, while standing still had an expected return of **$-0.047$ per step**.
   - Because `stay`, `turnL`, and `turnR` never collide and are never invalid, PPO quickly realized that **moving forward is a death sentence**. Freezing became the optimal local policy.

2. **The "1-Carton Swarm" Pathology**:
   - Training at 1 carton was supposed to make learning easier. Instead, it created an unintended traffic jam:
     - All 4 robots received a potential shaping gradient pointing to the **exact same carton**.
     - All 4 rushed into the same 1-meter corridor, collided, and got penalized.
     - When one robot finally picked up the carton, the other 3 robots' fallback potential pointed to the *held* carton—meaning they literally **chased and blockaded the carrier on its way to the depot**.
   - Under single-policy parameter sharing, the 3 redundant robots doing nothing but bumping into things generated $75\%$ of the gradient updates, telling the shared network: *"Whatever you do, do not move."*

---

## 2. The Neural Network Chokepoint in the Previous Code

The initial draft in `hivemind_env/models.py` had two major issues:

1. **The 105-dim Information Chokepoint**:
   - The environment produces a 177-dimensional observation: 72 LiDAR beams and 105 state dimensions.
   - The previous draft tried to process all 105 non-LiDAR numbers through a **single linear layer**:
     ```python
     # Old implementation
     self.state_net = nn.Sequential(nn.Linear(105, 64), nn.ReLU())
     ```
   - Those 105 numbers represent vastly different concepts:
     - 4 robot poses & velocities ($18$ numbers)
     - 12 carton statuses & 24 carton coordinates ($36$ numbers)
     - 48 discrete radio message tokens from 3 teammates ($48$ numbers)
     - Depot direction and clock ($3$ numbers)
   - Forcing coordinates, carrying flags, and communication tokens through one linear layer created an extreme bottleneck. The network couldn't compute spatial relationships (like *"where is carton 3 relative to me?"*) or decode teammate messages.

2. **LiDAR Distance Ambiguity**:
   - The LiDAR sweep gives distances, but not semantic tags (a shelf post, outer wall, or robot body at 0.5m look identical). A shallow 2-layer CNN struggled to resolve corridor openings from obstacles.

3. **Import Bugs**:
   - `models.py` had `import env`, which caused an immediate `ModuleNotFoundError` when imported from the repository root (e.g. by `train.py`).
   - `testCNN.py` had `import models`, which collided with the local `models/` directory containing checkpoints.

---

## 3. The Improved Architecture

We overhauled `hivemind_env/models.py` into a **Modular Semantic Feature Extractor** while keeping the 177-dimensional observation contract $100\%$ intact:

### A. Semantic State Decomposition
Instead of one generic linear layer, we divide the 105-dim state into dedicated sub-networks:
- **Self Encoder (9 dims $\rightarrow$ 64 dims)**: Processes own pose, linear/angular velocity, carrying flag, relative depot direction, and normalized episode clock through a 2-layer MLP with LayerNorm and GELU activations.
- **Teammates Encoder (12 dims $\rightarrow$ 64 dims)**: Processes other robots' poses and carrying statuses so the agent can learn collision avoidance and spacing.
- **Carton/Objective Encoder (36 dims $\rightarrow$ 64 dims)**: Correlates the 12 carton statuses (available/claimed/done) directly with their 2D world coordinates.
- **Comms Encoder (48 dims $\rightarrow$ 32 dims)**: A dedicated 2-layer network to interpret the broadcast tokens received from the other three agents.

### B. Deepened 1D-CNN LiDAR Perception
- 3-stage 1D convolutional pipeline over the 72 LiDAR rays (`Conv1d(1 $\rightarrow$ 32) $\rightarrow$ Conv1d(32 $\rightarrow$ 64) $\rightarrow$ Conv1d(64 $\rightarrow$ 64)`).
- Uses GELU activations and projects through a 128-neuron dense layer with LayerNorm to cleanly recognize walls, aisle openings, and tight corners.

### C. Multi-Layer Feature Fusion
- All five representations are concatenated ($64 + 64 + 64 + 32 + 128 = 352\text{ dims}$) and passed through a 2-layer fusion network with LayerNorm.
- Outputs a normalized **$256$-dimensional feature vector** (`features_dim = 256`), ready for Stable-Baselines3 actor-critic heads.

### D. Code Quality & Integration
- Fixed all module imports (`from . import env` with fallback).
- Added `get_policy_kwargs()` to easily hook into SB3 PPO and MaskablePPO.
- Fully verified with `python -m hivemind_env.testCNN` (tested forward pass, backward gradient flow, and CUDA portability).

---

## 4. Next Steps for the Team

1. **Training Pipeline**:
   - Wire `HiveMindExtractor` into a streamlined `train.py` using `get_policy_kwargs()`.
2. **Reward Balance**:
   - Ensure `shaping_scale` stays at $\ge 60.0$ so the positive gradient for approaching a carton comfortably outweighs the collision risk, giving the robots confidence to move.
3. **Multi-Agent Evaluation**:
   - Track `rollout/delivered_fraction` alongside `success_rate` during training so we know when the bots are actively clearing the warehouse.
