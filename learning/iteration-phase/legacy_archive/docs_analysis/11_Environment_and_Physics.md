# 11_Environment_and_Physics

This document details the PyBullet physics environment design, reward functions, and action spaces that power the HiveMind simulation.

## 1. Physics Engine: PyBullet
We utilized `pybullet` for our rigid-body physics simulation. It provides fast, headless rendering capabilities (via `rgb_array`) for server-side training, while supporting live GUI rendering (`human` mode) for local inference and debugging.

## 2. Action and Observation Spaces

### A. Continuous Action Space
The environment uses a continuous, multi-dimensional vector action space for each agent.
*   Instead of discrete actions (up, down, left, right), agents output continuous forces/velocities (e.g., `[-1.0 to 1.0]`).
*   This allows for smooth, physically realistic movements, acceleration, and precise steering around dynamic obstacles.

### B. Observation Space (LiDAR & State)
Each agent's observation vector consists of:
1.  **Raycasts (LiDAR):** 16 rays cast radially around the agent to detect walls, obstacles, and other agents.
2.  **Kinematic State:** The agent's current X/Y velocity and absolute position.
3.  **Relative Goal State:** The vector pointing towards the agent's currently assigned carton or drop-off zone.

## 3. Reward Engineering: Dual-Scale PBRS

The most critical factor in achieving convergence was our implementation of Potential Based Reward Shaping (PBRS). 

Sparse rewards (e.g., +100 only upon successful delivery) fail in complex environments because the chance of a random walk successfully completing the task is nearly zero.

**Dual-Scale PBRS:**
1.  **Macro-PBRS (Global Task):** Rewards the agent for decreasing the absolute distance between itself and the target carton, and subsequently, between the carton and the drop-off zone.
2.  **Micro-PBRS (Obstacle Avoidance):** Penalizes the agent for getting too close to dynamic obstacles or other agents (based on LiDAR thresholding).

By summing these potentials, the agent receives a dense, continuous reward signal at every step, guiding it smoothly toward the goal while naturally repelling it from collisions.
