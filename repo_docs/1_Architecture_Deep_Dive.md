# HiveMind Architecture Deep Dive

## 1. System Overview
HiveMind is a Multi-Agent Reinforcement Learning (MARL) framework designed to solve cooperative warehouse logistics tasks using continuous spatial simulation (PyBullet) and recurrent neural networks (RecurrentPPO).

The core philosophy of the architecture is **Centralized Training with Shared Policy Execution**. Instead of 4 independent networks, a single overarching neural policy controls all 4 autonomous agents simultaneously. 

## 2. The Multi-Agent Environment Wrapper (`hivemind_env`)
The environment is built on top of standard `gym.Env`, but adapted via a vectorization wrapper (`HiveMindSharedPolicyVecEnv`). 

### State Vectorization (N_AGENTS slots)
To enable the Shared Policy, the wrapper pretends the environment is actually `N` parallel environments for Stable-Baselines3. 
- In the simulation, there is **1 physical world**.
- To the PPO model, there are **4 environments**.

When `env.step([action1, action2, action3, action4])` is called, the wrapper passes these 4 actions into the single physics world, steps the world once, and extracts 4 independent observations.

## 3. Observation Space (177 Dimensions)
Each agent perceives the world through a highly localized, ego-centric continuous array of 177 floats.

| Index Range | Description | Size |
|-------------|-------------|------|
| `[0:3]`     | Ego-centric position `(x, y, theta)` | 3 |
| `[3:5]`     | Current velocity `(vx, omega)` | 2 |
| `[5]`       | Holding carton flag `(0 or 1)` | 1 |
| `[6:86]`    | LiDAR Raycasts (Obstacles & Walls) | 80 |
| `[86:146]`  | Goal/Carton tracking mechanisms | 60 |
| `[146:177]` | **Inter-Agent Communication Array** | 31 |

## 4. The Inter-Agent Communication Network
The most critical feature of the 177D space is the communication array. 
Because the policy is shared, robots need a way to differentiate themselves and understand where their peers are to avoid gridlock.
- The wrapper injects relative positional data of the other 3 agents directly into the observation space.
- The `RecurrentPPO` (LSTM) network uses this stream to learn predictive avoidance and swarm optimization.

## 5. Action Space
The model outputs a continuous `Box(-1.0, 1.0, shape=(2,))`.
- **Action[0] (Linear Velocity):** Moves the robot forward/backward.
- **Action[1] (Angular Velocity):** Steers the chassis left/right.

Because it's a differential drive, lateral strafing is physically impossible. The model learned to perform complex multi-point turns to navigate tight corners.
