# Architecture and Current Implementation

This document provides a deep dive into "what" and "how" the HiveMind multi-agent environment is implemented in the `multi-agent-rl` branch.

## 1. The Environment (`env.py`)

The multi-agent setting is a Gymnasium environment that acts as a wrapper for PyBullet physics. 

### Physics and Motion
- **Grid Layout**: A $13 \times 13$ m arena with 1m cells. The depot is at `(0,0)`. Solid shelves occupy odd rows with randomized gaps holding 12 cartons.
- **Motion Model**: Robots move via discrete actions (Forward, Backward, Left, Right). 
- **Substeps Interpolation**: A single grid-step action is split across `substeps` (default 5 for headless, 30 for GUI) using `resetBasePositionAndOrientation` and `stepSimulation` loops. This ensures physics interactions (like bumping cartons) occur smoothly without tunneling through shelf posts, though it's technically a teleportation sequence rather than true velocity control.
- **Solidity**: Shelves are solid obstacles. Unlike previous iterations where driving into shelves merely penalized the robot, the environment now actively blocks invalid moves (charging a -0.5 penalty) ensuring physics stability.

## 2. Observation Space (V3)

The observation is a flattened `Box` pinned strictly at **177 floats per robot**. Pinning the width is a critical architectural decision: altering this invalidates all previously trained model checkpoints.

**Layout per robot (`OBS_WORLD_DIM` = 129 + `OBS_MESSAGE_DIM` = 48):**
1. `[0:3]`: Own pose ($x, y$, wrapped heading).
2. `[3:5]`: Own velocity (displacement, clipped).
3. `[5:6]`: Carrying flag (0 or 1).
4. `[6:15]`: Poses of the 3 other robots.
5. `[15:18]`: Carrying flags of the 3 other robots.
6. `[18:30]`: Carton status (ordinal values: available, claimed by me, claimed by other, delivered).
7. `[30:54]`: Carton positions.
8. `[54:56]`: Depot direction.
9. `[56:57]`: Elapsed time fraction.
10. `[57:129]`: 72-ray LiDAR (0.1 to 10m range, 270° arc, injected with 1% Gaussian noise). The origin is placed carefully 0.28m from the chassis center to avoid self-collision with the robot's own geometry.
11. `[129:177]`: **Reserved EC Message Slots** (16 tokens $\times$ 3 robots). Currently zeroed, allowing future emergent communication (Step 7) without changing the network width.

## 3. The Feature Extractor (`models.py`)

Because the observation vector is a flattened bag of 177 floats, passing it directly into an MLP would destroy the spatial correlation of the 72 LiDAR rays. The `HiveMindExtractor` resolves this by slicing the observation into three streams:

1. **World Stream** (`[0:57]`): Processed through a standard 2-layer MLP (hidden dim 128).
2. **LiDAR Stream** (`[57:129]`): Cast into a 1D sequence and processed via a stride-2 1D Convolutional Neural Network (CNN). This allows the network to learn translation-invariant spatial features (e.g., "gap on the left").
3. **Message Stream** (`[129:177]`): Processed through a 64-dim MLP.

These three branches are concatenated and passed through a final projection head to output a 256-dim feature vector for the PPO actor/critic networks.

## 4. Multi-Processing Vectorization (`subproc_vec_env.py` / `vec_env.py`)

Stable-Baselines3 (SB3) inherently expects single-agent environments. To circumvent this without building a custom MAPPO implementation from scratch, HiveMind uses a wrapper trick:

- **`HiveMindSharedPolicyVecEnv`**: Unrolls `num_worlds` (e.g., 8) into `num_worlds * 4` policy slots. 
- To SB3, the batch appears as 32 independent robots. In reality, every 4 slots are interacting within the exact same shared PyBullet physics world.
- **`HiveMindSubprocVecEnv`**: Maps each of the 8 warehouses to its own CPU process. Training in PyBullet is highly CPU bound; using sub-processing increases throughput from ~34 steps/s to ~844 steps/s on 16 threads.
