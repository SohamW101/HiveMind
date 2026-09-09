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

## 4. Discovered Training Pathologies & Physics Fixes

### A. Geodesic Potential Shaping & Heading Traps
- **Multi-Trip Depot Blockade Fix**:
  - In earlier runs, when remaining active cartons were already picked up by teammates, the potential function attracted empty robots to the held cartons, causing them to swarm and blockade the carrier at the depot doorway.
  - Fix: When all uncollected cartons are held, empty robots receive a neutral potential gradient (`d_cells = 0.0`), allowing them to yield, disperse, and clear the corridor.
- **Why Artificial Heading Potentials Broke Training**:
  - In `v2_turn_curriculum`, an explicit heading term was added to reward facing the next BFS grid cell.
  - Pathology: Moving forward down a straight corridor into a turning junction (e.g. `(0, 2) -> (0, 3)`) caused the next-waypoint heading error to jump from 0° to 90° *before* the robot could turn. This penalized forward motion, causing value loss to spike to ~250 and trapping robots into spinning or idling.
  - Fix: Reverted to pure geodesic BFS distance. Without artificial heading terms, forward motion is smoothly rewarded and value loss drops to normal levels (~8-15).

### B. Robot Spawn Orientations
- Randomizing initial spawn angles (`corner_open_yaws`) caused 50% of robots to spawn facing walls or dead-end outer columns (e.g. Robot 0 driving south down column 0 into the bottom wall).
- Fix: Restored deterministic default spawn orientations (`yaw = 0.0`, facing East down warehouse corridors), maintaining consistent corridor exploration conditions.

### C. Turning Action Tax
- By default in the specification, robots turning on the spot have near-zero linear velocity and incurred an idle penalty (`R_IDLE_PENALTY = -0.02`). Combined with step time penalties, taking `turnL` or `turnR` was strictly worse than doing nothing (`stay`).
- Fix: Set `idle_penalises_turning = False` in `HiveMindMultiAgentEnv`, exempting turns from the idle tax.

### D. Curriculum Anti-Thrashing Safeguards
- **Grace Period**: Added a 400,000-step grace period (`min_steps_before_demote = 400_000`) before demotion checks can evaluate on newly promoted levels.
- **Fraction-Aware Protection**: If agents are delivering partial cartons (`avg_delivered_fraction >= 0.25`), the policy is actively learning multi-robot coordination and is protected from premature demotion.
- **Promotion Threshold**: Set `curriculum_fraction = 0.60` so delivering 2/3 cartons on Level 3 smoothly triggers promotion to 4 cartons.
- **Learning Rate Floor**: Added `min_value = 5e-5` to `linear_schedule` so policy gradients never freeze to zero after long runs.

---

## 5. Current Verified Training Status

- **Scratch Run Verification (`v2_clean_scratch`)**:
  - Starting from Level 1 with 1 carton, the policy converged and **promoted to Level 2 (2 cartons)**.
  - Reached 3.5M steps with delivered fraction climbing to **37.2%**, saving checkpoints every 25k steps (`ckpt_3498880_steps.zip`).
- **Resumed Run**:
  - Resumed from `ckpt_3498880_steps.zip` in tmux session `v2_training:0` on the server (`10.36.16.97`) to complete 2 -> 3 -> 4 -> 8 -> 12 carton ladder.
