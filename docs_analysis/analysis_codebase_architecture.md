# Complete Codebase Architecture & Change Log

This document maps every single file in the project, documents its purpose, lists every function, and tracks exactly what was changed between the original codebase and our v2 upgrades.

---

## 1. Project File Tree

```
Single-Agent-implementation/
├── hivemind_env/                    # The Gymnasium Environment Package
│   ├── __init__.py                  # Registers "HiveMind-SingleAgent" with gym
│   ├── env.py                       # ★ Core Environment (579 lines)
│   ├── models.py                    # ★ Custom CNN Feature Extractor (63 lines)
│   └── assets/
│       └── diff_drive_bot.urdf      # Robot URDF model (wheels, arm, gripper, LiDAR)
│
├── scripts/                         # Test & Validation Scripts
│   ├── test_arm_pickup.py           # Tests robotic arm pickup animation
│   ├── test_env.py                  # Basic environment smoke test
│   ├── test_perception.py           # Tests the 15x15x5 observation grid
│   └── test_robot_mechanics_end_to_end.py  # Full physics E2E test
│
├── models/                          # Saved Model Weights
│   ├── ppo_hivemind_test.zip        # Early 20K-step test model (PPO, Level 1)
│   └── checkpoints/                 # RecurrentPPO v2 checkpoints (250K intervals)
│       ├── recurrent_ppo_hivemind_250000_steps.zip
│       ├── ... (35 checkpoints total)
│       └── recurrent_ppo_hivemind_8750000_steps.zip
│
├── runs/ppo/                        # Original training run artifacts
│   ├── args.json                    # Config: 20K steps, 2 envs, difficulty=1
│   └── final_model.zip              # The original 20K-step PPO model
│
├── tensorboard_logs/                # TensorBoard Event Files
│   ├── PPO_1/ through PPO_6/        # Standard PPO runs (train.py)
│   └── RecurrentPPO_1/ through _3/  # LSTM PPO runs (train_v2.py)
│
├── train.py                         # ★ Original training script (PPO, 5M steps)
├── train_v2.py                      # ★ Upgraded training script (RecurrentPPO, 10M)
├── test_run.py                      # BFS-based deterministic demo (Level 4)
├── run_demo_train.py                # Loads trained model for visual demo
├── test_env.py                      # Quick environment verification
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package metadata
├── environment.yml                  # Conda environment spec
└── README.md                        # Project documentation
```

---

## 2. Core File: `hivemind_env/env.py` (579 lines)

### Function Map

| Function | Lines | Purpose |
|:---|:---|:---|
| `_bfs_path_exists(grid, start, goal)` | 11-27 | Validates that a path exists between two points on the grid (used during map generation to prevent unsolvable levels) |
| `HiveMindSingleAgentEnv.__init__()` | 32-65 | Initializes PyBullet, defines action space (Discrete 7), observation space (Dict: 15x15x5 grid + is_carrying) |
| `_grid_to_world(r, c)` | 67-70 | Converts grid coordinates (row, col) to PyBullet world coordinates (x, y) |
| `_world_to_grid(x, y)` | 72-75 | Inverse of above |
| `reset()` | 77-191 | Resets simulation, generates new map, spawns robot/resource/depot/obstacles/walls |
| `_set_arm_and_lidar_joints()` | 193-204 | Controls the robotic arm yaw, gripper fingers, and LiDAR mount height |
| `_get_cardinal_direction_angle()` | 206-217 | Calculates the nearest 90° angle from robot to target (for arm pointing) |
| `_generate_valid_map()` | 219-251 | Procedurally generates spawn positions and obstacles, validates with BFS |
| `_step_substep()` | 253-268 | Executes one physics micro-step with wheel rotation and joint updates |
| `step(action)` | 273-466 | **The main loop.** Executes action, applies physics, calculates rewards |
| `_get_lidar_scan()` | 468-494 | Fires 180 PyBullet rays in 360° arc, returns hit results |
| `_get_obs()` | 497-558 | Builds the 15x15x5 egocentric grid observation |
| `_get_info()` | 560-571 | Returns debug info (difficulty, position, LiDAR distances) |

### Changes Made (Original → v2)

| Component | Original | Changed To | Why |
|:---|:---|:---|:---|
| `_get_lidar_scan()` default `num_rays` | 36 | **180** | Eliminate blind spots for 0.2m obstacles |
| `_get_lidar_scan()` default `max_range` | 1.5 | **2.0** | Extend detection range |
| `_get_obs()` LiDAR call | `num_rays=36, max_range=1.5` | `num_rays=180, max_range=2.0` | Match new sensor config |
| `_get_info()` LiDAR call | `_get_lidar_scan()` (defaults) | `num_rays=180, max_range=2.0` | Match new sensor config |
| `_get_info()` distance calc | `hit[2] * 1.5` | `hit[2] * 2.0` | Match new max_range |
| Reward: PBRS comment | "PBRS" | "PBRS - The Attractive Force" | Documentation clarity |
| **NEW: APF Repulsive Force** | — | Lines 451-462 | Adds LiDAR proximity penalty for obstacles < 0.4m |

---

## 3. Core File: `hivemind_env/models.py` (63 lines)

### Architecture

```
Input: Dict(grid: 15x15x5, is_carrying: Discrete(2))
                    │                        │
            ┌───────┘                        └──────┐
            ▼                                        ▼
   Conv2d(5→32, 3x3, pad=1)                   One-Hot (2 dims)
   ReLU + MaxPool(2) → 7x7                         │
   Conv2d(32→64, 3x3, pad=1)                       │
   ReLU + MaxPool(2) → 3x3                         │
   Flatten → 576 dims                              │
            │                                        │
            └──────────┬─────────────────────────────┘
                       ▼
               Concatenate → 578 dims
                       ▼
               Linear(578 → 256) + ReLU
                       ▼
               Output: 256-dim feature vector
```

**No changes were made to models.py.** The same CNN architecture is shared between PPO and RecurrentPPO. In RecurrentPPO, this feature vector feeds into an LSTM layer that SB3 adds automatically via `MultiInputLstmPolicy`.

---

## 4. Training Scripts Comparison

### `train.py` (Original, 90 lines)

| Feature | Implementation |
|:---|:---|
| Algorithm | `PPO("MultiInputPolicy")` |
| Parallelism | `SubprocVecEnv` (up to 16 CPUs) |
| Monitor | ❌ None (critical bug) |
| Curriculum | `CurriculumCallback` (90% threshold, window=100) |
| Checkpoints | `CheckpointCallback` (every 100K steps) |
| Total Steps | 5,000,000 |
| Hyperparams | Defaults (`ent_coef=0.01, gamma=0.99, n_steps=2048`) |

### `train_v2.py` (Upgraded, 109 lines)

| Feature | Implementation |
|:---|:---|
| Algorithm | `RecurrentPPO("MultiInputLstmPolicy")` from sb3_contrib |
| Parallelism | `SubprocVecEnv` (up to 16 CPUs) |
| Monitor | ✅ `VecMonitor` wrapper (fixes reward logging) |
| Curriculum | `CurriculumCallback` (85% threshold) |
| Checkpoints | `CheckpointCallback` (every 250K steps) |
| Total Steps | 10,000,000 |
| Hyperparams | `ent_coef=0.05, gamma=0.995, n_steps=4096//num_cpu` |

### Key Differences

```diff
# Algorithm
- from stable_baselines3 import PPO
+ from sb3_contrib import RecurrentPPO

# Monitor Wrapper (THE critical fix for reward logging)
+ env = VecMonitor(env)

# Policy Type
- model = PPO("MultiInputPolicy", ...)
+ model = RecurrentPPO("MultiInputLstmPolicy", ...)

# Hyperparameters
- ent_coef=0.01 (default)
+ ent_coef=0.05

- gamma=0.99 (default)
+ gamma=0.995

- n_steps=2048 (default)
+ n_steps=4096 // num_cpu  # BUG: Should not divide by num_cpu
```

---

## 5. Demo & Test Scripts

### `test_run.py` — BFS Deterministic Demo
Uses BFS pathfinding to compute the mathematically optimal action sequence, then executes it step-by-step in the PyBullet GUI with LiDAR visualization. This demonstrates the environment mechanics work perfectly — the problem was always the AI's learning, not the physics.

### `run_demo_train.py` — AI Model Demo
Loads a saved `.zip` model and runs it in the GUI. Currently hardcoded to use `PPO.load()`, which will crash when loading a RecurrentPPO model. **Must be updated to use `RecurrentPPO.load()` for v2 models.**

---

## 6. What the Next Training Run Must Fix

> [!IMPORTANT]
> Based on the exhaustive analysis across all documents, the next `train_v3.py` must implement these 5 critical fixes:
> 
> 1. **`ent_coef = 0.005`** — Allow the policy to become deterministic
> 2. **`n_steps = 2048`** — Do NOT divide by num_cpu for RecurrentPPO
> 3. **PBRS scale = 5.0** — Make task reward dominate entropy noise
> 4. **Pickup reward = +5.0** — Stronger incentive for the critical action
> 5. **Fix CurriculumCallback** — Check per-env rewards, lower threshold to 70%
