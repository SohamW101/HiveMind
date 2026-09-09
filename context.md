# HiveMind Project Context & Codebase State

> **CRITICAL PERSISTENT MEMORY NOTICE**  
> Due to an IDE environment limitation, conversation history is not preserved across session reloads/logoffs.  
> **ANY AI ASSISTANT STARTING A NEW SESSION MUST READ THIS FILE (`context.md`) FIRST BEFORE TAKING ACTION.**  
> Whenever significant progress, code changes, or architectural decisions are made, this file **MUST BE UPDATED** to preserve context for future threads.

---

## 1. Executive Summary & Project Mission

- **Project**: HiveMind (Multi-Agent Warehouse Coordination — MAWC).
- **Core Problem**: 4 autonomous differential-drive robots operate in a 13×13 grid warehouse containing 6 double-shelf rows. They must collect 12 cartons (resources) located in aisle gaps and deliver them to a corner depot cell `(0, 0)` with minimum makespan.
- **Research Question**: Can the robots discover emergent coordination and meaningful communication through a discrete 16-token broadcast radio channel without central routing or predefined role assignment?
- **Current Phase**: Project lead/mentor intervention to achieve policy convergence before the imminent deadline. The team has started afresh on the `multi-agent-v2` branch to simplify and stabilize the pipeline.

---

## 2. Quantitative Targets & Benchmarks

| Metric | Target / Baseline | Description |
|---|---|---|
| **Greedy Baseline Makespan (12 cartons)** | **97 steps** | Deterministic scripted greedy solver over 30 fixed seeds (100% completion, 232.6m distance, 6.7 collisions). |
| **Greedy Baseline (4 cartons)** | **23 steps** | 100% completion over 30 seeds. |
| **Greedy Baseline (8 cartons)** | **58 steps** | 100% completion over 30 seeds. |
| **Success Criterion** | **Makespan < 97 steps** | Any RL policy that fails to complete 12 cartons faster than 97 steps has not beaten the baseline. |
| **Reward Split** | **90% Shared / 10% Individual** | +100.0 team delivery completion bonus, +10.0 per carton delivered, makespan bonus `50 * (T_max - T_actual)/T_max`. Individual rewards: +1.0 pickup, +2.0 delivery, step penalties. |

---

## 3. Hardware, Compute & Network Infrastructure

### Remote Training Server (`raid@10.36.16.97`)
- **Host**: `raid@10.36.16.97` (SSH working without password via key).
- **Directory**: `~/hivemind/HiveMind`
- **Python Environment**: `~/hivemind/HiveMind/.venv/bin/python` (PyBullet, PyTorch, Gymnasium, SB3 installed).
- **Compute**: Multi-core CPU + GPUs (physics simulation is CPU-bound; parallel worlds match physical CPU cores).
- **CRITICAL NETWORK CONSTRAINT**: The server **DOES NOT have outbound internet access to GitHub**.
  - Direct `git pull origin` fails on server.
  - **Code Synchronization Protocol**: All code must be pushed from local to server via SSH git remote:
    ```bash
    git push server <branch>
    # or scp / rsync for standalone files/weights
    ```
  - Remote `server` is configured locally as `raid@10.36.16.97:~/hivemind/HiveMind`.

### Local Development Machine
- **Workspace**: `/home/taksh/HiveMind`
- **Active Branch**: `multi-agent-v2` (tracking `origin/multi-agent-v2`)
- **Python Environment**: `/home/taksh/miniconda3/envs/hivemind/bin/python` (Python 3.10, PyTorch with CUDA support).
- **GitHub Remote**: `origin` (`git@github.com:SohamW101/HiveMind.git`).

---

## 4. Git Branch State & History

### Active Branch: `multi-agent-v2`
- **Origin**: Branch created by team members (`Het Thakkar`, `codr-shiv`) to start afresh.
- **Latest Commit**: `dd0637f` (*merged commits Merge branch 'multi-agent-v2'...*).
- **Synced Status**: Both local workspace and server `~/hivemind/HiveMind` are checked out to `multi-agent-v2`.
- **What was added in `multi-agent-v2`**:
  1. `hivemind_env/env.py`: Overhauled environment supporting V3 (177 floats) and V4 (1143 floats decentralized map), corner spawns `(0,0)`, `(0,12)`, `(12,0)`, `(12,12)`.
  2. `hivemind_env/models.py`: Added `HiveMindExtractor` (1D CNN for 72 LiDAR rays + MLP for state).
  3. `verify_environment.py`: Integration test suite validating observations, collisions, robot arbitration, pickup/drop, and comms. **Passes 100% locally and on server.**
  4. `implementation_plan.md`: Outlines the 1D CNN architecture for LiDAR.
- **What was deleted in `multi-agent-v2` (commit `f2eb796` "cleaned repo")**:
  - `train.py` (The main training script was wiped!).
  - `hivemind_env/training.py`, `hivemind_env/vec_env.py`, `hivemind_env/subproc_vec_env.py` (Vectorized environment wrappers and callbacks).
  - `scripts/` (Evaluation and diagnostic scripts like `run_evaluation.py`).
  - `smoke_test.py`.

### Legacy Branch: `multi-agent-rl`
- Preserves the previous monolithic codebase, including the old `train.py`, `scripts/`, and past checkpoints (`models/`).

---

## 5. Environment & Neural Architecture

### Observation Layouts
- **V3 Observation (177 floats - Default for PPO training)**:
  - `[0:3]`: Own pose `(x, y, theta)` normalized.
  - `[3:5]`: Own linear & angular velocity.
  - `[5:6]`: Own carrying status (0 or 1).
  - `[6:8]`: Relative vector to depot `(dx, dy)`.
  - `[8:9]`: Normalized episode step / time remaining.
  - `[9:57]`: Other agents' poses, carrying statuses, and 12 carton positions/statuses.
  - `[57:129]` (**72 floats**): 72-ray 2D planar LiDAR sweep (270° FOV, 3.75° spacing, range 0.1m - 10.0m).
  - `[129:177]` (**48 floats**): Discrete communication buffer (3 other agents × 16 one-hot tokens).
- **V4 Observation (1143 floats - Decentralized Local Map)**:
  - 6-channel 13×13 grid map (unknown, free, obstacle, resource, robot, depot) + local vector + 48 message slots.

### Neural Network: `HiveMindExtractor` (`hivemind_env/models.py`)
- Modular Semantic Feature Extractor (resolves the 105-dim information bottleneck):
  - **Self-State Branch** (9 dims: own pose, velocity, carrying, depot direction, elapsed time) -> `Linear(9 -> 64) -> LayerNorm -> GELU -> Linear(64 -> 64)`.
  - **Teammates Branch** (12 dims: other poses & carrying flags) -> `Linear(12 -> 64) -> LayerNorm -> GELU -> Linear(64 -> 64)`.
  - **Carton/Objective Branch** (36 dims: 12 statuses + 24 coordinates) -> `Linear(36 -> 128) -> LayerNorm -> GELU -> Linear(128 -> 64)`.
  - **Communication Branch** (48 dims: 3 teammate broadcast slots x 16 tokens) -> `Linear(48 -> 64) -> LayerNorm -> GELU -> Linear(64 -> 32)`.
  - **Planar LiDAR 1D-CNN Branch** (72 dims) -> 3-stage Conv1d (`1 -> 32 -> 64 -> 64`, k=5/3/3) -> Flatten (1152) -> `Linear(1152 -> 128) -> LayerNorm -> GELU`.
  - **Fusion Network**: Concatenates Self (64) + Teammates (64) + Cartons (64) + Comms (32) + LiDAR (128) = 352 dims -> `Linear(352 -> 256) -> LayerNorm -> GELU -> Linear(256 -> 256)` -> `features_dim = 256`.
- Policy / Value heads in Stable-Baselines3 attach cleanly via `get_policy_kwargs()`.

---

## 6. Discovered Bugs & Architectural Resolutions

1. **[RESOLVED] Relative Import Failure in `hivemind_env/models.py` & Namespace Collision in `testCNN.py`**:
   - Replaced `import env` with package-aware import `from . import env` (with fallback).
   - Fixed `testCNN.py` to import from `hivemind_env.models`, added forward/backward gradient checks and CUDA test. Both pass 100%.
2. **[DOCUMENTED] Incentive & Architecture Bottlenecks**:
   - Documented in detail in `ARCHITECTURE_IMPROVEMENTS.md`:
     - Collision Risk Tax (why bots froze: -4.5 collision penalty vs 0 progress reward).
     - The 1-Carton Swarm problem (4 bots chasing 1 carton in 1m aisles).
     - The 105-dim single-layer chokepoint (fixed by modular semantic extractor).
3. **[RESOLVED] Training Pipeline Implemented**:
   - Built `train.py` wired to `HiveMindExtractor` via `get_policy_kwargs()`.
   - Restored `hivemind_env/vec_env.py` and `hivemind_env/subproc_vec_env.py` for parallel multi-agent parameter sharing.
   - Restored `hivemind_env/training.py` with `CurriculumCallback` (starts at 1 carton, walks 1 -> 2 -> 3 -> 4 -> 8 -> 12), with `reset_lr_on_promotion=False` to eliminate destructive sawtooth LR spikes.
   - Validated end-to-end with smoke test (`train.py --smoke`).
4. **[RESOLVED] Evaluation Tooling & Greedy Baseline Restored**:
   - Restored `hivemind_env/greedy.py` and `scripts/run_evaluation.py`.
   - Fixed cardinal Manhattan distance bug (`abs(dr) + abs(dc) == 1`) in `GreedyController` to match `env.py`'s `INTERACTION_DISTANCE_CELLS = 1` contract (diagonal reach was previously causing greedy pickup/drop lockups).
   - Validated greedy baseline: 4 cartons completed in 33 steps (100% completion).
5. **[RESOLVED] Curriculum Thrashing & Anti-Demotion Safeguards**:
   - Diagnosed root cause of 6M–10M oscillation: `demote_after_checks = 4` (4,000 steps) was prematurely declaring newly promoted levels "hopeless" before the policy could explore multi-robot coordination.
   - Added **400,000-step grace period** (`min_steps_before_demote = 400_000`) before demotion checks can evaluate.
   - Added **fraction-aware protection** (`avg_fraction >= 0.20` prevents demotion if agents are delivering partial cartons).
   - Added `--init-from` support to warm-start policy weights from proven checkpoints.
6. **[ACTIVE] `v2_curriculum_fixed` Training Run in Remote tmux Session**:
   - Run Name: `v2_curriculum_fixed` (10,000,000 timesteps, 8 worlds / 32 slots on RTX A5000).
   - Warm-started from `ckpt_5998080_steps.zip` (mastery of 1 & 2 cartons at >80% success), starting at 2 cartons to immediately advance to 3, 4, 8, 12 cartons without thrashing.
   - Tmux Session: `v2_training` (Window 0: `train`, Window 1: `tensorboard` on port 6007).
   - Het's independent session `training` remains completely untouched on port 6006.

---

## 7. Immediate Roadmap & Action Plan

- [x] **Step 1: Codebase Synchronization**
  - Stash local experiments on `multi-agent-rl`.
  - Fetch `origin/multi-agent-v2` and checkout `multi-agent-v2` locally.
  - Push `multi-agent-v2` to server (`raid@10.36.16.97`) and checkout `multi-agent-v2` on the server.
  - Verify environment tests pass on both local and server (`verify_environment.py` PASS).
- [x] **Step 2: Fix Module Imports & Upgrade Feature Extractor**
  - Upgraded `hivemind_env/models.py` to modular semantic feature extractor.
  - Fixed import paths and verified forward/backward passes in `testCNN.py`.
  - Created `ARCHITECTURE_IMPROVEMENTS.md` explaining the freeze causes and fixes in accessible terms.
- [x] **Step 3: Implement Clean Training Pipeline (`train.py`)**
  - Implemented `train.py` with `HiveMindExtractor`, subproc vectorization, and stable curriculum.
  - Smoke test ran 4,096 steps and completed without error.
- [x] **Step 4: Restore Evaluation Tooling**
  - Restored `hivemind_env/greedy.py` and `scripts/run_evaluation.py`.
  - Corrected cardinal Manhattan interaction distance in greedy controller; verified 100% completion on 4 cartons (makespan 33).
- [x] **Step 5: Post-Mortem of `v2_turn_curriculum` Collapse & Clean Fix**
  - Diagnosed why `v2_turn_curriculum` collapsed (stuck at Level 1 with only 20% success over 10M steps):
    1. The artificial `heading_term` in `_potential` introduced severe local reward traps: moving forward into corner cells (e.g. `(0, 3)`) increased heading error from 0° to 90°, penalizing forward movement and incentivizing turning/spinning in place. `value_loss` spiked from ~40 to 255.
    2. Randomizing spawn orientation (`corner_open_yaws`) broke initial corridor line-of-sight and created massive collision/timeout variance.
    3. `ent-coef=0.03` created diffuse action noise, preventing trajectory crystallization.
  - Clean Resolution:
    1. Reverted `_potential` back to pure, monotonic geodesic BFS distance with the empty-robot yield fix (no heading distortion).
    2. Restored deterministic default spawn orientation (`yaw = 0.0`).
    3. Set `idle_penalises_turning=False` by default in `env.py` so turning on the spot is never penalized as idle.
    4. Restored `ent-coef=0.01` and set `curriculum-fraction=0.60` (allowing 2/3 deliveries on Level 3 to advance to 4 cartons).
    5. Preserved `min_value = 5e-5` in `linear_schedule` so policy updates never freeze at 0 learning rate.
- [ ] **Step 6: Launch Clean Fixed Curriculum Run (`v2_curriculum_resumed`)**
  - Warm-start from `ckpt_5998080_steps.zip` (mastery of 1 & 2 cartons at >80% success) at difficulty 2.
  - Run with clean geodesic shaping and `min_value = 5e-5` LR schedule to smoothly progress 2 -> 3 -> 4 -> 8 -> 12 cartons.
  - Monitor progression in tmux session `v2_training` on server.

---

## 8. Essential Commands Reference

```bash
# --- Het's Training Session ---
ssh -t raid@10.36.16.97 "tmux attach -t training"
# View Het's live logs:
ssh raid@10.36.16.97 "tmux capture-pane -t training:train -p -S -20"
# Forward Het's TensorBoard (port 6006):
ssh -L 6006:localhost:6006 raid@10.36.16.97

# --- v2 Training Session ---
ssh -t raid@10.36.16.97 "tmux attach -t v2_training"
# View v2 live logs:
ssh raid@10.36.16.97 "tmux capture-pane -t v2_training:train -p -S -30"
# Forward v2 TensorBoard (port 6007):
ssh -L 6007:localhost:6007 raid@10.36.16.97

# --- Hardware Monitor ---
ssh raid@10.36.16.97 "nvidia-smi"
```

