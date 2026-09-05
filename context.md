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
3. **[PENDING] Missing Training Pipeline**:
   - Since `train.py`, `vec_env.py`, and `training.py` were removed in `f2eb796`, `multi-agent-v2` needs a clean, modern `train.py` that hooks `HiveMindExtractor` into Stable-Baselines3 PPO, sets up parallel worlds, and implements curriculum learning (1 -> 2 -> 4 -> 8 -> 12 cartons).
4. **[PENDING] Missing Evaluation & Baseline Scoring**:
   - `scripts/run_evaluation.py` is needed to evaluate trained checkpoints against the 97-step greedy baseline.

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
- [ ] **Step 3: Implement Clean Training Pipeline (`train.py`)**
  - Implement a streamlined `train.py` supporting:
    - Custom feature extractor `HiveMindExtractor`.
    - Parameter-shared multi-agent execution (each 4-robot warehouse presented to SB3 as 4 policy slots).
    - Subproc vectorization for multi-core parallelism (`--worlds`).
    - Robust curriculum learning (1 carton -> 2 -> 4 -> 8 -> 12 cartons on rolling success threshold).
    - Checkpointing and TensorBoard metrics.
    - `--smoke` mode for immediate verification.
- [ ] **Step 4: Restore Evaluation Tooling**
  - Restore/adapt a greedy baseline evaluator and policy evaluator to compute makespan, collision rates, and completion percentages.
- [ ] **Step 5: Test & Launch on Server**
  - Push updated code to `server:multi-agent-v2`.
  - Run a smoke test on the server.
  - Start the 5M-step curriculum run in the background (detached / tmux) and monitor rollout metrics.

---

## 8. Essential Commands Reference

```bash
# Run environment verification
/home/taksh/miniconda3/envs/hivemind/bin/python verify_environment.py

# Push changes from local to remote server
git push server multi-agent-v2

# Run command on server via SSH
ssh raid@10.36.16.97 "cd ~/hivemind/HiveMind && .venv/bin/python verify_environment.py"

# Monitor server GPU / CPU
ssh raid@10.36.16.97 "nvidia-smi"
```
