# Comparative Analysis: PPO_6 vs RecurrentPPO_3 & The Path Forward

This document provides a definitive side-by-side comparison of the two architectures, diagnoses the exact mathematical reasons each failed, and provides the precise optimized configuration for a successful training run.

---

## 1. Head-to-Head Comparison (TensorBoard & Live Demo Results)

We executed a 40-episode live demo for both models across all 4 difficulty levels to confirm the TensorBoard forensic analysis.

| Metric | PPO_6 (Feed-Forward) | RecurrentPPO_3 (LSTM) | Verdict |
|:---|:---|:---|:---|
| **Duration** | 1.27h | 4.58h | RPPO is 3.6x slower per step |
| **Has Reward Graph?** | ❌ No (train.py missing VecMonitor) | ✅ Yes | RPPO wins |
| **Final Entropy** | **-0.450** (collapsed/deterministic) | **-1.619** (forced random) | Both are wrong extremes |
| **Final Value Loss** | **3.559** (diverging) | **0.004** (converged) | RPPO wins |
| **Live Demo Success Rate** | **7.5%** (3/40 episodes) | **0%** (0/40 episodes) | Both failed to learn |
| **Live Demo Timeouts** | **90%** (36/40 episodes) | **100%** (40/40 episodes) | Both learned to stall |
| **Live Demo Behavior** | Spammed "Pick Up" 83% of the time | Avoided walls, did nothing else | Neither solved the task |

---

## 2. What Each Architecture Got Right

### PPO_6: Correct Gradient Dynamics, Wrong Environment
- ✅ Clip fraction (0.14) and KL (0.013) are textbook-healthy
- ✅ Entropy collapsed normally (-0.45), showing the agent committed to a policy
- ❌ But the policy it committed to was "spin in circles" because:
  - 36-ray LiDAR couldn't see obstacles → inconsistent collision penalties
  - No memory → couldn't plan multi-step pickup-transport-dropoff
  - Euclidean reward trapped it in corners

### RecurrentPPO_3: Correct Architecture, Wrong Hyperparameters
- ✅ VecMonitor successfully logged reward graphs
- ✅ LSTM architecture is theoretically correct for POMDPs
- ✅ Value loss converged beautifully (18.7 → 0.004)
- ❌ But `ent_coef=0.05` drowned out all learning signal
- ❌ 60% clip fraction means the optimizer is fighting itself
- ❌ The agent literally cannot become deterministic, so it can never learn a purposeful policy

---

## 3. The Entropy Problem (Visualized)

The core failure of RecurrentPPO_3 can be understood through entropy:

```
Maximum Entropy (7 actions): -1.946 (pure random)
RecurrentPPO_3 Final:        -1.619 (83% random)  ← STUCK HERE
PPO_6 Final:                 -0.450 (23% random)  ← Over-collapsed
Ideal Target:                -0.800 to -1.200     ← Sweet spot
```

- **PPO_6** went too far — it became so deterministic it couldn't explore new strategies.
- **RecurrentPPO_3** didn't go far enough — it was mathematically forced to stay random by the entropy coefficient.
- **The fix:** Set `ent_coef = 0.005` to allow gradual, natural entropy decay into the sweet spot.

---

## 4. The n_steps Problem (LSTM Sequence Length)

RecurrentPPO processes experiences in sequences. The LSTM hidden state is reset at the beginning of each sequence. If the sequence is too short, the LSTM can't build up meaningful memory.

```
Current:    n_steps = 4096 // 16 cpus = 256 steps per env
            → LSTM sees 256-step chunks of a 500-step episode
            → Hidden state is reset mid-episode, destroying memory

Required:   n_steps = 2048 (NOT divided by num_cpu)
            → LSTM sees 2048-step sequences (4+ full episodes)
            → Hidden state can build across multiple complete episodes
```

> [!IMPORTANT]
> In SB3's RecurrentPPO, `n_steps` should NOT be divided by `num_cpu`. SB3 handles the parallelism internally. By dividing, we accidentally cut the sequence length by 16x.

---

## 5. The Reward Signal-to-Noise Ratio

The fundamental mathematical problem:

```
PBRS reward for 1 step closer:     +0.04 (0.2m × 1.0 scale)
Time penalty per step:              -0.01
APF repulsive penalty near wall:    -0.05 to -2.0
Entropy bonus per update:           ~0.05 × entropy_change

→ The entropy bonus is comparable to the ACTUAL TASK REWARD
→ The network can't distinguish "good move" from "random noise"
```

**Fix:** Multiply PBRS scaling from 1.0 → 5.0. Now a single correct step = +0.20, which dominates the entropy bonus and gives the gradient a clear signal.

---

## 7. What Needs to Change (Optimized v3 Configurations)

Depending on which architecture you choose to move forward with, here are the exact code fixes required.

### Option A: The RecurrentPPO Fixes (`train_v2.py`)
If you want to continue with the LSTM memory model (Recommended for full navigation):

| Parameter | RecurrentPPO_3 (broken) | Optimized v3 | Rationale |
|:---|:---|:---|:---|
| `ent_coef` | 0.05 | **0.005** | Allow deterministic policy emergence |
| `n_steps` | 256 (4096/16) | **2048** | Full LSTM sequence length, not divided |
| `gamma` | 0.995 | **0.99** | Slightly less future-looking to speed convergence on Level 1 |
| PBRS scale | 1.0 | **5.0** | Make task signal dominate entropy noise |
| Pickup reward | +2.0 | **+5.0** | Stronger incentive for pickup action |
| Curriculum threshold | 85% | **70%** | Lower bar for level progression |
| `max_steps` | 500 | **750** | Give more time for complex Level 3-4 navigation |
| `learning_rate` | 3e-4 | **1e-4** | Slower, more stable LSTM weight updates |

### Option B: The Standard PPO Fixes (`train.py`)
If you want to revert to the feed-forward model (Faster to train, but requires environment hacking to bypass amnesia):

1. **Add `VecMonitor` (CRITICAL):**
   ```python
   from stable_baselines3.common.vec_env import VecMonitor
   env = SubprocVecEnv([make_env(rank) for rank in range(16)])
   env = VecMonitor(env) # MUST ADD THIS to log rewards in TensorBoard
   ```
2. **Increase Sequence Length:** 
   Change `n_steps=512` to `n_steps=2048` to give the agent longer horizons per gradient update.
3. **Hack the Environment to bypass Amnesia:**
   Because feed-forward PPO has no memory, it will forget where the dropoff is the moment it picks up the resource. To fix this, you must change `hivemind_env/env.py` to mathematically point the agent toward the dropoff zone *only when it is carrying the resource*.
   ```python
   # In env.py step()
   if self.is_carrying == 1:
       target_pos = self.depot_pos
   else:
       target_pos = self.resource_pos
   # Calculate PBRS reward based on target_pos
   ```

---

## 8. Codebase Issues Found During Review

### Issue 1: `train_v2.py` Line 27 — Wrong Reward Check
```python
# CURRENT (broken): Only checks rewards[0], ignoring other parallel envs
success = 1.0 if self.locals["rewards"][0] > 5.0 else 0.0

# FIX: Check the reward for the specific env that finished
for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
    if done:
        success = 1.0 if self.locals["rewards"][i] > 5.0 else 0.0
```

### Issue 2: `train_v2.py` Line 41 — `env_method` vs `set_attr`
```python
# CURRENT: Uses env_method which requires the method to exist on the env
self.training_env.env_method("set_difficulty", self.current_level)

# BETTER: Use set_attr directly (more reliable through VecMonitor wrapper)
self.training_env.set_attr("difficulty_level", self.current_level)
```

### Issue 3: `train.py` Line 66 — Missing `import torch`
```python
# Line 66 uses torch but it's never imported at the top
device = "cuda" if torch.cuda.is_available() ...
# Missing: import torch
```

### Issue 4: `run_demo_train.py` Line 12 — Uses PPO.load for RecurrentPPO model
```python
# CURRENT: Tries to load LSTM model with standard PPO
model = PPO.load(model_path)

# FIX: Must use RecurrentPPO.load for LSTM models
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load(model_path)
```

### Issue 5: `test_run.py` Line 12 — Still uses old 36-ray LiDAR for debug visualization
```python
# CURRENT: Draws only 36 rays in the debug view
lidar_results, ray_froms, ray_tos = unwrapped_env._get_lidar_scan(num_rays=36, max_range=1.5)

# FIX: Match the upgraded 180-ray sensor
lidar_results, ray_froms, ray_tos = unwrapped_env._get_lidar_scan(num_rays=180, max_range=2.0)
```
