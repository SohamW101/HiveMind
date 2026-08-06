# Implementation Fixes & Improvements

This document outlines the exact, copy-paste ready code changes required to fix the training pipeline. Based on the complexity analysis, **Option A (Standard PPO with Environment Hacking)** is the highly recommended path forward. Option B is provided only if you insist on continuing with RecurrentPPO.

---

## OPTION A (RECOMMENDED): Standard PPO + Environment Hacking

This approach avoids the massive computational overhead of LSTMs by simply changing the Euclidean reward target mid-episode, curing the "Amnesia" problem mathematically.

### 1. Fix `hivemind_env/env.py` (The Target Hack)
Navigate to the `step()` function (around line 430), where the Euclidean reward is calculated. Replace the static distance calculation with a dynamic one that changes based on what the agent is carrying.

**DELETE THIS:**
```python
# PBRS - The Attractive Force
# Euclidean distance to the resource
dist_to_resource = np.linalg.norm(np.array(self.robot_pos) - np.array(self.resource_pos))
reward += (self.last_dist_to_resource - dist_to_resource) * 1.0
self.last_dist_to_resource = dist_to_resource
```

**ADD THIS:**
```python
# PBRS - The Dynamic Attractive Force
if self.is_carrying == 0:
    # Phase 1: Navigate to the Resource
    current_target = self.resource_pos
    dist_to_target = np.linalg.norm(np.array(self.robot_pos) - np.array(current_target))
    reward += (self.last_dist_to_resource - dist_to_target) * 5.0 # Increased scale to 5.0
    self.last_dist_to_resource = dist_to_target
else:
    # Phase 2: Navigate to the Depot (Cures Amnesia)
    current_target = self.depot_pos
    dist_to_target = np.linalg.norm(np.array(self.robot_pos) - np.array(current_target))
    
    # Initialize last_dist_to_depot if this is the exact step we picked it up
    if not hasattr(self, 'last_dist_to_depot'):
        self.last_dist_to_depot = dist_to_target
        
    reward += (self.last_dist_to_depot - dist_to_target) * 5.0 # Increased scale to 5.0
    self.last_dist_to_depot = dist_to_target
```

### 2. Fix `train.py` (The Logging and Decay Hacks)
The original `train.py` was missing the Monitor wrapper (no reward logging) and used static learning rates.

**DELETE THIS:**
```python
env = SubprocVecEnv([make_env(rank) for rank in range(16)])
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
```

**ADD THIS:**
```python
from stable_baselines3.common.vec_env import VecMonitor
from typing import Callable

# Create a Linear Learning Rate Decay function
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# Wrap the environment to enable reward logging
env = SubprocVecEnv([make_env(rank) for rank in range(16)])
env = VecMonitor(env)

# Initialize PPO with optimized parameters
model = PPO(
    "MultiInputPolicy", 
    env, 
    learning_rate=linear_schedule(3e-4), # Decays to 0.0
    n_steps=2048,                        # Increased sequence length
    ent_coef=0.01,                       # Standard exploration
    batch_size=256,
    verbose=1, 
    tensorboard_log=log_dir
)
```

### 3. Fix the Curriculum Threshold (`train.py`)
In the `CurriculumCallback` class, lower the requirement so the agent doesn't get stuck on Level 1 forever.

**DELETE THIS:**
```python
if success_rate > 0.85:
```

**ADD THIS:**
```python
# Lower threshold to 70% to trigger level graduation
if success_rate > 0.70:
```

---

## OPTION B (NOT RECOMMENDED): Fixing RecurrentPPO

If you absolutely must use the LSTM architecture, you must fix the hyperparameters in `train_v2.py` that forced the network into an entropy trap.

### 1. Fix `train_v2.py` (Hyperparameters)
**DELETE THIS:**
```python
model = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096 // 16,
    batch_size=256,
    ent_coef=0.05,
    gamma=0.995,
```

**ADD THIS:**
```python
model = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    learning_rate=1e-4,     # Slower, safer updates for LSTM weights
    n_steps=2048,           # DO NOT divide by num_cpus
    batch_size=256,
    ent_coef=0.005,         # Reduced by 10x to allow deterministic behavior
    gamma=0.99,             # Slightly shorter horizon
```

### 2. Fix `train_v2.py` (The Reward Bug)
In the `CurriculumCallback`, it was only checking the reward of the 0th environment, missing the other 15 parallel environments.

**DELETE THIS:**
```python
if "rewards" in self.locals:
    success = 1.0 if self.locals["rewards"][0] > 5.0 else 0.0
```

**ADD THIS:**
```python
if "rewards" in self.locals and "dones" in self.locals:
    # Check the specific environment that actually finished this step
    for i, done in enumerate(self.locals["dones"]):
        if done:
            success = 1.0 if self.locals["rewards"][i] > 5.0 else 0.0
            self.success_history.append(success)
```
