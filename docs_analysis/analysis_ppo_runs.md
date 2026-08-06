# Forensic Analysis: PPO Runs (PPO_1 → PPO_6)

This document is an exhaustive, metric-by-metric analysis of every standard PPO training run that was executed using `train.py`. It covers the exact numerical data extracted from TensorBoard, diagnoses why reward graphs are missing, and explains what the neural network actually learned.

---

## 1. Run Inventory & Configuration

All PPO runs used the original `train.py` script with these shared settings:
- **Algorithm:** Standard PPO (feed-forward, no memory)
- **LiDAR:** 36 rays, 1.5m range (the old blind configuration)
- **Rewards:** Euclidean PBRS only (no APF repulsive force)
- **No `VecMonitor`:** This is the critical bug. Without a `Monitor` wrapper, SB3 cannot calculate or log `rollout/ep_rew_mean` or `rollout/ep_len_mean`.

| Run | Steps Logged | Duration | Tags Found | Has Reward Graph? |
|:---|:---|:---|:---|:---|
| **PPO_1** | 2,048 → 10,240 | 0.01h (~36s) | 8 | ❌ No |
| **PPO_2** | 2,048 → 10,240 | 0.01h (~36s) | 8 | ❌ No |
| **PPO_3** | 2,048 → 10,240 | 0.01h (~36s) | 8 | ❌ No |
| **PPO_4** | 2,048 → 10,240 | 0.01h (~36s) | 8 | ❌ No |
| **PPO_5** | 2,048 → 20,480 | 0.02h (~72s) | 8 | ❌ No |
| **PPO_6** | 16,384 → 5,005,312 | 1.27h | 8 | ❌ No |

> [!CAUTION]
> **PPO_1 through PPO_5 are micro-runs.** They only trained for 10K-20K steps (under 2 minutes each). These were early test runs to verify the environment booted. They contain zero meaningful learning signal and should be ignored entirely.

> [!WARNING]
> **PPO_6 is the only real training run.** It ran for 5,005,312 steps over 1.27 hours on the server. However, it is MISSING the two most important graphs (`rollout/ep_rew_mean` and `rollout/ep_len_mean`) because `train.py` did not use `VecMonitor`.

---

## 2. PPO_6: The 5-Million Step Run (Deep Dive)

### 2.1 What Tags Were Logged (and What's Missing)

**Tags present (8):** `train/approx_kl`, `train/clip_fraction`, `train/clip_range`, `train/entropy_loss`, `train/explained_variance`, `train/learning_rate`, `train/loss`, `train/policy_gradient_loss`, `train/value_loss`

**Tags MISSING:** `rollout/ep_rew_mean`, `rollout/ep_len_mean`

**Why they're missing:** In Stable-Baselines3, the `rollout/` scalars are only logged when the environment is wrapped in a `Monitor` (or `VecMonitor`). The original `train.py` used raw `SubprocVecEnv` without any monitor wrapper. The `train/` tags come from PyTorch gradient computations and are always logged regardless.

### 2.2 Entropy Loss (The Exploration Curve)

| Metric | Value |
|:---|:---|
| Start (step 16,384) | **-1.935** |
| End (step 5,005,312) | **-0.450** |
| Min | -1.935 |
| Max | -0.421 |

**What this means:** Entropy measures the "randomness" of the agent's action distribution. At the start (-1.935), the agent is choosing actions almost uniformly at random (maximum entropy for 7 discrete actions is -ln(1/7) ≈ -1.946). By the end (-0.450), the agent has become highly deterministic — it has collapsed onto a small set of preferred actions.

**The problem:** An entropy of -0.45 means the agent is extremely confident in its chosen behavior. But since there are no reward graphs to verify, we cannot confirm whether it learned *good* behavior or *bad* behavior. Based on the test demos, the agent learned to spin in circles and avoid walls — a locally optimal but globally useless policy.

### 2.3 Value Loss (The Prediction Accuracy)

| Metric | Value |
|:---|:---|
| Start | **0.090** |
| End | **3.559** |
| Min | 0.090 |
| Max | 4.498 |
| Mean | **2.746** |

**What this means:** Value loss measures how accurately the critic network predicts future rewards. A rising value loss is a major red flag — it means the network's internal model of "what will happen next" is getting worse over time, not better.

**Root cause:** The Euclidean PBRS reward creates a non-stationary reward landscape. As the agent learns to avoid walls (and crashes less), the distribution of rewards it sees shifts. But because it never actually reaches the depot, it can't learn a stable value function. The critic is perpetually confused.

### 2.4 Explained Variance

| Metric | Value |
|:---|:---|
| Start | **-0.297** |
| End | **0.221** |
| Max | **0.806** |
| Mean | **0.338** |

**What this means:** Explained variance measures how well the value function predicts actual returns. A value of 1.0 is perfect prediction, 0.0 is random noise, and negative means worse than random. 

The fact that it oscillated wildly between -0.30 and 0.81 means the value function was unstable — sometimes it could predict returns well (in simple Level 1 scenarios) and sometimes it was completely lost (when the Euclidean trap kicked in).

### 2.5 Policy Gradient Loss

| Metric | Value |
|:---|:---|
| Start | **-0.024** |
| End | **-0.027** |
| Mean | **-0.031** |

**What this means:** This is relatively stable and small, which is normal for PPO. The policy gradient loss being consistently negative means the agent is finding actions that improve its expected return. However, the "return" it's optimizing is the flawed Euclidean reward, so it's optimizing for the wrong objective.

### 2.6 Clip Fraction

| Metric | Value |
|:---|:---|
| Start | **0.075** |
| End | **0.145** |
| Mean | **0.099** |

**What this means:** Clip fraction measures what percentage of policy updates were clipped by PPO's trust region. A healthy range is 0.05-0.15. PPO_6 sits right in this range, indicating stable gradient updates. The algorithm itself is working correctly — the problem is the environment, not the optimizer.

### 2.7 Approximate KL Divergence

| Metric | Value |
|:---|:---|
| Start | **0.008** |
| End | **0.013** |
| Mean | **0.010** |

**What this means:** KL divergence measures how much the policy changed between updates. Values below 0.02 are healthy for PPO. This confirms the training was numerically stable with no policy collapse or divergence.

---

## 3. Verdict: What PPO_6 Actually Learned (Based on Live Demo)

### The Neural Network Converged Mathematically...
Every gradient metric (KL, clip fraction, entropy decay) shows textbook-perfect convergence. The optimizer did its job flawlessly.

### ...But It Converged on the Wrong Behavior
We ran a 40-episode live demo of the trained PPO model (task-1592). The results reveal exactly what the agent learned:

| Level | Success Rate | Collisions | Timeouts | Pickups | Avg Reward |
|:---|:---|:---|:---|:---|:---|
| 1 | 0% | 0% | 100% | **100%** | -1.79 |
| 2 | **20%** | 0% | 80% | 40% | -187.52 |
| 3 | 10% | 10% | 80% | 20% | -64.51 |
| 4 | 0% | 0% | 100% | 30% | -239.01 |

**Behavioral Observations from the Demo:**
1. **The "Action Spam" Strategy:** In Level 1, the agent picked up the resource 100% of the time. However, it spammed the `Pick Up` action a massive **83.0%** of the time. It never dropped it off.
2. **Lucky Successes:** In Levels 2 and 3, it managed 3 total successful deliveries. But looking at the massive negative average rewards (e.g., -187.52), it's clear these were "lucky walks" where it happened to stumble into the dropoff zone while mashing buttons, rather than purposeful navigation.
3. **No Memory = Total Amnesia:** Because the agent has no LSTM memory, the moment it picks up the resource, its visual input (`is_carrying` flag) flips to 1. But it has no memory of where the dropoff zone is or what path it just took. It essentially wakes up with a package in its hands and no idea how it got there.

### The Three Root Causes
1. **No Memory (Amnesia):** PPO cannot solve a multi-step fetch quest without an LSTM to remember where the target is after pickup.
2. **No Monitor Wrapper:** No reward feedback loop was logged in TensorBoard, preventing humans from diagnosing these issues early.
3. **36-Ray LiDAR Blindness:** Robot couldn't physically see small obstacles, leading to inconsistent collision penalties.
