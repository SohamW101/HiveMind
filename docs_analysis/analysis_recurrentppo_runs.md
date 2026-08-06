# Forensic Analysis: RecurrentPPO Runs (RecurrentPPO_1 → RecurrentPPO_3)

This document is an exhaustive, metric-by-metric analysis of every RecurrentPPO (LSTM) training run executed using `train_v2.py`. Unlike the PPO runs, these runs successfully log reward and episode length graphs thanks to the `VecMonitor` wrapper.

---

## 1. Run Inventory & Configuration

All RecurrentPPO runs used `train_v2.py` with the upgraded architecture:
- **Algorithm:** RecurrentPPO (LSTM hidden state for spatial memory)
- **LiDAR:** 180 rays, 2.0m range (the upgraded high-res configuration)
- **Rewards:** Euclidean PBRS + APF Repulsive Force (LiDAR proximity penalty)
- **`VecMonitor`:** ✅ Active — reward and episode length are tracked!
- **Entropy Coefficient:** 0.05 (5x higher than PPO's 0.01)
- **Gamma:** 0.995 (vs PPO's default 0.99)

| Run | Steps Logged | Duration | Tags Found | Has Reward Graph? |
|:---|:---|:---|:---|:---|
| **RecurrentPPO_1** | 4,096 → 7,364,608 | 3.74h | 12 | ✅ Yes |
| **RecurrentPPO_2** | 4,096 → 3,055,616 | 1.60h | 12 | ✅ Yes |
| **RecurrentPPO_3** | 4,096 → 8,908,800 | 4.58h | 12 | ✅ Yes |

> [!IMPORTANT]
> **RecurrentPPO_3 is the primary run** — it trained for 8.9M steps over 4.58 hours. This is the model whose checkpoints are saved in `models/checkpoints/` (up to `8,750,000_steps.zip`). RecurrentPPO_1 and _2 were earlier attempts that were terminated early.

---

## 2. RecurrentPPO_3: The 8.9M Step Run (Deep Dive)

### 2.1 Episode Reward Mean (`rollout/ep_rew_mean`) — THE CRITICAL GRAPH

| Metric | Value |
|:---|:---|
| Start (step 4,096) | **-39.59** |
| End (step 8,908,800) | **-5.15** |
| Min (worst) | **-47.49** |
| Max (best) | **-4.63** |
| Mean | **-5.46** |

**Curve Shape:**
- **Steps 0 → ~50K:** Rapid improvement from -39.59 to approximately -5.5. This is the agent learning Level 1 basics (don't crash, move toward the target).
- **Steps ~50K → 8.9M:** Flat plateau at approximately **-5.0 to -5.5**. The reward oscillates in a narrow band for the remaining 8.85M steps.

> [!CAUTION]
> **This is the most critical finding.** The reward plateaued at -5.15 and never improved for 8.85 MILLION steps. This is a catastrophic training stall. The agent has converged on a fixed policy and is not learning anything new.

**What -5.15 reward means mathematically:**
- The time penalty is -0.01 per step. Over 500 steps (the max episode length), the time penalty alone = -5.0.
- A reward of -5.15 means the agent is surviving the full 500 steps (no collision termination) but never picking up the resource (+2.0) or dropping it off (+10.0). It is literally just existing for 500 steps, collecting the time penalty, plus a tiny bit of APF repulsive penalty.

### 2.2 Episode Length Mean (`rollout/ep_len_mean`) — CONFIRMS THE STALL

| Metric | Value |
|:---|:---|
| Start (step 4,096) | **124.40** |
| End (step 8,908,800) | **500.00** |
| Mean | **499.31** |

**What this means:** The agent learned to avoid all collisions almost immediately. Within the first ~50K steps, episode length jumped from 124 to 500 and stayed pegged at 500.0 for the entire remaining training.

**The bad news:** An episode length of exactly 500 means every single episode ends by truncation (timeout), never by termination (collision or successful dropoff). The agent has learned a single behavior: "survive without doing anything useful."

### 2.3 Entropy Loss — The Overexploration Problem

| Metric | Value |
|:---|:---|
| Start (step 8,192) | **-1.941** |
| End (step 8,908,800) | **-1.619** |
| Min | -1.941 |
| Max | **-1.516** |
| Mean | **-1.656** |

**Comparison with PPO_6:**
- PPO_6 entropy went from -1.935 → **-0.450** (massive collapse to deterministic)
- RecurrentPPO_3 entropy went from -1.941 → **-1.619** (barely moved)

**What this means:** The entropy barely decreased over 8.9M steps! Maximum entropy for 7 actions is -1.946. The agent is still choosing actions almost uniformly at random. The `ent_coef=0.05` setting is too aggressive — it is actively penalizing the network for becoming deterministic, forcing it to maintain near-random behavior forever.

> [!WARNING]
> **Root Cause Identified:** The entropy coefficient of 0.05 is 5x too high. It is preventing the network from committing to any learned behavior. The agent is being mathematically forced to explore forever and never exploit what it learns.

### 2.4 Clip Fraction — Dangerously High

| Metric | Value |
|:---|:---|
| Start | **0.058** |
| End | **0.593** |
| Mean | **0.603** |

**What this means:** A clip fraction of 0.60 means 60% of all policy gradient updates are being clipped. The healthy range for PPO is 0.05-0.15. At 0.60, the trust region constraint is being violated on the majority of updates.

**Root cause:** The high entropy coefficient forces the policy to stay random, which conflicts with the value function trying to update the policy toward better actions. The two signals fight each other, causing massive KL divergence and excessive clipping.

### 2.5 Approximate KL Divergence

| Metric | Value |
|:---|:---|
| Start | **0.012** |
| End | **0.134** |
| Mean | **0.125** |

**What this means:** KL divergence of 0.125 is extremely high for PPO (healthy is < 0.02). The policy is changing dramatically between updates — a sign of training instability caused by the entropy/clip fraction conflict.

### 2.6 Value Loss — Good Convergence, Bad Signal

| Metric | Value |
|:---|:---|
| Start | **18.70** |
| End | **0.004** |
| Min | **0.0007** |

**What this means:** The value loss dropped beautifully from 18.70 to near zero. But this is actually bad news in context — the value function learned to perfectly predict a reward of approximately -5.0 every episode. It converged on predicting "the agent will do nothing and timeout." The prediction is accurate, but the predicted behavior is useless.

### 2.7 Explained Variance

| Metric | Value |
|:---|:---|
| Start | **-0.008** |
| End | **0.681** |
| Max | **0.970** |
| Mean | **0.636** |

**What this means:** The explained variance improved significantly, reaching 0.97 at peak. This confirms the value function can accurately predict returns — but again, it's predicting the wrong thing (timeout penalty, not successful delivery).

---

## 3. Cross-Run Comparison (RecurrentPPO_1, _2, _3)

| Metric | RPPO_1 (7.4M) | RPPO_2 (3.1M) | RPPO_3 (8.9M) |
|:---|:---|:---|:---|
| **Final Reward** | -4.75 | -5.69 | -5.15 |
| **Final Ep Length** | 500.0 | 500.0 | 500.0 |
| **Final Entropy** | -1.613 | -1.687 | -1.619 |
| **Final Clip Frac** | 0.601 | 0.607 | 0.593 |
| **Final KL** | 0.138 | 0.122 | 0.134 |
| **Final Value Loss** | 0.005 | 0.008 | 0.004 |

**All three runs exhibit the identical failure pattern:**
1. Rapid initial improvement (first ~50K steps)
2. Immediate plateau at reward ≈ -5.0
3. Episode length pegged at 500 (timeout)
4. Entropy barely decreasing (stuck near -1.6)
5. Clip fraction dangerously high (~0.60)

This proves the failure is **systematic and architectural**, not a random training fluke.

---

## 4. Diagnosis: Why RecurrentPPO Failed

### Problem 1: Entropy Coefficient Too High (`ent_coef=0.05`)
The entropy bonus of 0.05 overwhelms the actual task reward signal. The network receives a stronger gradient signal from "keep exploring randomly" than from "move closer to the resource." It is mathematically impossible for the agent to learn a deterministic, purposeful policy under these conditions.

### Problem 2: The Reward is Too Small Relative to Entropy
The PBRS reward for moving one cell closer is approximately `+0.04` (0.2m × 1.0 scaling). The entropy bonus for maintaining randomness is `-0.05 × entropy_change`. The entropy bonus is comparable in magnitude to the task reward, drowning out the learning signal.

### Problem 3: Curriculum Never Triggered
The CurriculumCallback requires 85% success rate. Since the agent never completes a single delivery (reward never exceeds +5.0), the curriculum **never upgraded past Level 1**. The entire 8.9M steps were spent on Level 1's fixed-position, zero-obstacle scenario — and it still couldn't solve it.

### Problem 4: n_steps Too Small for LSTM
With `n_steps = 4096 // num_cpu` and 16 CPUs, each environment only collects `256` steps per rollout. LSTMs need long, uninterrupted sequences to build meaningful hidden states. A sequence length of 256 is too short for the LSTM to learn temporal dependencies across a 500-step episode.

---

## 5. What Needs to Change (Recommendations)

| Fix | Current | Recommended | Why |
|:---|:---|:---|:---|
| `ent_coef` | 0.05 | **0.005** | 10x reduction. Allow the network to become deterministic and exploit learned behavior. |
| `n_steps` | 256 per env | **2048 per env** | LSTM needs long sequences. Set `n_steps=2048` (not divided by num_cpu). |
| Reward scaling | 1.0x | **5.0x** | Multiply PBRS by 5.0 so the task signal dominates the entropy bonus. |
| Pickup reward | +2.0 | **+5.0** | Increase the incentive to attempt pickup. |
| Curriculum threshold | 85% | **70%** | Lower the bar so the agent can progress even with imperfect behavior. |
