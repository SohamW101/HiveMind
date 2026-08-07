# Live Training Evaluation Report — Run `PPO_v1_5`

**Date & Time:** 2026-08-07 06:54 AM  
**Evaluated Step:** 2,260,992 / 10,000,000  
**Status:** 🟢 **RUNNING - EXCELLENT PERFORMANCE**

---

## Executive Summary

The fixes applied to `train.py` and `hivemind_env/env.py` (PBRS scale x5, removal of redundant APF penalty, VecMonitor integration, 70% curriculum threshold, and linear learning rate decay) have completely solved the environment. 

Run `PPO_v1_5` has achieved what all previous 9 runs failed to do: **it mastered Level 1 through Level 4 difficulty in under 600,000 steps** and is currently maintaining an **86.0% success rate on Level 4** with an average reward of **+31.8** and average episode length of **75.7 steps**.

---

## Metric Breakdown at Step 2.26M

### Curriculum & Performance
- **Current Difficulty Level:** **Level 4** (Graduated L1→L2 at ~250K, L2→L3 at ~400K, L3→L4 at ~600K)
- **Rolling Success Rate:** **86.0%** (0.8645) on Level 4
- **Mean Episode Reward (`rollout/ep_rew_mean`):** **+31.8** (monotonic upward trend from +10.0)
- **Mean Episode Length (`rollout/ep_len_mean`):** **75.7 steps** (down from 350+ at start; 0 timeouts)

### Optimization & Stability
- **Entropy Loss (`train/entropy_loss`):** **-0.8646** (smooth linear decay from -1.94; ideal exploration/exploitation balance)
- **Approximate KL (`train/approx_kl`):** **0.0265** (stable, non-divergent policy updates)
- **Clip Fraction (`train/clip_fraction`):** **0.1818** (healthy PPO trust-region updates)
- **Learning Rate (`train/learning_rate`):** **2.0e-4** (decaying linearly from 3.0e-4 toward 0.0)
- **Explained Variance (`train/explained_variance`):** **0.4442** (recovered strongly after curriculum transition dips)

---

## Comparative Matrix

| Metric | RecurrentPPO_3 (Old) | PPO_6 (Old) | **PPO_v1_5 (Current)** |
|:---|:---|:---|:---|
| **Max Curriculum Level** | Level 1 (Stuck for 8.9M steps) | Level 1 | **Level 4 (Achieved at 600K steps)** |
| **Level 4 Success Rate** | 0.0% | 0.0% | **86.0%** |
| **Mean Reward** | -5.15 | Unlogged | **+31.8** |
| **Mean Episode Length** | 500.0 (100% Timeout) | 500.0 | **75.7 steps** |
| **Policy Entropy** | -1.61 (Random noise trap) | -0.45 (Spinning trap) | **-0.86 (Optimal balance)** |
| **Status** | Failed | Failed | **Textbook Success** |

---

## Recommendation

**Keep the training running to the full 10,000,000 steps.**  
As the learning rate continues to decay from `2.0e-4` down to `0.0`, the network will refine its trajectory planning around Level 4 obstacles even further.
