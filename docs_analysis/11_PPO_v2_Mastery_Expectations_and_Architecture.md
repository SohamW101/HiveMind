# PPO v2 (Mastery Update) — Full Architecture Changes and Expectations

This document serves as the definitive record of the changes made from `PPO_v1` to `PPO_v2`, the exact logic behind them, and what we expect to see during the 20-million step training run.

---

## 1. Full Architecture Changes (The "Mastery" Update)

We kept the Standard PPO algorithm (avoiding the complexity of Masked PPO) but gave the agent significant environmental and sensory upgrades.

### Change A: Expanded Observation Horizon (The "Eyes")
*   **Previous:** 15x15 grid (Agent could only see ~7 meters ahead).
*   **New:** **21x21 grid** (Agent can now see ~10 meters ahead).
*   **Logic:** The 30% collision rate on Level 4 was caused by "aliasing" and myopia. The agent would turn a corner into a dense obstacle field and get trapped because it couldn't see the walls coming soon enough. Expanding the grid allows the agent to plan paths *around* large clusters before getting near them.

### Change B: Dual-Scale PBRS (The "Scent")
*   **Previous:** PBRS Scale = `5.0` at all times.
*   **New:** PBRS Scale = **`5.0`** (Seeking Resource) and **`10.0`** (Carrying Resource).
*   **Logic:** The demo run showed 30% of failures on Level 4 were timeouts *after* picking up the resource. The agent would get stuck in a "local minimum" corner. By doubling the reward pull (gravity) towards the depot once the agent is carrying the resource, it is incentivized to take more risks to break out of corners to reach the massive reward gradient.

### Change C: Dynamic CNN Brain Capacity
*   **Previous:** `features_dim = 256` and hardcoded 15x15 dummy grid.
*   **New:** `features_dim = 512` and dynamically sized grid processing.
*   **Logic:** Since the input grid grew from 225 cells to 441 cells, we doubled the neural network's processing capacity so it has enough "memory" and parameters to process the wider spatial awareness effectively.

### Change D: Extended Training Budget
*   **Previous:** 10,000,000 steps.
*   **New:** **20,000,000 steps**.
*   **Logic:** The v1 agent spent 9.5 million steps on Level 4 and was still slowly improving when the run ended. Giving it a full 20M steps ensures it has the time required to solve the most difficult possible obstacle configurations.

---

## 2. What to Expect from the Live Training Run

When monitoring the TensorBoard logs (`localhost:6006`), here are the milestones and curves we expect to see:

### The Curriculum Trace (`curriculum/difficulty_level`)
*   **Expectation:** It should graduate through Levels 1, 2, and 3 even faster than last time. We expect it to reach **Level 4** before the 1 million step mark.
*   **Why?** The dual-scale PBRS makes finding the depot much easier, so the success rate will spike to 70% faster.

### The Reward Curve (`rollout/ep_rew_mean`)
*   **Expectation:** The peak reward will be significantly higher than v1.
*   **Why?** In v1, dropping off the resource yielded +10, plus some PBRS. In v2, the PBRS while carrying is doubled. We expect the mean reward to hit **+80 to +110** at its peak (compared to +60 in v1).

### The Episode Length (`rollout/ep_len_mean`)
*   **Expectation:** A sharp drop early on as it solves Levels 1 and 2, followed by a slight rise as it enters Level 4. It should stabilize around **150 - 200 steps** by the end of training.

### Value Loss (`train/value_loss`)
*   **Expectation:** Value loss will be noticeably **higher** than v1.
*   **Why?** Value loss scales with the square of the reward magnitude. Since our rewards are higher (due to dual-PBRS), the value loss will naturally scale up. This is healthy and expected.

### KL Divergence (`train/approx_kl`)
*   **Expectation:** We expect to see a spike in KL divergence (and `clip_fraction`) right when the agent enters Level 4, as it struggles to adapt to maximum obstacles. This will then smoothly decay toward `0.0` as the learning rate decays over the 20 million steps.

---

## 3. Final Goal

If the theory holds, when we run the demo evaluation script on the `ppo_hivemind_v2_final.zip` model, we expect:
*   **Level 1-3:** 90-100% Success.
*   **Level 4:** 80-90% Success (up from 40%).
*   **Collisions:** Near 0%.
