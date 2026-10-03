# RL Training and Evaluation

Training four highly-coupled robots from scratch is notoriously difficult. The HiveMind training setup makes several calculated tradeoffs to make this viable.

## 1. Parameter Sharing vs. MAPPO

Currently, HiveMind uses **Parameter Sharing PPO** (often called "Cheap MAPPO"). 
- **The Setup**: There is one Actor network and one Critic network. `HiveMindSharedPolicyVecEnv` unwraps a 4-robot PyBullet world into 4 independent PPO "slots." PPO trains on the pooled experience as if it were observing 4 independent single-agent runs.
- **Why it works (for now)**: Because the observation vector is densely packed with global information (positions of all cartons and all other robots), the decentralized critic acts very similarly to a centralized critic. It has near-global knowledge.
- **When it will fail**: Parameter sharing struggles heavily when agents need highly differentiated roles. If the optimal strategy requires Robot 1 to exclusively act as a "fetcher" and Robot 2 to exclusively act as a "deliverer," a single shared Actor network will experience severe gradient conflict trying to learn both contradictory behaviors simultaneously.

## 2. Evaluation and The Greedy Baseline

Before training, the system runs a `greedy_baseline.json`. 
- **The Greedy Strategy**: Each robot finds the nearest available carton and pathfinds directly to it (ignoring other robots).
- **The Target**: On average, the greedy script clears 12 cartons in **97.6 steps**, with 6.7 collisions.
- **The Metric**: Any learned policy MUST beat an average makespan of 97 to be considered successful. Success rate is meaningless on its own (a policy could take 390 steps to succeed, which is far worse than a scripted baseline).

## 3. Curriculum Learning

The training utilizes a `CurriculumCallback` that scales difficulty.
- `Level 1`: 4 Cartons (Cap: 60 steps)
- `Level 2`: 8 Cartons (Cap: 90 steps)
- `Level 3`: 12 Cartons (Cap: 150 steps)
- **Why this is critical**: The episode terminates early if all cartons are collected. The `R_ALL_DELIVERED = +100.0` bonus and makespan bonus only apply at termination. A random policy in a 12-carton room will *never* stumble into 12 deliveries within the step limit, meaning it will never experience the massive +100 reward signal. By starting at 4 cartons, the policy learns the termination trigger, and the critic learns to value it.

## 4. Episode Step Limits and Gamma

Previously, the step limit was flat 2000. 
- With $\gamma = 0.99$, a reward of $+100$ at step 2000 is discounted back to the start state as $100 \times 0.99^{2000} \approx 0.0000001$.
- The critic physically could not see the value of completion. Capping the episodes tightly to the curriculum level ensures the discounted return remains mathematically significant.
