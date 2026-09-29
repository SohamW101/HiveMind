# HiveMind: Slide Content

---

## Slide 1: Title

# HiveMind
### Teaching a team of warehouse robots to cooperate with multi-agent reinforcement learning

---

## Slide 2: Problem Statement

**The problem**
- Warehouses need several robots collecting items and bringing them to a drop-off point.
- Robots share narrow aisles, compete for the same items and get in each other's way.
- Hand-coded coordination breaks as soon as the layout changes.

**Our goal**
- 4 robots, one warehouse: find the cartons, pick them up, deliver them to the depot, as a team.
- No map is given. The layout is shuffled every episode, so robots must discover the aisles through their own sensors.
- Finish fast, and avoid collisions.

**Our approach**
- Robots *learn* the behaviour through reinforcement learning, with no hand-written rules.

---

## Slide 3: The Environment

- A simulated warehouse built in **PyBullet** (a physics engine).
- **The task:**
  - A grid floor with rows of shelves.
  - 12 cartons placed in random gaps between shelves.
  - A depot in one corner.
- **The robots:** 4 differential-drive robots, each with a gripper arm and a LiDAR sensor.
- **Actions:** move forward/back, turn, pick up, drop off, wait.
- **Partial observability:**
  - no map, and a different layout every episode;
  - obstacles are only seen through a limited-range, noisy LiDAR.
- An episode ends when all cartons are delivered or time runs out.

`[Image: top-down screenshot of the warehouse with the 4 robots]`

---

## Slide 4: RL Architecture

- **Formulation:** cooperative multi-agent RL as a **Dec-POMDP**.
  - The robots share a team reward, and each acts only on its own local observation.
- **Parameter sharing, decentralised execution:**
  - One actor-critic network is shared by all 4 agents.
  - A custom SB3 `VecEnv` adapter presents each 4-robot world as 4 single-agent slots.
- **Algorithm:** **Recurrent PPO** (`sb3-contrib`).
  - Clipped surrogate objective, **GAE** advantages and an entropy bonus.
  - An **LSTM** hidden state gives the policy memory under partial observability.
- **Observation:** a fixed-size **177-d vector** per agent.
  - Own pose and velocity, teammate poses, carton status and positions, depot offset.
  - A **72-ray, 270° LiDAR** with Gaussian noise.
  - **48 reserved message slots:** 3 teammates × 16-token one-hot.
- **Action space:** `MultiDiscrete([7, 16])`, i.e. 7 motion/manipulation actions plus a 16-token discrete message.
- **Feature extractor (custom PyTorch module), three branches, fused and passed to the LSTM:**
  - **MLP** over world-state features.
  - **1-D CNN** (strided convolutions) over the LiDAR sweep, which exploits its angular structure.
  - **Multi-head self-attention** over the incoming messages, max-pooled so sender order doesn't matter.
- **Separate actor and critic extractors,** so value gradients don't interfere with the policy representation.
- **Critic:** decentralised (per-agent observation). A centralised MAPPO-style critic is a future upgrade.

`[Diagram: 177-d obs → MLP | 1-D CNN | MHA → concat → LSTM → π (7×16) / V]`

---

## Slide 5: Reward Design and Training

- **Reward: a team/individual split** (0.8 shared, 0.2 individual).
  - Shared terms: per-delivery reward, a completion bonus and a **makespan bonus**, minus a time penalty and collision costs.
  - Individual terms: own pick-ups and deliveries, invalid actions, and idling.
- **Potential-based reward shaping** (Ng, Harada & Russell, 1999):
  - F = Φ(s′) − Φ(s), with Φ = −(work remaining).
  - It **telescopes** over an episode, so it can't be farmed and doesn't change which policy is optimal.
- **Credit assignment:** asymmetric collision blame. Only the agent that moved into another is penalised.
- **Training setup:**
  - **Curriculum learning:** 1 → 12 cartons, promoted automatically on rolling success rate.
  - **Parallel simulation:** multi-process `SubprocVecEnv`, 8 worlds = 32 agent streams.
  - Reward normalisation with `VecNormalize`, and long-horizon discounting (γ = 0.999).
- **Tech stack:** Python · PyBullet · Gymnasium · PyTorch · Stable-Baselines3 / sb3-contrib · TensorBoard

---

## Slide 6: Results

`[Graph: presentation_graphs/1_success_rate.png]`
`[Graph: presentation_graphs/3_episode_length.png]`
`[Graph: presentation_graphs/6_eval_success.png]`

- **13.5 M environment steps.** The curriculum reached the full 12-carton task at 2.56 M.
- **Evaluation (deterministic policy, 50 episodes per level):**
  - **100% success at 12 cartons.**
  - 84% at 8 cartons.
  - 18% at 4 cartons, a curriculum-overfitting effect.
- **Makespan** (steps to finish) kept falling throughout training.
- **Critic convergence:**
  - Explained variance went from 0.41 to **0.999**.
  - Value loss fell from 2.5 to 0.003.
- **Policy updates grew aggressive late in training:**
  - Approximate KL divergence ~0.06, clip fraction ~0.36.
  - Lesson learned: PPO step-size control, e.g. `target_kl` or a faster learning-rate decay.

*Backup graphs:* `2_cartons_delivered.png`, `4_episode_reward.png`, `5_curriculum.png`

---

## Slide 7: Demo

`[Video: 4 robots completing the 12-carton task]`

---

## Slide 8: Challenges and How We Solved Them

| Challenge | Solution |
|---|---|
| Robots learned to just stand still: moving risked collisions long before any reward | Progress-based reward shaping so every step of real progress pays |
| Picking up a carton was accidentally *punished* | Redesigned the progress measure so pick-ups and deliveries always count as progress |
| Constant shelf crashes dominated learning | Made shelves solid: a blocked move is simply refused |
| The team never finished an episode, so never saw the completion bonus | Curriculum starting from a single carton |
| Episodes too long, so the final reward was invisible to the learner | Episode length scaled to the task size |
| LiDAR slowly started seeing *through* shelves | Found the robot body sinking under gravity; fixed its height |
| Training was too slow | One process per warehouse, with lighter physics |
