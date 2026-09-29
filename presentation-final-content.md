# HiveMind: Slide Content

---

## Slide 1: Title

# HiveMind
### Multi-agent reinforcement learning and emergent communication for cooperative warehouse robots

---

## Slide 2: Problem Statement

**The problem**
- Warehouses use fleets of robots to collect items and bring them to a drop-off point.
- Robots share narrow aisles, compete for the same items and block each other.
- Hand-coded coordination breaks when the layout changes.

**Our goals**
1. **Cooperative MARL:** 4 robots learn to find, pick up and deliver 12 cartons to a depot as a team.
   - No map is given, and the layout is randomised every episode.
   - Obstacles are sensed only through LiDAR.
2. **Emergent communication:** give the robots a discrete message channel and study whether they learn to use it.
   - Ablation: *no comm* vs *comm*.

**Metrics:** success rate, makespan (steps to finish), collisions.

---

## Slide 3: Environment

- **Simulator:** PyBullet rigid-body physics, wrapped as a Gymnasium environment.
- **World:**
  - 13 × 13 grid with 6 shelf rows.
  - Randomised gaps between shelves each episode.
  - 12 cartons and a corner depot.
- **Agents:** 4 differential-drive robots (URDF) with a gripper arm and a 2-D LiDAR (72 rays, 270°, noisy).
- **Actions:** forward, backward, turn left/right, pick up, drop off, stay.
- **Partial observability:** no map, and obstacles are visible only within LiDAR range.
- **Termination:** all cartons delivered, or the step cap.

`[Image: top-down screenshot of the warehouse]`

---

## Slide 4: RL Architecture (1/5): MARL Formulation

- **Cooperative Dec-POMDP:**
  - Shared team objective.
  - Each agent acts on its own local observation.
- **Parameter sharing:** one policy network for all 4 agents; the agents stay distinct through their own observations.
- **Decentralised execution:** each robot runs the shared network on its own input.
- **Bridge to single-agent RL:** a custom Stable-Baselines3 `VecEnv` presents each 4-robot world as 4 agent slots, so a standard PPO trainer does multi-agent training.
- **Critic:** decentralised (per-agent input). A centralised MAPPO-style critic is the next upgrade.

`[Diagram: 1 world → 4 agent slots → shared policy → 4 actions]`

---

## Slide 5: RL Architecture (2/5): Observation and Action Spaces

**Observation: a fixed 177-d vector per agent**

| Block | Contents |
|---|---|
| Ego state | pose, velocity, carrying flag |
| Teammates | poses and carrying flags |
| Task state | carton status and positions, depot offset, elapsed time |
| LiDAR | 72 normalised range readings |
| Messages | 3 teammates × 16-token one-hot (48) |

**Action: `MultiDiscrete([7, 16])`**
- One of 7 movement/manipulation actions, plus one of 16 message tokens, chosen every step.

**Fixed observation width:** message slots are reserved in both arms, so the comm and no-comm models have identical input/output shapes.

---

## Slide 6: RL Architecture (3/5): Network and Algorithm

**Custom feature extractor (PyTorch), three branches:**
- **World features** → MLP.
- **LiDAR sweep** → 1-D CNN (strided convolutions) that exploits the angular structure.
- **Messages** → multi-head self-attention over the 3 senders, max-pooled so sender order doesn't matter.
- The branches are concatenated into a shared embedding.

**Recurrent PPO** (`sb3-contrib`):
- An **LSTM** on top of the embedding gives memory under partial observability.
- Clipped surrogate objective, GAE advantages and an entropy bonus.
- **Separate actor and critic extractors**, so value gradients don't interfere with the policy.

`[Diagram: 177-d obs → MLP | 1-D CNN | MHA → concat → LSTM → policy heads (7, 16) / value]`

---

## Slide 7: RL Architecture (4/5): Emergent Communication

- **Channel:** 16 discrete tokens, broadcast every step.
  - Each robot hears the other 3; teammates receive a token the next step.
- **Emergent:** no meaning is assigned to any token. Whether to use the channel, and what each token means, must be learned purely from the team reward.
- **Receiver:** multi-head attention over incoming messages.
- **Ablation design, one variable changed:**
  - **No comm:** tokens are produced, but the message slots are held at zero, so the channel is closed.
  - **Comm:** tokens are delivered to teammates.
  - Same network, action space, reward and hyperparameters.
- **Why hyperparameters stay fixed:** tuning only the comm run would confound the comparison.

---

## Slide 8: RL Architecture (5/5): Reward Design and Training

**Reward:** 0.8 × shared team reward + 0.2 × individual reward.
- **Shared terms:**
  - per-delivery reward;
  - completion bonus;
  - makespan bonus;
  - time penalty;
  - collision cost.
- **Individual terms:** own pick-ups and deliveries, invalid actions, idling.
- **Potential-based reward shaping** (Ng et al., 1999):
  - F = Φ(s′) − Φ(s), with Φ = −(work remaining).
  - It telescopes, so it can't be farmed and doesn't change which policy is optimal.
- **Credit assignment:** asymmetric collision blame. Only the agent that moved takes the individual penalty.

**Training setup:**
- **Curriculum:** 1 → 2 → 4 → 8 → 12 cartons, promoted on rolling success rate.
- **Parallel simulation:** multi-process `SubprocVecEnv`, 8 worlds = 32 agent streams.
- `VecNormalize` reward normalisation.
- γ = 0.999.

**Tech stack:** Python · PyBullet · Gymnasium · PyTorch · Stable-Baselines3 / sb3-contrib · TensorBoard

---

## Slide 9: Results (1/3): Training

`[Graph: presentation_graphs/t1_success_rate_compare.png]`
`[Graph: presentation_graphs/t2_episode_length_compare.png]`

- **Both arms learned the full task with the same curriculum.**
  - The comm run reached 12 cartons at ~2.2 M steps; no-comm at ~2.6 M.
  - Training success rate converged to ~100% in both.
- **Critic converged in both:** explained variance ≈ 0.999, value loss → ~0.001.
- **Training budget differs:** no comm 13.5 M steps, comm 30 M steps (19.2 h).
  - At equal steps (~13.5 M), episode length is about the same (~200 steps).
  - The comm run kept improving to ~123 steps by 30 M.
- **PPO diagnostics:**
  - Approx. KL peaked ~0.09 and clip fraction ~0.39 mid-run: aggressive updates.
  - Both decay to 0 as the linear learning-rate schedule reaches 0.
  - The late rise in policy loss is expected at that point, not divergence.

*Backup:* `t3_episode_reward_compare.png`, `t4_entropy_compare.png`, `t5_approx_kl_compare.png`, `t6_explained_variance_compare.png`, `5_curriculum.png`

---

## Slide 10: Results (2/3): Evaluation, No Comm vs Comm

`[Graph: presentation_graphs/e1_success_by_cartons.png]`
`[Graph: presentation_graphs/e2_makespan_12.png]`
`[Graph: presentation_graphs/e3_collisions_12.png]`

100 headless episodes per setting, deterministic policy, identical warehouse layouts (fixed seeds) for both models.

| 12 cartons (full task) | No comm | Comm |
|---|---|---|
| Success rate | 100% | 100% |
| Steps to finish (mean / median) | 143 / 133 | **110 / 108** |
| Collision events per episode | 9.4 | **6.8** |
| Invalid actions per episode | 93 | **33** |

- **Full task:** comm finishes **~23% faster** with ~28% fewer collisions.
- **Fewer cartons:** both models are much weaker (curriculum overfitting to the 12-carton task).
  - Comm is better at 4 cartons (39% vs 26%).
  - No comm is better at 8 cartons (84% vs 60%).

---

## Slide 11: Results (3/3): Did Communication Emerge?

`[Graph: presentation_graphs/e4_intervention_makespan.png]`
`[Graph: presentation_graphs/e5_token_usage.png]`
`[Graph: presentation_graphs/e6_token_by_state.png]`

- **Intervention test:** we zeroed the comm model's incoming messages.
  - Still 100% success.
  - 105 vs 110 steps: **no loss at all**.
- **Token statistics:**
  - Entropy is 3.80 of a maximum 4 bits: close to uniform.
  - Mutual information with "carrying or not" is only 0.013 bits.
- **Conclusion: the channel is open, but the policy does not rely on it.**
  - The gain over no-comm comes from longer training (30 M vs 13.5 M steps), not from messages.
- **What we learned:**
  - An open channel alone doesn't produce communication. With a shared policy and teammate poses already in the observation, messages add little new information.
  - Measuring emergent communication needs **causal tests** (interventions), not just performance.
  - **Next steps:**
    - hide teammate and carton information so that messages carry it;
    - add a listener or information bonus;
    - use a centralised critic;
    - compare at matched training budgets.

---

## Slide 12: Thank You

# Thank You
### Questions?
