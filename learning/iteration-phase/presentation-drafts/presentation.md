# HiveMind: Cooperative Multi-Robot Warehouse Logistics with Multi-Agent Reinforcement Learning and Emergent Communication

---

## 1. Problem Statement
- Warehouses rely on fleets of mobile robots (AGVs) to move goods from shelves to dispatch points.
- Coordinating many robots is hard:
  - they share narrow aisles and compete for the same items;
  - they can block each other;
  - a single blocked robot can stall the whole fleet.
- Classical approaches (central planners, hand-written rules and fixed message protocols) need a full map, engineered routing and re-tuning whenever the layout changes.
- **Questions:**
  1. Can a team of robots *learn* to divide the work, navigate an unknown layout and avoid each other, purely from experience?
  2. If we give them a message channel with **no predefined meaning**, do they learn to **communicate**?

## 2. Project Goals
- **Goal 1, cooperative MARL:** train **4 robots** to **find cartons, pick them up and deliver them to a shared depot**.
  - Coordination must emerge from learning, not from hand-written rules.
  - Robots operate under **partial observability**:
    - no map of the warehouse is provided;
    - the shelf layout is **randomised every episode**;
    - obstacles are perceived only through a **range-limited, noisy LiDAR**.
  - Complete the job **fast** and **with few collisions**.
- **Goal 2, emergent communication:**
  - Give every robot a **16-token discrete message channel**.
  - Test, as a controlled ablation (**no comm vs comm**), whether a useful protocol emerges from the team reward alone.
- **Metrics:**
  - task: success rate, makespan (steps to finish), collisions;
  - communication: token entropy, mutual information with robot state, and a message-intervention test.

## 3. Environment: The Simulated Warehouse
- **Simulator:** PyBullet rigid-body physics, wrapped as a Gymnasium environment.
- **Floor:** 13 × 13 m grid (1 m cells) enclosed by boundary walls.
- **Shelving:**
  - 6 solid shelf rows, on every other grid row.
  - Each row is split into 3 shelf units by **2 randomly placed gaps**, so aisle connectivity changes every episode.
  - Shelves are impassable, which makes robots route through the gaps.
- **Cartons:**
  - 12 cartons in total, one in each shelf gap.
  - Positions change with the randomised layout every episode.
- **Depot:** a fixed drop-off zone in one corner (grid cell 0,0).
- **Robots:** 4 robots spawn next to the depot at the start of every episode.
- **Episode ends when:**
  - all cartons are delivered (success), or
  - a step cap is reached. The cap scales with task size (60 / 90 / 150 / 250 / 400 steps for 1 / 2 / 4 / 8 / 12 cartons).
- **Why the environment suits emergent communication:** robots compete for the same cartons and aisles. Announcing intent ("I'm taking this carton", "I'm in this aisle") is exactly the kind of information a learned protocol could carry.

## 4. The Robot
- Differential-drive AGV modelled in URDF:
  - chassis and wheels;
  - a yaw-rotating arm with a two-finger gripper;
  - a LiDAR mast that rises while carrying.
- **Discrete action set (7 actions):** Forward, Backward, Turn Left, Turn Right, Pick Up, Drop Off, Stay.
- **Motion model:**
  - A move covers one grid cell; turns are 90°.
  - Motion is animated smoothly over physics sub-steps, then snapped to the grid for exact, drift-free state.
- **Pick-up:** succeeds only if a carton is within 1.5 cells.
- **Drop-off:** succeeds only within 1.5 cells of the depot.
- **Invalid actions are refused and penalised:** driving into a shelf or wall, grabbing at nothing, dropping away from the depot.
- **Speaker output:** besides moving, each robot emits **one of 16 discrete message tokens** every step, so every robot is both a *speaker* and a *listener*.

## 5. Perception and Partial Observability
**What robots are NOT given:**
- No map, occupancy grid or shelf layout. The layout is regenerated every episode and never appears in the observation.
- No path, route or planner output.
- Robots detect shelves and walls only when their LiDAR beams reach them.
- No predefined meaning for any message token.

**LiDAR sensor (per robot):**
- 72 rays across a **270° forward-facing arc**.
- Range 0.1–10 m.
- Gaussian noise: σ = 1 cm + 1% of range.
- Mounted at chassis height, so it sees exactly what the robot would collide with.

**Task state in the observation:**
- The robot's own pose, velocity and carrying flag.
- Poses and carrying flags of the 3 teammates.
- Status of each of the 12 cartons (available / carried by me / carried by another / delivered) and their positions.
- Offset to the depot and the fraction of the episode elapsed.

**Social information:** the last message token from each of the 3 teammates.

## 6. Observation Vector (177 values per robot)

| Component | Size | Encoding |
|---|---|---|
| Own pose (x, y, heading) | 3 | normalised to [-1, 1]; heading wrapped to [0, 1) |
| Own velocity | 2 | last-step displacement (cells/step) |
| Own carrying flag | 1 | 0 / 1 |
| Teammates' poses | 9 | 3 robots × (x, y, heading) |
| Teammates' carrying flags | 3 | 0 / 1 |
| Carton status | 12 | 0 available · ⅓ mine · ⅔ other's · 1 delivered |
| Carton positions | 24 | 12 × (x, y) |
| Depot offset | 2 | direction and distance |
| Elapsed time | 1 | step / max_steps |
| LiDAR scan | 72 | normalised ranges |
| **Message slots** | **48** | **3 teammates × 16-token one-hot** |
| **Total** | **177** | all values in [-1, 1] |

- **The observation size is pinned at 177 values.** Saved models only fit an environment with the same size, so any layout change must become a new version.
- Three safeguards enforce this:
  - an import-time check that the components sum to 177;
  - a constructor check that rejects any other size;
  - a runtime shape assertion.
- **Message slots exist in both ablation arms.**
  - In the no-comm arm they are held at zero; in the comm arm they carry teammates' tokens.
  - Input size, network and action space are therefore identical across the two arms.

## 7. Emergent Communication Channel
**Channel:**
- **Vocabulary:** K = 16 discrete tokens, one-hot encoded.
- **Topology:** broadcast. Each robot hears the other three robots, never itself.
- **Timing:** a token sent at step *t* appears in the teammates' observation at step *t + 1*.

**Speaker:** the policy has a second categorical head.
- Joint action per robot: aᵢ = (mᵢ, cᵢ), with mᵢ ∈ {1..7} a movement and cᵢ ∈ {1..16} a token.
- The action factorises as

  π(aᵢ | oᵢ) = π(mᵢ | oᵢ) · π(cᵢ | oᵢ)

  so log π(aᵢ | oᵢ) = log π(mᵢ | oᵢ) + log π(cᵢ | oᵢ) enters the PPO ratio.

**Listener:** a self-attention module over the three incoming messages.
- Each one-hot is embedded to 64-d.
- 4-head self-attention runs across the 3 senders.
- The result is max-pooled into a 64-d permutation-invariant message feature.

**Why "emergent":**
- There is **no reward for sending or reading messages**.
- A token affects the return only through how teammates react to it, so any protocol must be discovered purely from the shared team reward.
- This creates a **chicken-and-egg problem**: speakers have no incentive to send informative tokens until listeners react to them, and listeners have no reason to attend to tokens that carry no information.

**Ablation design (one variable changed):**

| | No comm | Comm |
|---|---|---|
| Tokens produced | yes | yes |
| Tokens delivered to teammates | **no** (slots held at 0) | **yes** |
| Network, action space, reward, curriculum, hyperparameters, seed | same | same |

## 8. Reward Design
The reward follows the MAWC technical specification, extended with potential-based shaping.

### 8.0 Reward tables (as implemented in `hivemind_env/env.py`)

**Shared team reward (weight 0.8, identical for all robots):**

| Event | Reward |
|---|---|
| Carton delivered (by anyone) | +10 per carton |
| All cartons delivered | +100 (once per episode) |
| Makespan bonus | +50 × (T_max − T) / T_max (once, on completion) |
| Collision event (robot–robot or robot–wall/shelf), team share | −1 per event |
| Time penalty | −0.05 per step |

**Individual reward (weight 0.2, per robot):**

| Event | Reward |
|---|---|
| Own pick-up | +1 |
| Own delivery | +2 |
| Collision the robot caused (moved into another robot) or hitting a wall/shelf | −5 per event |
| Invalid action (blocked move, pick-up with nothing in reach, drop away from depot, pick/drop in the wrong carrying state) | −0.5 |
| Idle (speed < 0.1 cell/step, not at the depot; turning exempt) | −0.02 per step |

**Potential-based shaping (added on top, outside the 0.8 / 0.2 split):**

| Term | Value |
|---|---|
| Shaping reward | F = 30 × (Φ(s′) − Φ(s)) every step |
| Potential | Φ = −(cartons remaining + 0.5 if not carrying + distance to objective / (2 × 13 m)) |

**Total per robot per step:** r = 0.8 × shared + 0.2 × individual + F.

**Defined but not used:** the specification's replanning penalty (−0.1) has no trigger, because there is no path planner in this environment.

The subsections below give each term's formula, its effective weighted value and why it exists.

### 8.1 Overall structure
For robot *i* at step *t*:

  **rᵢ(t) = w_s · R_shared(t) + w_ind · R_ind,i(t) + Fᵢ(t)**, with w_s = 0.8 and w_ind = 0.2

- **R_shared:** identical for all four robots. It makes the task genuinely cooperative: every robot benefits from every delivery.
- **R_ind,i:** robot-specific. It supplies a per-agent learning signal and counters the *lazy agent* problem, where one robot free-rides on the team reward.
- **Fᵢ:** potential-based shaping. It sits **outside** the 0.8 / 0.2 split, so its scale means exactly what it says.
- **No communication term:** there is no reward for messaging (see §7). Communication can only matter through R_shared.

### 8.2 Shared team reward
  **R_shared(t) = 10·Dₜ + 𝟙[done]·(100 + 50·(T_max − T)/T_max) − 1·Cₜ − 0.05**

| Term | Formula | Effective value per robot (× 0.8) | Purpose |
|---|---|---|---|
| Per-delivery | +10 · Dₜ (Dₜ = cartons delivered this step, by anyone) | +8 per carton | Dense-ish progress signal |
| Completion | +100 when all cartons are delivered | +80 | The actual objective |
| Makespan bonus | +50 · (T_max − T) / T_max on the final step | up to +40 | Rewards finishing *faster*: 0 at the step cap, 50 for instant completion |
| Collision (team share) | −1 · Cₜ (Cₜ = new collision events this step) | −0.8 per event | Everyone shares the cost of congestion |
| Time penalty | −0.05 per step | −0.04 per step | Pressure to finish; over 400 steps ≈ −16 |

- **Collision events** count robot–robot contacts and robot–wall/shelf contacts.
  - Each is counted **once per event** (on the step contact begins), not once per step for as long as two robots stay touching.
- **Makespan:** T is the step the job finished on; T_max is the episode's step cap.

### 8.3 Individual reward
  **R_ind,i(t) = 1·pickᵢ + 2·delivᵢ − 0.5·invalidᵢ − 5·collᵢ − 0.02·idleᵢ**

| Term | Condition | Effective value (× 0.2) | Purpose |
|---|---|---|---|
| Own pick-up | robot *i* picked up a carton | +0.2 | Credit to the robot that did the work |
| Own delivery | robot *i* delivered a carton | +0.4 | As above |
| Invalid action | move into shelf/wall, pick up with nothing in reach, drop away from depot, or pick/drop in the wrong carrying state | −0.1 | Teaches the action preconditions |
| Collision | robot *i* hit an obstacle, or moved into another robot | −1.0 per event | Blame on the robot that caused it |
| Idle | ‖vᵢ‖ < 0.1 cell/step, not at the depot, not turning | −0.004 per step | Discourages parking in aisles |

**Asymmetric collision credit assignment:**

  collᵢ = #(obstacle events of i) + #(robot events involving i where aᵢ ∈ {Forward, Backward})

- A stationary robot that gets hit pays only the small shared share, not the −5.
- This removes the incentive to "freeze" out of fear of being hit.

**Idle rule:** turning in place has zero linear speed but is necessary for navigation, so turns are exempt from the idle penalty.

### 8.4 Potential-based reward shaping
The specification's reward alone is too sparse: nothing pays until a carton is picked up, and the big terms only pay on completion. We add **potential-based shaping** (Ng, Harada & Russell, 1999):

  **Fᵢ(t) = κ · (Φᵢ(s_{t+1}) − Φᵢ(s_t))**, κ = 30

with the potential defined as **minus the work remaining, measured in cartons**:

  **Φᵢ(s) = −( N_left + 0.5 · (1 − carryᵢ) + dᵢ / (2L) )**

- N_left: cartons not yet delivered.
- carryᵢ ∈ {0, 1}: whether robot *i* holds a carton.
- dᵢ: Euclidean distance to the robot's current objective: the depot if carrying, otherwise the nearest available carton.
- L = 13 m: the arena span, so dᵢ / (2L) ≤ 0.5.

**Properties:**
1. **Telescoping, so it can't be farmed.**
   - Summed over an episode, ΣFᵢ = κ(Φᵢ(s_T) − Φᵢ(s_0)).
   - That depends only on the start and end states, never on the path. Circling or hovering earns exactly zero net shaping.
2. **Every sub-task transition is rewarded, never punished.**
   - Pick-up: carry flips 0 → 1 and the objective switches to the depot.

     ΔΦ = 0.5 + (d_carton − d_depot)/(2L) ≥ 0.5 − 0.5 = 0

   - Delivery: N_left drops by 1, carry flips 1 → 0, and the objective switches to the next carton.

     ΔΦ = 1 − 0.5 + (d_depot − d_next)/(2L) ≥ 0

   - An earlier distance-only potential jumped *down* at pick-up, so completing a pick-up gave **negative** total reward. The 0.5 "hand-off" term fixed it.
3. **Dense gradient:**
   - each one-cell move toward the objective pays up to κ/(2L) ≈ **+1.15**;
   - that dominates the expected collision risk of moving (≈ 10% × −1.8 early in training);
   - so moving is worth more than standing still.
4. **γ = 1 inside the shaping term (a deliberate deviation):**
   - The textbook form γΦ(s′) − Φ(s) adds a per-step drift of −(1−γ)Φ.
   - Φ < 0 here, so that drift is a *positive* bonus for loitering far from the goal.
   - Using γ = 1 removes it, at the cost of strict policy invariance under discounting.

### 8.5 How the reward is used in training
- **Return being optimised:** Gᵢ = Σₜ γᵗ rᵢ(t) with γ = 0.999.
  - Over a 400-step episode, terminal rewards keep 0.999⁴⁰⁰ ≈ 0.67 of their value.
  - With γ = 0.99 and the original 2000-step episodes they kept only 0.99²⁰⁰⁰ ≈ 2·10⁻⁹: invisible.
- **Advantage estimation:** GAE(λ = 0.95).

  Âₜ = Σₗ (γλ)ˡ δₜ₊ₗ, where δₜ = rₜ + γV(oₜ₊₁) − V(oₜ)

- **Reward normalisation:** `VecNormalize` divides rewards by a running estimate of the return's standard deviation, which keeps the value targets well-scaled (this is why value losses of ~10⁻³ are normal here).
- **Episode cap sized to the task,** so the time penalty and makespan bonus stay meaningful at every curriculum level.

### 8.6 Worked example: one delivery (effective per-robot values)
Robot 2 drops a carton at the depot; 5 cartons remain and nothing collides.

| Component | Robot 2 (deliverer) | Other robots |
|---|---|---|
| Shared: per-delivery (+10 × 0.8) | +8.0 | +8.0 |
| Shared: time (−0.05 × 0.8) | −0.04 | −0.04 |
| Individual: own delivery (+2 × 0.2) | +0.4 | 0 |
| Shaping: N_left −1, hand-off +0.5, new objective | +κ·ΔΦ ≥ 0 | +κ·(1 + Δd/(2L)) ≥ 0 |

Every robot is rewarded for the team's progress, and the deliverer gets a little extra credit.

## 9. Learning Architecture

**Formulation:** a cooperative **Dec-POMDP**. There is one team objective; each agent sees only its own observation oᵢ, including its teammates' messages.

**Paradigm:** parameter sharing, decentralised execution.
- One policy network π_θ(aᵢ | oᵢ, hᵢ) controls all 4 robots, where hᵢ is each robot's own LSTM state.
- The same weights produce both **speaker** (token) and **listener** (message attention) behaviour.

**Algorithm:** Recurrent PPO (sb3-contrib), with an LSTM for memory under partial observability. The clipped objective is

  **L(θ) = E[ min(ρₜ Âₜ, clip(ρₜ, 1−ε, 1+ε) Âₜ) ] − c_v · E[(V_θ(oₜ) − V̂ₜ)²] + c_e · E[H(π_θ(·|oₜ))]**

- ρₜ = π_θ(aₜ|oₜ) / π_θ_old(aₜ|oₜ), with ε = 0.2, c_v = 0.5, c_e = 0.02.
- With the factorised action, ρₜ = ρₜ^move · ρₜ^token and H = H_move + H_token. The entropy bonus therefore also keeps the **token distribution** exploratory.

**Feature extractor (HiveMindExtractor), three branches:**
- **World branch:** pose, teammates, cartons, depot and time → MLP (128 → 128).
- **LiDAR branch:** 72 rays → 1-D CNN (two stride-2 convolutions, 32 channels) that detects obstacle patterns at every bearing.
- **Message branch (listener):** 3 × 16 one-hot tokens → linear embedding (64-d) → 4-head self-attention across senders → max-pool → 64-d.
- The branches are concatenated and projected to 256-d, then passed through an LSTM (256).
- **Heads:**
  - policy: movement (7-way) and **token (16-way)** categorical heads;
  - value: a scalar V(o).
- Actor and critic have **separate** feature extractors, so value-function gradients don't disturb the policy representation.
- **Critic:** decentralised (per-agent observation). A centralised critic over the joint state is a future upgrade.

**Model size:** ≈ 1.64 M parameters, identical in both ablation arms.

## 10. Training Pipeline

**Multi-agent to single-agent bridge:**
- A custom vectorised environment presents each 4-robot warehouse as 4 policy slots to Stable-Baselines3.
- All 4 robots act on the same physics step; episodes end and reset together per warehouse.
- In the comm arm, each slot's (move, token) pair is routed into the joint action, and tokens are written into teammates' message slots.

**Parallelism:**
- Each warehouse runs in its own OS process (`SubprocVecEnv`).
- Training is ~95% physics-bound, so this gives ~4.3× throughput.
- 8 parallel warehouses give 32 robot streams.

**Curriculum (1 → 2 → 4 → 8 → 12 cartons):**
- Start at 1 carton so that finishing an episode, and the +100 / makespan bonuses, can happen by chance.
- Move up a level when the rolling success rate reaches 85%.
- The step cap grows with each level.

**Hyperparameters (identical for no comm and comm):**

| Parameter | Value |
|---|---|
| Algorithm | RecurrentPPO, MlpLstmPolicy |
| Rollout | 512 steps/slot × 32 slots = 16,384 |
| Mini-batch | 4,096 |
| Epochs per update | 10 |
| Learning rate | 3e-4, linear decay |
| γ / GAE λ | 0.999 / 0.95 |
| Clip range ε | 0.2 |
| Entropy coefficient c_e | 0.02 |
| Value coefficient c_v / max grad norm | 0.5 / 0.5 |
| Reward normalisation | VecNormalize (rewards only) |
| Seed | 0 |

**Hardware:** NVIDIA RTX A5000 server.

## 11. Training Results (no comm vs comm)
Both runs are compared over the same range of training.

**Graphs:**
- `t1_success_rate_compare.png`
- `t2_episode_length_compare.png`
- `t7_curriculum_compare.png`
- `t8_total_loss_compare.png`
- `t9_value_loss_compare.png`
- `t10_policy_loss_compare.png`
- (backup) `t3_episode_reward_compare.png`, `t4_entropy_compare.png`, `t5_approx_kl_compare.png`, `t6_explained_variance_compare.png`

| At the same point in training | No comm | Comm |
|---|---|---|
| Training success rate | 96.8% | 99.6% |
| Episode length (steps) | 208 | 187 |
| Mean episode reward | 531 | 542 |
| Critic explained variance | 0.999 | 0.999 |

**Takeaways:**
- **Both arms learn the full task through the same curriculum.** The comm run reached the 12-carton stage and ≥90% success sooner.
- **Critic convergence:** explained variance → 0.999 and value loss falls by ~3 orders of magnitude in both runs.
- **Loss curves:**
  - The total loss and the clipped-surrogate (policy-gradient) loss drop sharply in the first ~2–3 M steps, then plateau.
  - The value loss falls by about three orders of magnitude.
  - In every loss the comm run's curve drops earlier, consistent with it passing the curriculum stages sooner.
  - PPO losses measure optimisation progress, not task performance; success rate and makespan are the performance metrics.
- **Single run per arm:** these differences are observations, not statistically established effects.

## 12. Evaluation Results
100 headless episodes per setting, deterministic policy, **identical warehouse layouts for both arms** (fixed seeds).

| 12 cartons (full task) | No comm | Comm |
|---|---|---|
| Success rate | 100% | 100% |

**Graphs:** `e4_intervention_makespan.png`, `e5_token_usage.png`, `e6_token_by_state.png`

### 12.1 Did communication emerge?
**Intervention test:** we zeroed all incoming messages for the comm model at test time.
- Success stays at 100%, with no increase in steps to finish.
- The policy does **not** rely on the channel.

**Token statistics:**
- Entropy: H(token) = 3.80 of a maximum log₂16 = 4 bits, close to uniform.
- Mutual information with robot state:

  I(token; carrying) = Σ p(c, k) log₂ [p(c, k) / (p(c) p(k))] = **0.013 bits**

  Tokens are nearly independent of what the robot is doing.

**Conclusion:** the channel is open, but no informative protocol emerged.

### 12.2 Why, and what we learned
- **Redundancy:** teammates' poses, carrying flags and carton statuses are already in every observation. Messages have little new information to add.
- **Chicken-and-egg:** with no direct incentive, speaker and listener must co-adapt from noise. A shared-reward gradient through a teammate's reaction is weak.
- **Measurement:** task performance alone can't show communication. Causal tests (interventions) and information measures (entropy, mutual information) are needed.

## 13. Key Engineering Challenges and Solutions

| Problem observed | Root cause | Fix |
|---|---|---|
| Robots learned to stand still ("idle freezing"); a 5 M-step run scored below doing nothing | Sparse reward: moving risked collisions long before any delivery reward | Potential-based shaping (κ = 30), tuned so a move is worth more than its collision risk |
| Pick-ups gave *negative* total reward | Distance potential jumped when the objective switched from carton to depot | Added the 0.5 "hand-off" term and the cartons-remaining term, so every pick-up and delivery raises Φ |
| Shaping was 10× weaker than intended | Shaping was added *inside* the 0.2 individual bucket | Moved F outside the 0.8 / 0.2 split |
| ~94 shelf collisions/episode dominated learning | Robots could drive into shelves and were just charged for it | Shelves made impassable; the move is refused with a −0.5 invalid-action penalty |
| Stationary robots penalised when hit | Symmetric collision penalty | Only the robot that moved takes the −5 |
| Terminal +100 bonus never observed | At 4 cartons a random team never finishes, so the critic never saw success | Curriculum starts at 1 carton |
| Terminal rewards invisible to the learner | 2,000-step episodes: 0.99²⁰⁰⁰ ≈ 2·10⁻⁹ | Step caps sized to the task; γ = 0.999 |
| Curriculum had no effect | Promotion changed a variable the world never read | Promotion now sets the carton count and step cap |
| LiDAR saw "through" shelves after ~250 steps | Chassis slowly sank under gravity, lowering the beam below the shelf plate | Height snapped every step; beam fixed at 0.17 m |
| LiDAR hit the robot's own wheels | Rays started inside the chassis | Rays start 0.28 m from the centre |
| Training too slow (one core) | Physics-bound, sequential stepping | One process per warehouse (~4.3×); 5 physics sub-steps instead of 30 (~5×, identical outcomes) |
| Critic destabilising the policy | Shared extractor between actor and critic | Separate feature extractors |
| Comm vs no comm not comparable if the network changes | Adding message inputs later would change the input size | Message slots and token head reserved from the start; the no-comm arm holds slots at zero |

## 14. Technology Stack
- **Simulation:** PyBullet, URDF robot, shelf and carton models.
- **RL interface:** Gymnasium.
- **Learning:** Stable-Baselines3 + sb3-contrib (RecurrentPPO), PyTorch (custom extractor with 1-D CNN and multi-head attention).
- **Monitoring and analysis:** TensorBoard, NumPy, Matplotlib (token entropy and mutual-information analysis).
- **Language:** Python.

## 15. Repository Structure
- `hivemind_env/env.py`: warehouse environment, observation, reward, LiDAR, physics, message channel.
- `hivemind_env/models.py`: three-branch feature extractor with message attention.
- `hivemind_env/vec_env.py`, `subproc_vec_env.py`: shared-policy vectorisation (in-process and multi-process), including (move, token) routing.
- `hivemind_env/training.py`: curriculum and metrics callbacks, schedules, model loading.
- `scripts/train.py`: training entry point (`--communication` enables the channel).
- `scripts/inference.py`, `scripts/evaluate_all.py`: evaluation and video rendering.
- `scripts/make_presentation_graphs.py`: training comparison and communication-analysis graphs.
- `scripts/diagnostics/`: reward, observation and incentive verification tools.
- `tests/`: environment (including message propagation) and post-training test suites.
- `models/`, `tensorboard_logs/`: trained weights and training logs for both arms.

## 16. Limitations
- Motion is grid-based: cell-sized moves and 90° turns, not continuous velocity control.
- Carton positions and teammates' poses are part of the observation; only the map and obstacles are hidden. This redundancy weakens the incentive to communicate.
- The critic is decentralised (each robot's own view), not a centralised critic over the joint state.
- One training run per arm; results are not averaged over seeds.

## 17. Future Work
1. **Make communication necessary:** reveal cartons and teammates only within sensor range, so messages must carry what each robot has seen.
2. **Communication incentives:** positive-listening / positive-signalling losses or a mutual-information bonus to break the speaker–listener chicken-and-egg problem.
3. **Centralised critic (MAPPO / CTDE):** give the critic the joint state for better credit assignment of messages and actions.
4. **Multiple seeds** per arm for statistically robust comparisons.
5. **Mixed-difficulty training:** randomise the carton count per episode for robustness across task sizes.
6. **Continuous control and Sim2Real:** velocity-controlled differential drive, then transfer to physical robots via ROS 2.
