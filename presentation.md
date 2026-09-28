# HiveMind: Cooperative Multi-Robot Warehouse Logistics with Multi-Agent Reinforcement Learning

---

## 1. Problem Statement
- Warehouses rely on fleets of mobile robots (AGVs) to move goods from shelves to dispatch points.
- Coordinating many robots is hard:
  - they share narrow aisles and compete for the same items;
  - they can block each other;
  - a single blocked robot can stall the whole fleet.
- Classical approaches (central planners, hand-written rules) need a full map, engineered routing and re-tuning whenever the layout changes.
- **Question:** can a team of robots *learn* to divide the work, navigate an unknown layout and avoid each other, purely from experience?

## 2. Project Goal
- Train a team of **4 robots** to **find cartons, pick them up and deliver them to a shared depot** in a warehouse.
- Coordination must emerge from learning, not from hand-written rules.
- Robots operate under **partial observability**:
  - no map of the warehouse is provided;
  - the shelf layout is **randomised every episode**;
  - obstacles are perceived only through a **range-limited, noisy LiDAR**.
- Complete the job **fast** and **without collisions**.
- Lay the groundwork for **emergent communication** between robots, evaluated as an ablation.

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
  - a step cap is reached: 60 / 90 / 150 / 250 / 400 steps for 1 / 2 / 4 / 8 / 12 cartons, about 3× what a scripted baseline needs.

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
- **Communication output:** each robot can also emit **one of 16 discrete message tokens** every step.

## 5. Perception and Partial Observability
**What robots are NOT given:**
- No map, occupancy grid or shelf layout. The layout is regenerated every episode and never appears in the observation.
- No path, route or planner output.
- Robots detect shelves and walls only when their LiDAR beams reach them.

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
| Message slots | 48 | 3 teammates × 16-token one-hot |
| **Total** | **177** | all values in [-1, 1] |

- **The observation size is pinned at 177 values.** Saved models only fit an environment with the same size, so any layout change must become a new version.
- Three safeguards enforce this:
  - an import-time check that the components sum to 177;
  - a constructor check that rejects any other size;
  - a runtime shape assertion.
- Message slots are reserved now, so adding communication never changes the input size, and models with and without communication stay directly comparable.

## 7. Communication Channel
- **Vocabulary:** 16 discrete tokens (one-hot).
- **Delivery:**
  - Each robot hears the other three robots, never itself.
  - A token sent at step *t* appears in the teammates' next observation.
- **Action space with communication:** each robot outputs a pair (movement ∈ 7, token ∈ 16).
- **Receiver network:** a self-attention module (4 heads, 64-d) over the three incoming messages, max-pooled so the order of senders doesn't matter.
- **Status:**
  - The final model was trained with the communication action head present but **message slots held at zero**, so it is the **no-communication baseline**.
  - The **communication-enabled run is the ablation** that follows. Its size, architecture and hyperparameters are identical; the only difference is that messages are delivered.

## 8. Reward Design
Based on the MAWC technical specification.

**Shared team reward (weight 0.8, identical for all robots):**

| Event | Reward |
|---|---|
| Carton delivered (by anyone) | +10 |
| All cartons delivered | +100 |
| Makespan bonus | +50 × (T_max − T) / T_max |
| Collision event (team share) | −1 |
| Time penalty | −0.05 per step |

**Individual reward (weight 0.2, per robot):**

| Event | Reward |
|---|---|
| Own pick-up | +1 |
| Own delivery | +2 |
| Collision the robot caused (moved into another robot) or hitting an obstacle | −5 |
| Invalid action | −0.5 |
| Idle (not moving, away from the depot; turning exempt) | −0.02 |

**Potential-based shaping (added on top, outside the 0.8/0.2 split):**
- Φ = −(cartons remaining + 0.5 if not carrying + distance to current objective / (2 × arena span)).
- Per-step shaping reward F = 30 × (Φ(s′) − Φ(s)).
- It telescopes over an episode, so it **cannot be farmed** by circling; it only rewards real progress.
- It is shaped so that **picking up and delivering always increase Φ**. An earlier distance-only version accidentally punished pick-ups.

**Asymmetric collision blame:** the robot that moved into another takes the individual penalty, so a stationary robot isn't punished for being hit.

## 9. Learning Architecture

**Paradigm:** parameter sharing.
- One policy network controls all 4 robots.
- Each robot feeds it its own observation and acts independently: decentralised execution with a shared brain.

**Algorithm:** Recurrent PPO (PPO with an LSTM, from sb3-contrib).
- The LSTM memory lets robots remember what they have already seen and explored.

**Feature extractor (HiveMindExtractor), three branches:**
- **World branch:** pose, teammates, cartons, depot and time → MLP (128 → 128).
- **LiDAR branch:** 72 rays → 1-D CNN (two stride-2 convolutions, 32 channels) that detects obstacle patterns at every bearing.
- **Message branch:** 48 message values → multi-head self-attention → 64-d.
- The branches are concatenated and passed through a 256-d layer, then an LSTM (256).
- Actor and critic have **separate** feature extractors (share_features_extractor = False), so value-function gradients don't disturb the policy.
- Actor head: 128 → 128. Critic head: 128 → 128.

**Model size:** ≈ 1.64 M parameters.

## 10. Training Pipeline

**Multi-agent to single-agent bridge:**
- A custom vectorised environment presents each 4-robot warehouse as 4 policy slots to Stable-Baselines3.
- All 4 robots act on the same physics step; episodes end and reset together per warehouse.

**Parallelism:**
- Each warehouse runs in its own OS process.
- Training is ~95% physics-bound, so this gives ~4.3× throughput.
- 8 parallel warehouses give 32 robot streams.

**Curriculum (1 → 2 → 4 → 8 → 12 cartons):**
- Start at 1 carton so that finishing an episode, and the +100 / makespan bonuses, can happen by chance.
- Move up a level when the rolling success rate reaches 85%.
- The step cap grows with each level.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Algorithm | RecurrentPPO, MlpLstmPolicy |
| Total training | 13.5 M robot-steps (of a 30 M budget) |
| Rollout | 512 steps/slot × 32 slots = 16,384 |
| Mini-batch | 4,096 |
| Epochs per update | 10 |
| Learning rate | 3e-4, linear decay |
| γ / GAE λ | 0.999 / 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.02 |
| Value coefficient / max grad norm | 0.5 / 0.5 |
| Reward normalisation | VecNormalize (rewards only) |

**Hardware:** NVIDIA RTX A5000 server; ~280–300 robot-steps/s during full-task training.

## 11. Training Results (TensorBoard)

**Curriculum progression:**

| Robot-steps | Cartons in play |
|---|---|
| 0 | 1 |
| 1.25 M | 2 |
| 1.51 M | 4 |
| 1.70 M | 8 |
| 2.56 M → 13.5 M | **12 (full task)** |

**Learning curves:**

| Metric | Start | 3.4 M | 6.8 M | 10.2 M | 13.5 M |
|---|---|---|---|---|---|
| Success rate | 0.16 | 0.56 | 0.92 | 0.88 | **1.00** |
| Cartons delivered / episode | 0.24 | 11.2 | 11.88 | 11.8 | **11.96** |
| Episode length (steps) | 52* | 367 | 304 | 254 | **198** |
| Mean episode reward | 18.9 | 444 | 511 | 510 | **538** |
| Value loss | 2.46 | 0.014 | 0.005 | 0.004 | **0.003** |
| Entropy loss | −4.72 | −3.55 | −3.37 | −3.23 | −2.96 |

\*At the start, episodes are short because the task is 1 carton with a 60-step cap.

**Takeaways:**
- About 11 M of the 13.5 M steps were on the full 12-carton task.
- Success reached 100% and episode length kept falling, so the team kept getting faster.
- The critic stabilised: value loss fell ~800×.

## 12. Evaluation Results
Final model `ppo_recurrent_final.zip`; 50 episodes per level; deterministic policy.

| Cartons | Success rate | Avg makespan (steps) | Step cap | Collision events / episode |
|---|---|---|---|---|
| 4 | 18% (9/50) | 141.1 | 150 | ~3.5 |
| 8 | 84% (42/50) | 152.4 | 250 | ~7.2 |
| **12 (full task)** | **100% (50/50)** | **151.9** | 400 | **~9.7** |

- **Collision counts are per warehouse.** The raw evaluation script summed each world's count once per robot, which reported 4× the true figure (14.2 / 28.7 / 38.8).
- **Full task:** every one of the 12 cartons was delivered in every episode, in ~38% of the step budget.
- **Why 4 cartons is lower:**
  - The policy was specialised on the 12-carton task for ~11 M steps.
  - It keeps a roughly fixed working pace of ~140–150 steps.
  - The 4-carton cap is only 150 steps, so most failures are timeouts close to the limit, not an inability to deliver.

**Reference: scripted greedy controller (30 seeds, 12 cartons):**
- Every robot claims the nearest unclaimed carton.
- It navigates with BFS over the **true shelf map, which the learned policy is never given**.
- Results: 100% success, makespan 97.6 (median 96.5, range 82–123), 6.3 collisions per episode.
- It is an upper-reference controller with full map knowledge. It shows how much makespan remains to gain, especially from communication and better task allocation.

## 13. Key Engineering Challenges and Solutions

| Problem observed | Root cause | Fix |
|---|---|---|
| Robots learned to stand still ("idle freezing"); a 5 M-step run scored below doing nothing | Sparse reward: moving risked −5 collisions long before any delivery reward | Potential-based shaping (scale 30), tuned so a move is worth more than its collision risk |
| Pick-ups gave *negative* total reward | Distance potential jumped when the objective switched from carton to depot | Added a 0.5 "not carrying" term and a cartons-remaining term, so every pick-up and delivery raises Φ |
| ~94 shelf collisions/episode dominated learning | Robots could drive into shelves and were just charged for it | Shelves made impassable; the move is refused with a −0.5 invalid-action penalty |
| Terminal +100 bonus never observed | At 4 cartons a random team never finishes, so the critic never saw success | Curriculum starts at 1 carton |
| Terminal rewards invisible to the learner | 2,000-step episodes: 0.99²⁰⁰⁰ ≈ 2·10⁻⁹ | Step caps sized to ~3× the task (60–400) |
| Curriculum had no effect | Promotion changed a variable the world never read | Promotion now sets the carton count and step cap |
| LiDAR saw "through" shelves after ~250 steps | Chassis slowly sank under gravity, lowering the beam below the shelf plate | Height snapped every step; beam fixed at 0.17 m |
| LiDAR hit the robot's own wheels | Rays started inside the chassis | Rays start 0.28 m from the centre |
| Stationary robots reported motion | Physics settling noise | All poses snapped to the grid before observation |
| Training too slow (one core) | Physics-bound, sequential stepping | One process per warehouse (~4.3×); 5 physics sub-steps instead of 30 (~5×, identical outcomes) |
| Critic destabilising the policy | Shared extractor between actor and critic | Separate feature extractors |
| Stationary robots penalised when hit | Symmetric collision penalty | Only the robot that moved takes the −5 |

## 14. Technology Stack
- **Simulation:** PyBullet, URDF robot, shelf and carton models.
- **RL interface:** Gymnasium.
- **Learning:** Stable-Baselines3 + sb3-contrib (RecurrentPPO), PyTorch.
- **Monitoring:** TensorBoard (success rate, deliveries, pick-ups, curriculum level, losses).
- **Language:** Python.

## 15. Repository Structure
- `hivemind_env/env.py`: warehouse environment, observation, reward, LiDAR, physics.
- `hivemind_env/models.py`: three-branch feature extractor with message attention.
- `hivemind_env/vec_env.py`, `subproc_vec_env.py`: shared-policy vectorisation (in-process and multi-process).
- `hivemind_env/training.py`: curriculum and metrics callbacks, schedules, model loading.
- `hivemind_env/greedy.py`: scripted greedy baseline.
- `scripts/train.py`: training entry point.
- `scripts/inference.py`, `scripts/evaluate_all.py`: evaluation and video rendering.
- `scripts/pretrain_bc.py`: optional behaviour-cloning warm start.
- `scripts/diagnostics/`: reward, observation and incentive verification tools.
- `tests/`: environment and post-training test suites.
- `models/`, `tensorboard_logs/`: trained weights and training logs.

## 16. Limitations
- Motion is grid-based: cell-sized moves and 90° turns, not continuous velocity control.
- Carton positions and teammates' poses are part of the observation; only the map and obstacles are hidden.
- The critic is decentralised (each robot's own view), not a centralised critic over the joint state.
- There are still more collisions and a longer makespan than the map-aware greedy reference.
- Performance on sparser tasks (4 cartons) drops because of specialisation on the full task.

## 17. Future Work
1. **Communication ablation:** train with messages delivered and compare against this baseline (success, makespan, collisions), plus analyse what the tokens come to mean.
2. **Stricter partial observability:** reveal cartons and teammates only when they are within sensor range, and let communication share what each robot has seen.
3. **Centralised critic (MAPPO / CTDE):** give the critic the joint state during training for better credit assignment.
4. **Mixed-difficulty training:** randomise the carton count per episode to remove the 4-carton regression.
5. **Dynamic obstacles:** moving humans or forklifts to force genuine reactive avoidance.
6. **Continuous control and Sim2Real:** velocity-controlled differential drive, then transfer to physical robots via ROS 2.
