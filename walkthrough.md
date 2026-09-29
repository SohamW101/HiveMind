# HiveMind: Complete Repository Walkthrough

This document covers the whole repository and the exact implementation, from the physics of a single robot step to the communication-ablation results. Where it states a number, the number was read from the code, the trained weights, the TensorBoard logs or a measured evaluation.

---

## Table of contents
1. [What HiveMind is](#1-what-hivemind-is)
2. [Repository map](#2-repository-map)
3. [Setup](#3-setup)
4. [The environment (`hivemind_env/env.py`)](#4-the-environment-hivemind_envenvpy)
5. [Observation space](#5-observation-space)
6. [Action space and the communication channel](#6-action-space-and-the-communication-channel)
7. [Reward function](#7-reward-function)
8. [Neural architecture (`hivemind_env/models.py`)](#8-neural-architecture-hivemind_envmodelspy)
9. [Multi-agent vectorisation (`vec_env.py`, `subproc_vec_env.py`)](#9-multi-agent-vectorisation)
10. [Training utilities (`hivemind_env/training.py`)](#10-training-utilities-hivemind_envtrainingpy)
11. [Training script (`scripts/train.py`)](#11-training-script-scriptstrainpy)
12. [The communication ablation](#12-the-communication-ablation)
13. [Scripted greedy controller (`hivemind_env/greedy.py`)](#13-scripted-greedy-controller)
14. [Behaviour-cloning warm start (`scripts/pretrain_bc.py`)](#14-behaviour-cloning-warm-start)
15. [Evaluation, inference and graph scripts](#15-evaluation-inference-and-graph-scripts)
16. [Diagnostics (`scripts/diagnostics/`)](#16-diagnostics)
17. [Tests (`tests/`)](#17-tests)
18. [Assets (`hivemind_env/assets/`)](#18-assets)
19. [Trained models and logs](#19-trained-models-and-logs)
20. [Results](#20-results)
21. [Development history and lessons learned](#21-development-history-and-lessons-learned)
22. [Known issues and caveats](#22-known-issues-and-caveats)
23. [Presentation material](#23-presentation-material)
24. [Git branches and contributors](#24-git-branches-and-contributors)

---

## 1. What HiveMind is

HiveMind is a **multi-agent reinforcement learning (MARL)** project in which **4 warehouse robots** learn, from reward alone, to:
- find cartons in a warehouse whose shelf layout changes every episode;
- pick them up and deliver all 12 to a shared depot as a team;
- avoid each other in narrow aisles.

It has two research goals:
1. **Cooperative MARL under partial observability.** The robots get no map and sense obstacles only with a range-limited, noisy LiDAR.
2. **Emergent communication.** Each robot can broadcast one of 16 discrete tokens every step. Nothing defines what the tokens mean. The project tests, as a controlled **no comm vs comm ablation**, whether a useful protocol emerges from the team reward.

**Core stack:**
- PyBullet physics, wrapped as a Gymnasium environment.
- One shared policy trained with **Recurrent PPO** (sb3-contrib, on top of Stable-Baselines3 and PyTorch).
- A custom three-branch feature extractor: MLP, 1-D CNN, and multi-head attention for messages.
- Potential-based reward shaping and a carton-count curriculum.

---

## 2. Repository map

```
HiveMind/
├── README.md                     project front page
├── walkthrough.md                this document
├── presentation.md               long-form presentation reference
├── presentation-final-content.md slide-by-slide content
├── training_instructions.md      exact server setup + training commands
├── requirements.txt              pip dependencies
├── pyproject.toml                package metadata (hivemind-env 0.1.0, setuptools)
├── environment.yml               conda alternative (Python 3.10, not the tested path)
├── uv.lock                       uv lockfile
├── .gitignore / .gitattributes
│
├── hivemind_env/                 the Python package
│   ├── env.py                    HiveMindMultiAgentEnv: world, physics, observation, reward, LiDAR, messages
│   ├── models.py                 HiveMindExtractor + MessageAttention + DEFAULT_POLICY_KWARGS
│   ├── vec_env.py                HiveMindSharedPolicyVecEnv (in-process, 4 slots per world)
│   ├── subproc_vec_env.py        HiveMindSubprocVecEnv (one OS process per world)
│   ├── training.py               curriculum + metrics callbacks, schedules, model loading, device probe
│   ├── greedy.py                 scripted BFS greedy controller (reference / demonstrator)
│   └── assets/                   URDFs: robot, carton, shelves 1–7 m, shelf generator
│
├── scripts/
│   ├── train.py                  training entry point (run as `python -m scripts.train`)
│   ├── inference.py              headless evaluation + GIF/MP4 rendering
│   ├── evaluate_all.py           runs inference.py at 4 / 8 / 12 cartons
│   ├── evaluate_ablation.py      comm / no comm / intervention evaluation → JSON lines
│   ├── make_presentation_graphs.py  training-comparison + evaluation + message-analysis graphs
│   ├── pretrain_bc.py            behaviour cloning from the greedy controller
│   ├── check_curriculum.py       prints curriculum promotions from a TensorBoard log
│   ├── parse_metrics.py          prints success / length metrics from several TB logs
│   └── diagnostics/              verification, probing, demo and monitoring tools (§16)
│
├── tests/
│   ├── test_environment.py       11 environment / communication unit tests
│   └── test_post_training.py     threshold checks for a trained policy
│
├── models/                       trained weights (.zip) + checkpoints/
├── tensorboard_logs/             TensorBoard event files for every run
├── presentation_graphs/          all generated graphs + architecture diagram + env screenshot
├── presentation_videos/          top-down demo videos (comm, no comm)
├── repo_docs/                    earlier documentation suite (see §22 on accuracy)
└── legacy_archive/               archived analysis docs, original spec PDF, old demos
```

---

## 3. Setup

**Python:** 3.10 or newer; 3.12 and 3.14 are both used on this project.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt tensorboard
```

- **`requirements.txt`:** gymnasium, numpy, pybullet, python-dotenv, tqdm, rich, stable-baselines3, sb3-contrib, torch. TensorBoard is optional; `train.py` detects whether it is installed and warns if not.
- **Graphs:** `matplotlib` is additionally needed for `make_presentation_graphs.py`.
- **Run modules from the repo root** with `python -m scripts.<name>`. `scripts/train.py` does not add the repo root to `sys.path`, so `python scripts/train.py` fails with `No module named 'hivemind_env'`. `inference.py` and the diagnostics add it themselves.
- **GPU:** `get_device()` uses CUDA only on compute capability ≥ 7.0 (and disables cuDNN); otherwise it falls back to CPU. Training is physics-bound (~95% of wall time), so CPU cores matter more than the GPU.
- **ROS machines:** if ROS is sourced in `~/.bashrc` it prepends ROS paths to `PYTHONPATH`, which breaks imports with `No module named 'catkin_pkg'`. Run `unset PYTHONPATH` before activating the venv. HiveMind itself has no ROS dependency.

---

## 4. The environment (`hivemind_env/env.py`)

`HiveMindMultiAgentEnv(gym.Env)` is one warehouse with 4 robots. It is a *joint* environment:
- `step()` takes all 4 robots' actions at once;
- it returns a `(4, 177)` observation, a list of 4 rewards, `terminated`, `truncated` and `info`.

### 4.1 Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `render_mode` | `None` | `"human"` (PyBullet GUI) or `"rgb_array"` (headless camera) |
| `difficulty_level` | 1 | curriculum level label; the world reads `num_cartons`, not this |
| `obs_dim` | 177 | must equal `OBS_DIM_V3`; anything else raises (see §5.3) |
| `show_lidar` | GUI only | draw LiDAR debug lines |
| `obs_size` | `None` | legacy single-agent argument; passing it raises a `TypeError` with an explanation |
| `idle_penalises_turning` | `False` | if True, turning in place counts as idle |
| `lidar_noise` | `True` | Gaussian range noise |
| `substeps` | 5 headless, 30 GUI | physics sub-steps per env step (animation only) |
| `max_steps` | from carton count | episode cap |
| `num_cartons` | `None` (= 12) | cartons in play (curriculum) |
| `shaping` | `True` | potential-based reward shaping |
| `shaping_scale` | 30.0 | κ in the shaping formula |
| `gamma` | 0.99 | stored; the shaping term deliberately uses γ = 1 (§7.4) |
| `communication` | `False` | enables the message channel (§6) |
| `comm_encoding` | `"multi"` | `"multi"` = MultiDiscrete([7, 16]) per robot; `"merged"` = Discrete(112) |

### 4.2 World geometry
- **Grid:** 13 × 13 cells of 1 m (`grid_size = 13`, `cell_size = 1.0`). Physics time step 1/240 s.
- **Grid ↔ world:** `x = c − 6`, `y = 6 − r` (cell centres); `_world_to_grid` inverts this with rounding.
- **Floor:** a static 13 × 13 m box, top flush with z = 0.
- **Walls:** 4 static boxes just outside the arena (±6.6 m), 1 m tall.
- **Depot:** grid cell (0, 0), world (−6, 6), shown as a translucent black square. It has no collision body.
- **Robot spawn cells:** (0,1), (1,0), (0,2), (2,0), all next to the depot. Robots spawn facing east (yaw 0).

### 4.3 Procedural warehouse generation (every `reset`)
- **Shelf rows:** grid rows 1, 3, 5, 7, 9 and 11.
- **Per row:**
  - Split the 11 inner columns (1–11) into three shelf units of lengths x, y, z ≥ 1 with x + y + z = 9.
  - This is done with two random cut points: `cuts = sorted(random.sample(range(1, 9), 2))`.
  - That leaves exactly **2 gap cells** per row, at columns x+1 and x+y+2.
  - Each unit loads `shelf_{length}m.urdf` (1–7 m) with a fixed base.
- **Cartons:** one carton (`carton.urdf`, a 0.5 m cube) is placed in **each gap**, giving 6 rows × 2 = **12 cartons**.
- **Columns 0 and 12** are always free, so the outer corridors connect every aisle.
- **Blocked cells:** every cell on a shelf row in columns 1–11 except the two gaps. Robots can never enter a blocked cell (§4.5).
- **Carton home cells:** each carton's starting cell is recorded before the curriculum removes any. The gaps stay walkable whether or not their carton is still in play.
- **Seeding:** `reset(seed=s)` seeds Python `random` and `numpy.random` (world generation) and Gymnasium's `np_random` (LiDAR noise). The same seed gives the same layout and noise.
- **Settling:** after spawning, 20 physics steps let bodies settle. The robots' resting height is then recorded as `_spawn_z`.

### 4.4 Curriculum-controlled task size
- `active = num_cartons` (clamped to 1–12).
- Slots `active … 11` are removed from the world and **marked delivered at reset**.
- The observation still reports all 12 slots: inactive ones read "delivered" at the depot position. The observation width never changes with difficulty.
- **Episode cap** (`max_steps_for`): 1 → 60, 2 → 90, 4 → 150, 8 → 250, 12 → 400 steps. Counts in between round up to the next rung. The caps are roughly 3× what a scripted controller needs.

### 4.5 One environment step, exactly
`step(actions)`:

1. **Decode communication** (if enabled).
   - Each robot's (move, token) pair is split.
   - `self.messages[i]` is set to the one-hot of its token.
   - Only the move part drives the robot.
2. **Snap every robot** to its grid-cell centre, its yaw to the nearest cardinal direction, and z to `_spawn_z`. This removes drift and prevents slow sinking under gravity.
3. **Resolve each robot's action** into a start and target state:

   | Action | Effect | Invalid when |
   |---|---|---|
   | 0 Forward | move 1 cell along heading | target cell is off-grid or a shelf cell |
   | 1 Backward | move 1 cell against heading | same |
   | 2 Turn left | yaw + 90° | never |
   | 3 Turn right | yaw − 90° | never |
   | 4 Pick up | grab the **nearest available carton within 1.5 cells** (Euclidean from the robot's cell centre) | already carrying, or no carton in reach |
   | 5 Drop off | deliver the carried carton | not carrying, or farther than 1.5 cells from the depot |
   | 6 Stay | nothing | never |

   - An invalid action is **refused**: the robot stays put and the individual −0.5 penalty applies.
   - Robots may still move into another robot's cell. That is a collision, charged but not prevented, because avoiding it is part of the coordination problem.
4. **Pick-up mechanics:**
   - The carton is claimed immediately: removed from `resource_ids` and recorded as `carried_resource_ids[i]`.
   - The arm yaws toward the carton, the fingers close, and the LiDAR mast rises (prismatic joint to 0.5 m).
   - **Diagonal cartons:** if the carton is diagonally adjacent, the animation side-steps the robot into a free cardinal neighbour during the grab and returns it afterwards. The final pose is unchanged.
   - Pick-up steps use 3× the sub-steps, for a smoother animation.
5. **Carrying:** while carrying, the carton is kinematically placed each sub-step at `gripper_reach = 0.3 m` from the robot along the arm direction, at z = 0.25 m.
6. **Drop-off mechanics:** the carton animates from the gripper to the depot. It is then removed from the world, `delivered[slot] = True`, and the robot's carrying state clears.
7. **Physics execution:**
   - All robots move **simultaneously**. For each sub-step `k = 1…N`, poses, arm/finger/mast joints and wheel angles are interpolated and set with `resetBasePositionAndOrientation` / `resetJointState`, then `stepSimulation` is called.
   - This is interpolated teleport motion, not torque control. Outcomes (makespan, collisions, deliveries) are identical for any sub-step count; sub-steps only affect animation smoothness and speed.
8. **Kinematics update:** velocity = snapped displacement since the last step, in cells/step.
9. **Collision detection** (`_detect_collisions`):
   - Contact points are queried for every robot pair and for every robot against shelves and walls. Carton contacts are not counted.
   - A contact set is kept across steps, and only **new** contacts count as collision events. Two robots that stay touching are charged once.
10. **Termination:**
    - `terminated = all 12 delivered flags true`. Inactive slots are pre-marked, so this means all active cartons are delivered.
    - `truncated = not terminated and step ≥ max_steps`.
11. **Reward** (§7) is computed, and the episode return accumulates per robot.

**Info dict:**
- **Positions and progress:** `robot_pos`, `remaining_resources`, `delivered` (active cartons delivered), `delivered_flags_total`, `active_cartons`, `is_carrying`, `obs_dim`.
- **Contacts:** `shelf_contacts`.
- **LiDAR:** `lidar` (normalised scans), `lidar_distances` (metres).
- **Per-step events:** `collisions`, `collision_pairs`, `pickups`, `deliveries`, `invalid_actions` (4 bools each).
- **Outcome:** `all_delivered`, `is_success`, `reward_breakdown` (every reward term per robot), `episode_reward`.

**Rendering and cleanup:**
- `render()` in `rgb_array` mode returns a 600 × 600 overhead frame (ER_TINY_RENDERER, so it works headless).
- `close()` is idempotent (safe to call twice), which matters for subprocess teardown.

### 4.6 LiDAR
- **Geometry:** 72 rays over a 270° arc centred on the heading (−135° to +135°), range 0.1–10 m.
- **Casting:**
  - one `rayTestBatch` per robot;
  - rays start **0.28 m** from the robot centre, so they don't hit the robot's own wheels;
  - beams are at a fixed **z = 0.17 m**, inside both the chassis band and the bottom shelf plate, so the sensor sees exactly what the body would hit.
- **Noise:** added to real hits only, with σ = 0.01 m + 1% of range, drawn from the seeded `np_random`.
- **Output:** distances clipped to [0.1, 10] and normalised as `(d − 0.1) / 9.9` ∈ [0, 1].
- **Caching:** scans are computed once per step and reused by `_get_obs` and `_get_info`.
- **Ray count:** the MAWC spec asks for 720 rays. 72 (one-tenth) keeps 3.75° resolution, so 1 m obstacles remain resolvable across the arena, at a fraction of the cost.

---

## 5. Observation space

### 5.1 Layout (V3, 177 floats per robot)
Observation space: `Box(-1, 1, shape=(4, 177), float32)`, one row per robot, clipped to [−1, 1].

| Slice | Size | Component | Encoding |
|---|---|---|---|
| [0:3] | 3 | own pose | x/6.5, y/6.5, heading wrapped to [0,1) (E=0, N=0.25, W=0.5, S=0.75) |
| [3:5] | 2 | own velocity | last-step displacement (cells/step) |
| [5:6] | 1 | own carrying | 0 / 1 |
| [6:15] | 9 | other robots' poses | 3 × (x, y, heading), fixed agent order, self skipped |
| [15:18] | 3 | other robots' carrying | 0 / 1 |
| [18:30] | 12 | carton status | 0 available · ⅓ carried by me · ⅔ carried by another · 1 delivered |
| [30:54] | 24 | carton positions | 12 × (x, y); a carried carton reports its live position, a delivered one the depot |
| [54:56] | 2 | depot offset | (depot − robot) / 13, which carries direction and distance |
| [56:57] | 1 | elapsed time | step / max_steps |
| [57:129] | 72 | LiDAR | normalised ranges |
| [129:177] | 48 | messages | 3 teammates × 16-token one-hot |

**Notes:**
- Every pose in the observation is the **snapped** pose, so the observation matches exactly what `step()` acts on.
- Heading is one wrapped float, not sin/cos. That is valid because headings are always cardinal.
- Carton status is one ordinal float per carton, not a one-hot, to keep the width small.

### 5.2 What is hidden
- **Not given:** the shelf layout (no map or occupancy grid), paths, and any predefined meaning for message tokens. Shelves and walls are perceived only through LiDAR.
- **Given:** teammate poses and carton positions and statuses.

### 5.3 Width pinning (three safeguards)
The observation width is baked into saved weights, so it is pinned:
1. `_build_slices()` must tile [0, 177) exactly, checked **at import time**.
2. The message block must be last, so world features keep stable indices.
3. The constructor rejects any `obs_dim ≠ 177`, naming the superseded versions: V1 = 81 (no carton positions) and V2 = 105 (no LiDAR).

A future width change must be introduced as a new `OBS_DIM_V4`, never as an edit to V3.

---

## 6. Action space and the communication channel

### 6.1 Action spaces
- `communication=False`: `MultiDiscrete([7] * 4)`, one movement action per robot.
- `communication=True, comm_encoding="multi"`: `MultiDiscrete([7, 16] * 4)`, i.e. `[m0, t0, m1, t1, m2, t2, m3, t3]`.
- `communication=True, comm_encoding="merged"`: `MultiDiscrete([112] * 4)`, where `a = move × 16 + token`.

### 6.2 Message mechanics
- In `step()`, each robot's token becomes a one-hot vector in `self.messages[i]`.
- `_get_obs()` writes the other three robots' message vectors, in fixed order with self skipped, into slots [129:177].
- The observation returned by `step(t)` contains the tokens chosen at step t. The policy therefore sees teammates' messages when choosing its action at step t + 1.
- Messages are zeroed at every `reset()`.

### 6.3 Speaker and listener
- **Speaker:** the policy's 16-way categorical token head. The joint per-robot action factorises as π(m, c | o) = π(m | o) · π(c | o), so PPO's log-probability and entropy are the sums over both heads.
- **Listener:** `MessageAttention` in the feature extractor (§8).
- **Incentive:** there is **no reward for communicating**. A token can only affect the return through teammates' reactions, so any protocol must emerge from the shared team reward.

---

## 7. Reward function

### 7.1 Tables (as implemented)

**Shared team reward (weight 0.8, identical for all robots):**

| Event | Reward |
|---|---|
| Carton delivered (by anyone) | +10 per carton |
| All cartons delivered | +100 (once) |
| Makespan bonus | +50 × (T_max − T) / T_max (once, on completion) |
| Collision event (robot–robot or robot–wall/shelf), team share | −1 per event |
| Time penalty | −0.05 per step |

**Individual reward (weight 0.2, per robot):**

| Event | Reward |
|---|---|
| Own pick-up | +1 |
| Own delivery | +2 |
| Collision the robot caused (moved into another robot) or hitting a wall/shelf | −5 per event |
| Invalid action | −0.5 |
| Idle (speed < 0.1 cell/step, not within 1.5 cells of the depot; turning exempt) | −0.02 per step |

**Shaping (outside the split):** F = 30 · (Φ(s′) − Φ(s)).

**Unused:** the spec's replanning penalty (−0.1) is defined as a constant but has no trigger, because there is no path planner.

### 7.2 Equations
For robot *i*:

  rᵢ = 0.8 · R_shared + 0.2 · R_ind,i + Fᵢ

  R_shared = 10·Dₜ + 𝟙[done]·(100 + 50·(T_max − T)/T_max) − 1·Cₜ − 0.05

  R_ind,i = 1·pickᵢ + 2·delivᵢ − 0.5·invalidᵢ − 5·collᵢ − 0.02·idleᵢ

  collᵢ = #obstacle events of i + #robot events involving i **where i moved** (Forward/Backward)

- Dₜ: deliveries this step.
- Cₜ: new collision events this step.
- **Collision blame:** the asymmetric definition of collᵢ means a stationary robot that gets hit pays only its 0.8 × (−1) team share.

**Effective per-robot values (after weighting):**

| Term | Per robot |
|---|---|
| Per delivery (anyone's) | +8 |
| Completion | +80 |
| Makespan bonus | up to +40 |
| Collision, team share | −0.8 per event |
| Time | −0.04 per step |
| Own pick-up | +0.2 |
| Own delivery | +0.4 |
| Collision the robot caused | −1.0 per event |
| Invalid action | −0.1 |
| Idle | −0.004 per step |

### 7.3 Why each piece exists
- **Shared 0.8:** makes the task cooperative; every robot is paid for every delivery.
- **Individual 0.2:** gives each agent its own credit signal against free-riding (the lazy-agent problem).
- **Makespan bonus:** rewards finishing sooner, not just finishing.
- **Collisions counted per event, not per step:** otherwise two touching robots would bleed reward every step.
- **Shelves refuse the move rather than charging a collision:** this removed ~94 shelf collisions per episode that were taxing exploration (§21).
- **Turning exempt from idle:** turning in place is necessary for navigation.

### 7.4 Potential-based shaping
  Φᵢ(s) = −( N_left + 0.5·(1 − carryᵢ) + dᵢ / (2·13) )

- N_left: undelivered active cartons.
- dᵢ: Euclidean distance to the depot if carrying, otherwise to the nearest available carton.

**Properties:**
- **Telescoping:** ΣFᵢ = κ(Φ_T − Φ_0), independent of the path taken, so the shaping cannot be farmed.
- **Pick-up is never punished:**

  ΔΦ = 0.5 + (d_carton − d_depot)/26 ≥ 0

- **Delivery is never punished:**

  ΔΦ = 0.5 + (d_depot − d_next)/26 ≥ 0

  A teammate's delivery also raises everyone's Φ, by 1 + Δd/26.
- **Dense gradient:** each one-cell step toward the objective pays up to κ/26 ≈ +1.15, more than the expected collision cost of moving.
- **γ = 1 in F (a deliberate deviation):** the textbook γΦ(s′) − Φ(s) adds a drift of −(1−γ)Φ > 0, a bonus for loitering far from the goal.
- **Placement:** F sits outside the 0.8 / 0.2 split so that κ means what it says.
- **Reporting:** the breakdown records F as `shaping (unweighted)`.

### 7.5 Discounting and normalisation
- The policy optimises Σ γᵗ rₜ with **γ = 0.999** in PPO.
- Over a 400-step episode, terminal rewards keep 0.999⁴⁰⁰ ≈ 0.67 of their value.
- Advantages use GAE with λ = 0.95.
- `VecNormalize(norm_obs=False, norm_reward=True)` scales rewards by a running estimate of the return's standard deviation.

---

## 8. Neural architecture (`hivemind_env/models.py`)

`presentation_graphs/architecture_diagram.png` / `.svg` shows this diagram. Every shape below was verified against the trained weights.

### 8.1 `HiveMindExtractor` (custom `BaseFeaturesExtractor`, output 256)
The 177-d observation is split by `OBS_SLICES`, never by literal indices:

| Branch | Input | Layers | Output |
|---|---|---|---|
| World | [0:57] (57) | Linear 57→128, ReLU, Linear 128→128, ReLU | 128 |
| LiDAR | [57:129] (72) as 1 channel | Conv1d(1→32, k5, s2, p2), ReLU, Conv1d(32→32, k3, s2, p1), ReLU, flatten | 18 × 32 = **576** |
| Messages | [129:177] (48) → 3 × 16 | `MessageAttention`: Linear 16→64 per sender, MultiheadAttention(64, 4 heads) over the 3 senders, max-pool over senders | 64 |
| Head | concat 128 + 576 + 64 = **768** | Linear 768→256, ReLU | **256** |

**Design reasons:**
- The 1-D CNN exploits the angular order of the LiDAR sweep: a wall is a run of adjacent rays.
- The attention and max-pool make the message features independent of sender order.
- The extractor validates the observation width against `OBS_SLICES` at construction.

### 8.2 Policy (`MlpLstmPolicy`, RecurrentPPO)
`DEFAULT_POLICY_KWARGS`:
- `features_extractor_class = HiveMindExtractor`, `features_dim = 256`.
- `share_features_extractor = False`: the actor and critic each have **their own copy** of the extractor.
- `net_arch = {pi: [128, 128], vf: [128, 128]}`.

**sb3-contrib defaults in effect:**
- a **separate LSTM for the actor and the critic** (`shared_lstm=False`, `enable_critic_lstm=True`);
- hidden size 256, 1 layer.

**Full data path:**
```
obs(177) ─► extractor_π ─► LSTM_π(256) ─► MLP 256→128→128 ─► Linear 128→23 ─► [7 move logits | 16 token logits]
obs(177) ─► extractor_V ─► LSTM_V(256) ─► MLP 256→128→128 ─► Linear 128→1  ─► V(o)
```

**Total parameters: 1,638,232.** This is identical in both ablation arms.

### 8.3 PPO objective
  L = E[min(ρÂ, clip(ρ, 0.8, 1.2)Â)] − 0.5·E[(V − V̂)²] + 0.02·E[H(π)]

- ρ = π_θ / π_θ_old, taken over the joint (move, token) action.
- H = H_move + H_token.
- Gradients are clipped at norm 0.5. Advantages are normalised per batch.

---

## 9. Multi-agent vectorisation

Stable-Baselines3 expects single-agent environments. HiveMind presents each 4-robot world as **4 policy slots** that share one policy (**parameter sharing**).

**`HiveMindSharedPolicyVecEnv` (in-process):**
- `num_worlds` worlds give `4 × num_worlds` slots, **world-major** (slots 0–3 are world 0's robots).
- `world_of(slot) = slot // 4`, `agent_of(slot) = slot % 4`.
- **Stepping:** per world, the slots' actions are gathered into the joint action, the world is stepped, and the 4 rows of obs/reward/done are scattered back. In comm "multi" mode each slot's `[move, token]` pair is interleaved into `[m0, t0, m1, t1, …]`.
- **Done is per world:** all 4 slots terminate together and auto-reset. The final observation goes to `info["terminal_observation"]` and `TimeLimit.truncated` is set, following SB3's contract.
- **Per-slot info:** `world`, `agent`, `is_success`, `all_delivered`, `delivered`, `remaining_resources`, `collisions`, `picked_up`, `delivered_by_me`, `invalid_action`.
- **Seeding:** each world's seed is `seed + 1000·world + episode_counter`, so worlds differ and never replay one layout.
- **Attribute plumbing:**
  - `get_attr` maps a slot to its world;
  - `set_attr` and `env_method` apply once per affected world;
  - this is how the curriculum changes `num_cartons` / `max_steps`.

**`HiveMindSubprocVecEnv` (one OS process per world):**
- Same slot layout and API.
- Workers run over pipes with the `spawn` start method, so the entry point needs an `if __name__ == "__main__"` guard.
- Workers send compact per-slot infos, not the full info. That avoids pickling the LiDAR arrays every step.
- Auto-reset happens inside the worker.
- **Throughput:** training is ~95% physics-bound, so this is ~4.3× faster at 12 workers than in-process. It is the default training backend.

---

## 10. Training utilities (`hivemind_env/training.py`)

### 10.1 Curriculums
- **Training ladder:** `TRAINING_CURRICULUM = {1: 1, 2: 2, 3: 4, 4: 8, 5: 12}` cartons.
- **Why it starts at 1 carton:** an episode ends only when **every** carton is delivered. At 1 carton a random team completes ~33% of episodes, so the critic sees terminal rewards. At 4 cartons it completes none.
- **Reported ladder:** `CURRICULUM_CARTONS = {1: 4, 2: 8, 3: 12}`.

### 10.2 `CurriculumCallback`
- Records success (`info["is_success"]`) for **every slot that finishes**, so the window counts robot-episodes.
- Every `check_freq` calls (2048 in `train.py`) it logs:
  - `curriculum/difficulty_level`;
  - `curriculum/target_cartons`;
  - once the window of 300 is full, `curriculum/success_rate`.
- **Promotion:** when the rolling success rate is ≥ 0.85, it moves up one level. It sets `difficulty_level`, `num_cartons` and `max_steps` on every world (effective at the next reset), clears the history, re-installs the learning-rate schedule and sets `ent_coef = 0.02`.

### 10.3 Other utilities
- **`MetricsCallback`:** every 1000 calls it logs rolling means of `metrics/pickups_per_episode`, `metrics/deliveries_per_episode` and `metrics/cartons_delivered_per_episode` over finished episodes.
- **Schedules:**
  - `linear_schedule(lr)`: linear decay to 0 over the run;
  - `restart_schedule(lr, progress)`: a rescaled decay for a genuine mid-run restart.
- **`INFERENCE_CUSTOM_OBJECTS` / `load_policy`:** load checkpoints with stand-in `learning_rate` / `lr_schedule` / `clip_range`. This avoids unpickling training closures across Python versions, which segfaults on 3.14. `load_policy` picks RecurrentPPO when the filename contains "recurrent".
- **`make_env`:** a SubprocVecEnv-style factory for the raw joint environment.
- **`get_device`:** chooses CUDA or CPU as described in §3.
- **`num_parallel_envs`:** `min(cap, cpu_count)`.

---

## 11. Training script (`scripts/train.py`)

**Pipeline:**
1. Build `HiveMindSubprocVecEnv` (or the in-process env with `--backend inprocess`).
2. Wrap it in `VecMonitor` (episode stats) and `VecNormalize` (rewards only).
3. Create `RecurrentPPO("MlpLstmPolicy", …)` with `DEFAULT_POLICY_KWARGS`.
4. Add the callbacks: `MetricsCallback`, optionally `CurriculumCallback`, and `CheckpointCallback`.
5. Call `learn()`. `Ctrl+C` still saves `models/<run>_final.zip`.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Algorithm / policy | RecurrentPPO / MlpLstmPolicy |
| Worlds / slots | `--worlds 8` → 32 agent streams |
| Rollout (`--n-steps`) | 512 per slot → buffer 16,384 |
| Mini-batch (`--batch-size`) | 4,096 (auto-clamped to the buffer; kept ≥ 4 mini-batches) |
| Epochs | 10 |
| Learning rate | 3e-4, linear decay to 0 over `--timesteps` |
| γ / GAE λ | 0.999 / 0.95 |
| Clip range | 0.2 |
| Entropy / value coefficients | 0.02 / 0.5 |
| Max grad norm | 0.5 |
| Seed | 0 |
| Checkpoints | every `--checkpoint-every` 250k steps → `models/checkpoints/<run>_<steps>_steps.zip` |

**All flags:**
- **Training length and parallelism:** `--timesteps`, `--worlds`, `--backend {subproc, inprocess}`.
- **Optimiser:** `--lr`, `--gamma`, `--n-steps`, `--batch-size`.
- **Task and curriculum:** `--difficulty`, `--curriculum`, `--num-cartons`, `--max-steps`.
- **Reward:** `--shaping-scale`, `--no-shaping`.
- **Physics:** `--substeps`.
- **Communication:** `--communication`, `--comm-encoding {multi, merged}`.
- **Run management:** `--seed`, `--init-from`, `--run-name`, `--save-dir`, `--log-dir`, `--checkpoint-every`.
- **Quick check:** `--smoke`, a 4,096-step end-to-end check.

**Canonical command** (the exact server procedure is in `training_instructions.md`):
```bash
python -m scripts.train --communication --curriculum --num-cartons 1 \
  --worlds 8 --timesteps <T> --seed 0 --run-name <name>
```
- Use the same `<T>` for both ablation arms; it also sets the learning-rate decay horizon.
- `--num-cartons 1` is required with `--curriculum`, so the environment starts on the curriculum's first rung.

---

## 12. The communication ablation

**Design: exactly one variable changed.**

| | No comm | Comm |
|---|---|---|
| Model file | `models/ppo_recurrent_final.zip` | `models/ppo_recurrent_comm_s0_final.zip` |
| TensorBoard run | `tensorboard_logs/ppo_recurrent_30M_1` | `tensorboard_logs/ppo_recurrent_comm_s0_2` |
| Action space | `[7, 16]` | `[7, 16]` |
| Tokens produced | yes | yes |
| Tokens delivered to teammates | **no** (message slots held at zero) | **yes** |
| Network, parameter count, reward, curriculum, hyperparameters, seed | identical | identical |

- **Why the no-comm arm keeps a token head:** its input size, network and action space stay identical to the comm arm. Only information flow differs.
- **Evaluating the no-comm model:** zero its message slots, matching how it was trained. `evaluate_ablation.py --variant nocomm` does this.
- **Intervention variant:** `comm_zeroed` evaluates the comm model with its incoming messages zeroed. If the channel carried information the policy relies on, performance would drop.

**Message-analysis metrics** (computed in `make_presentation_graphs.py`):
- **Token entropy:** H = −Σ p(k) log₂ p(k), with a maximum of log₂ 16 = 4 bits.
- **Mutual information with robot state:**

  I(token; carrying) = Σ p(c, k) log₂ [p(c, k) / (p(c) p(k))]

- **Mutual information with agent identity:** I(token; agent).

---

## 13. Scripted greedy controller

`hivemind_env/greedy.py`, `GreedyController(env)`:
- **Claiming:** each robot claims the nearest unclaimed carton. Claims are released when the carton is delivered or taken.
- **Navigation:** the robot BFS-navigates (4-connected) on the **true blocked-cell map**, to any free cell within reach (1.5 cells) of its carton or of the depot.
- **Pick-up and drop:** it picks up or drops as soon as the target is in range; no turning is spent lining up.
- **Collision handling:**
  - robots are planned in index order, and each avoids cells already reserved by earlier robots (strict priority prevents swap livelocks);
  - a progress watchdog (`MAX_WAIT = 3`) stops endless yielding.
- **180° turns:** handled as a single Backward action.
- **Communication envs:** it formats actions for communication-enabled environments, always sending token 0.

**Purpose:**
- a map-aware reference controller;
- the demonstrator for behaviour cloning;
- a solvability check on generated worlds.

**Measured** (30 seeds, 12 cartons): 100% completion, makespan mean 97.6 (median 96.5, range 82–123), 6.3 collision events per episode. Mean makespans at 4 / 8 / 12 cartons: 23.0 / 59.1 / 97.6.

---

## 14. Behaviour-cloning warm start

`scripts/pretrain_bc.py`:
- **Method:** roll out the greedy controller, record (observation, action) for every robot, and fit a PPO policy's action distribution by cross-entropy. The result is saved to `models/bc_pretrained.zip`.
- **Arguments:** `--episodes 60`, `--num-cartons 4`, `--epochs 12`, `--batch-size 512`, `--lr 1e-3`, `--out`.
- **Result:** 76.4% action match, but 0/10 completions deterministically. The clone oscillates forward/backward because greedy uses Backward for 180° turns, so the same observation maps to two modes.
- **Status:** kept as a later-resort tool. The real fixes were to the environment and the reward (§21).

---

## 15. Evaluation, inference and graph scripts

| Script | What it does |
|---|---|
| `scripts/inference.py` | Loads a RecurrentPPO model, runs N headless episodes (or a GUI with `--render`), prints per-episode success / delivered / steps and a summary. `--gif` / `--mp4` render one episode. Flags: `--episodes`, `--num-cartons`, `--model-path`. |
| `scripts/evaluate_all.py` | Runs `inference.py`'s `evaluate` at 4, 8 and 12 cartons and prints a summary table. |
| `scripts/evaluate_ablation.py` | The ablation evaluator: `--variant {comm, nocomm, comm_zeroed}`, `--cartons`, `--episodes`, `--start`, `--seed0` (episode e uses seed seed0 + e, so variants see identical layouts), `--out` (JSON lines with success, steps, delivered, collisions, invalid actions, and per-step tokens, moves and carrying flags). |
| `scripts/make_presentation_graphs.py` | From `--eval-dir` JSON lines and TensorBoard logs (`--nocomm-log`, `--comm-log`): training comparisons over an identical step range (t1–t10), evaluation graphs (e1–e6), and a summary including entropy and mutual information. |
| `scripts/check_curriculum.py` | Prints curriculum promotions for a TensorBoard run. |
| `scripts/parse_metrics.py` | Prints last and best success / length metrics for a list of runs. |

**Reproducing the ablation evaluation:**
```bash
for v in comm nocomm comm_zeroed; do
  python -m scripts.evaluate_ablation --variant $v --cartons 12 --episodes 100 --out eval/episodes.jsonl
done
for c in 4 8; do for v in comm nocomm; do
  python -m scripts.evaluate_ablation --variant $v --cartons $c --episodes 100 --out eval/episodes.jsonl
done; done
python scripts/make_presentation_graphs.py --eval-dir eval --comm-log tensorboard_logs/ppo_recurrent_comm_s0_2
```

---

## 16. Diagnostics

| Script | Purpose |
|---|---|
| `smoke_test.py` | Import / device / reset / step checks with PASS / FAIL / PENDING states; prints the observation layout. |
| `verify_observations.py` | Drives one robot through a full pick-up and delivery, checking every observation component against ground truth read from PyBullet. Exits non-zero on failure. |
| `verify_rewards.py` | Drives a full delivery, prints the reward breakdown every step, checks every reward term against the spec, and recomputes the weighted split by hand. |
| `diagnose_incentives.py` | Measures in seconds what the reward pays for: approach gain per cell, P(collision \| move), and the expected value of moving vs standing. Built after training runs failed from reward arithmetic. |
| `probe_policy.py` | Action mix, pick-ups, deliveries and collisions of a checkpoint, deterministic vs stochastic. Separates "stands still", "thrashes" and "works". |
| `probe_pickup.py` | When a carton is in reach, does the policy press PICKUP? Diagnoses "drives to the carton and stops". |
| `run_evaluation.py` | Fixed-seed evaluation across curriculum levels with Wilson confidence intervals; JSON + text report (makespan-first). |
| `test_run.py` | Watch a trained policy in the GUI; `--stochastic` samples instead of argmax; `--headless` for numbers only. |
| `play_multi.py` | Scripted BFS demo in the GUI (no learning). |
| `demo.py` | Renders a demo episode from a trained model. |
| `watch_run.py` | Tails a training log and prints one compact line per PPO iteration. |
| `check_model_spaces.py` | Prints a saved model's action and observation spaces. |

---

## 17. Tests

**`tests/test_environment.py` (11 tests):**
- **Observation:** shape (4, 177) and bounds; LiDAR range; message slots zero without communication; message slots filled with communication.
- **Idle rule:** turning is exempt from the idle penalty.
- **Environment basics:** `close()` is idempotent; the curriculum sets `active_cartons`.
- **Communication:** the `multi` and `merged` action spaces; a sent token appears in teammates' slots; a robot never sees its own token.

Run with `python -m pytest tests/test_environment.py` (install pytest first).

**`tests/test_post_training.py`:** threshold checks for a trained policy:
- completion ≥ 50%;
- makespan < 120;
- no single action > 60% of all actions;
- ≥ 1 pick-up per episode;
- deterministic vs stochastic comparison;
- token entropy > 1 bit when communicating.

---

## 18. Assets

- **`diff_drive_bot.urdf`:**
  - chassis box 0.32 × 0.24 × 0.10 m, 1.2 kg;
  - four wheels, r = 0.07 m, continuous joints;
  - a 0.30 m LiDAR post plus a LiDAR head on a **prismatic mast joint (0–0.5 m)** that rises while carrying;
  - an arm base on a **revolute yaw joint** (±2π), a 0.2 m boom and a gripper palm;
  - two **prismatic fingers** (±0.04 m).
- **`carton.urdf`:** a 0.5 × 0.5 × 0.5 m cardboard box with tape and seam visuals; the bottom sits at z = 0.
- **`shelf_1m.urdf` … `shelf_7m.urdf`:** 1 m deep shelving of each length.
  - Plates are 0.08 m thick, at heights 0.18 / 1.0 / 1.8 m, with corner posts.
  - Decorative cartons are stocked on the lower two plates.
  - The bottom plate spans 0.14–0.22 m, overlapping the robot's chassis height, so shelves are real obstacles.
- **`shelf.urdf`:** the base shelf model.
- **`generate_shelves.py`:** generates the shelf URDFs. It uses the same height list for plates and stocked cartons, so the two can't drift apart.

---

## 19. Trained models and logs

| File | What it is |
|---|---|
| `models/ppo_recurrent_final.zip` | **No-comm** arm, the final RecurrentPPO model |
| `models/ppo_recurrent_comm_s0_final.zip` | **Comm** arm, the final RecurrentPPO model |
| `models/ppo_recurrent_30k_final.zip` | short RecurrentPPO pipeline check |
| `models/ppo_shared_2026090*_final.zip` | earlier non-recurrent PPO runs from the pipeline's development |
| `models/checkpoints/ppo_shared_20260905_202240_*` | periodic checkpoints of one of those earlier PPO runs |
| `tensorboard_logs/ppo_recurrent_30M_1` | no-comm training log |
| `tensorboard_logs/ppo_recurrent_comm_s0_2` | comm training log |
| `tensorboard_logs/ppo_shared_*`, `my_first_run_*`, `ppo_recurrent_30k_1` | earlier development runs |

**Loading a model** for inspection or evaluation:
```python
from sb3_contrib import RecurrentPPO
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS
model = RecurrentPPO.load(path, device="cpu", custom_objects=INFERENCE_CUSTOM_OBJECTS)
```
Pass the LSTM state and `episode_start` flags to `model.predict` each step.

**Logged TensorBoard tags:**
- **rollout:** `ep_len_mean`, `ep_rew_mean`, `success_rate`;
- **train:** `loss`, `policy_gradient_loss`, `value_loss`, `entropy_loss`, `approx_kl`, `clip_fraction`, `clip_range`, `explained_variance`, `learning_rate`;
- **curriculum:** `difficulty_level`, `target_cartons`, `success_rate`;
- **metrics:** `cartons_delivered_per_episode`, `pickups_per_episode`, `deliveries_per_episode`;
- **time:** `fps`.

---

## 20. Results

### 20.1 Training (both arms compared over an identical step range)
Graphs: `presentation_graphs/t1_…` to `t10_…`.

| At the same point in training | No comm | Comm |
|---|---|---|
| Training success rate | 96.8% | 99.6% |
| Episode length (steps) | 208 | 187 |
| Mean episode reward | 531 | 542 |
| Critic explained variance | 0.999 | 0.999 |
| Value loss | 0.0029 | 0.0022 |
| Reached the 12-carton stage | 2.56 M | 2.16 M |
| First ≥ 90% success on the full task | 4.59 M | 3.82 M |

- **Curriculum promotions:**
  - no comm: 1 → 2 at 1.25 M, → 4 at 1.51 M, → 8 at 1.70 M, → 12 at 2.56 M;
  - comm: → 2 at 0.98 M, → 4 at 1.18 M, → 8 at 1.51 M, → 12 at 2.16 M.
- **Losses:** total and clipped-surrogate losses drop sharply in the first ~2–3 M steps, then plateau. The value loss falls by ~3 orders of magnitude. The comm run's losses drop earlier.
- **PPO diagnostics:** approximate KL rises to ~0.06–0.09 and clip fraction to ~0.37–0.39 in both runs. Updates are fairly aggressive, with no divergence.
- **Caveat:** this is one run per arm, so differences are observations, not statistically established effects.

### 20.2 Evaluation (100 headless episodes per setting, deterministic, identical layouts per episode index)

| Cartons | No comm success | Comm success |
|---|---|---|
| 12 (full task) | **100%** | **100%** |
| 8 | 84% | 60% |
| 4 | 26% | 39% |

- **Full task:** both arms solve it in every episode.
- **Smaller tasks:** both arms are weaker. They were trained to specialise on 12 cartons, and the smaller tasks have tighter step caps.

### 20.3 Did communication emerge?
Graphs: `e4_intervention_makespan.png`, `e5_token_usage.png`, `e6_token_by_state.png`.

**Intervention:** the comm model with its incoming messages zeroed still succeeds in 100/100 episodes, and its steps to finish don't increase. The policy does not rely on the channel.

**Token statistics** (12 cartons, 100 episodes):

| Metric | Comm model |
|---|---|
| Token entropy | 3.80 of 4.00 bits (near uniform) |
| Tokens covering 90% of messages | 12 of 16 |
| I(token; carrying) | 0.013 bits |
| I(token; agent) | 0.002 bits |

**Conclusion:** the channel is open, but no informative protocol emerged.

**Reasons:**
- Teammate poses and carton states are already observed, so messages have little new to add.
- Speaker and listener face a chicken-and-egg problem with no direct incentive to communicate.

**Lesson:** emergent communication has to be *measured* with interventions and information metrics, not inferred from task performance.

---

## 21. Development history and lessons learned

Each row is a real failure, the diagnosis behind it and the fix. Most are also documented inline in `env.py`.

| Symptom | Root cause | Fix |
|---|---|---|
| A 5 M-step run completed 0 of ~2,500 episodes and scored −103, *below* the −94 of standing still | Sparse reward; moving risked collisions long before any payoff | Potential-based shaping |
| Completing a pick-up gave negative total reward | Distance-only Φ jumped when the objective switched from carton to depot | 0.5 hand-off term + cartons-remaining term |
| A shaping scale of 15 behaved like 1.5 | F was added inside the 0.1 / 0.2 individual bucket | F moved outside the split |
| Canary run went to `stay` 100% under argmax | At κ = 6 the expected value of a move was −0.23 | κ = 30, set from measured P(collision \| move) |
| 105.8 collisions/episode, 93.8 of them against shelves | Shelves were enterable and only charged | Shelf cells are blocked; −0.5 invalid action |
| Robots drove *under* shelves | Bottom plate at 0.30 m, above the chassis | Plate lowered to 0.18 m (spans 0.14–0.22) |
| Terminal +100 never observed in any run | At 4+ cartons a random team never finishes | Curriculum from 1 carton |
| Terminal rewards invisible | 2,000-step episodes: 0.99²⁰⁰⁰ ≈ 2·10⁻⁹ | Caps sized to the task (60–400); γ = 0.999 |
| Curriculum promotion did nothing | It set a variable the world never read | Promotion sets `num_cartons` and `max_steps` |
| LiDAR saw through shelves after ~250 steps | Chassis sank ~0.17 mm/step, lowering the beam | z snapped each step; beam at fixed 0.17 m |
| LiDAR min 0.12 m, max 2.2 m in a 13 m arena | Rays started inside the robot and hit its wheels | Rays start at 0.28 m |
| One-cell moves measured 0.992; a static robot "moved" | Physics settling noise | All observed poses snapped |
| One heading encoded as both +1 and −1 | Yaw/π at the ±π boundary | Heading wrapped to [0, 1) |
| Five seeds gave identical observations | V1 had carton status but no positions | V2 added carton positions; V3 added LiDAR |
| Curriculum credited env 0's outcome to all workers | Callback indexed `rewards[0]` | Each done paired with its own info |
| Training used one core | Worlds stepped sequentially in-process | SubprocVecEnv (~4.3×); 5 sub-steps instead of 30 (~5×, identical outcomes) |
| Value gradients disturbed the policy | Shared extractor | Separate actor and critic extractors |
| Stationary robots punished when hit | Symmetric collision penalty | Only the moving robot takes −5 |
| Stale call sites could silently build a wrong-width env | Unpinned observation width | Width pinned with three safeguards |
| Comm vs no comm would not be comparable if inputs changed later | Message inputs added after the fact change the network | Message slots and token head reserved from the start |
| Unpickling checkpoints across Python versions crashed | SB3 pickles schedule closures | `INFERENCE_CUSTOM_OBJECTS` |
| BC clone oscillated forward/backward | Greedy uses Backward for 180° turns (bimodal labels) | Evaluate stochastically; BC kept as a last resort |

---

## 22. Known issues and caveats

**Training code:**
- **`train.py` imports:** run it as `python -m scripts.train`; the plain `python scripts/train.py` form fails to import.
- **`--curriculum` needs `--num-cartons 1`:** without it the environment starts at 12 cartons while the curriculum starts at level 1, and the first promotion drops it to 2.
- **Learning-rate "restart" on promotion does nothing:** it re-installs `linear_schedule`, which SB3 evaluates at global progress. The correct `restart_schedule` exists but isn't used.
- **Promotion changes the entropy coefficient:** it also sets `ent_coef = 0.02` (the same value as the default).
- **Two TensorBoard metrics are always 0:** `MetricsCallback` reads `info["pickups"]` / `info["deliveries"]`, but the vec envs provide `picked_up` / `delivered_by_me`. So `metrics/pickups_per_episode` and `metrics/deliveries_per_episode` are always 0. `cartons_delivered_per_episode` is correct.
- **`--gamma` doesn't reach the model:** it only affects `VecNormalize` and the environment; the model's γ is hard-coded to 0.999.
- **`--init-from` is broken for recurrent models:** it loads the donor with `PPO.load` and copies its state dict into a RecurrentPPO policy, which fails because of the LSTM keys. It only suits non-recurrent donors.

**Evaluation and tests:**
- **Collisions over-counted in `inference.py`:** it adds up the per-world collision count across the 4 slots, so it reports 4× the real number. `evaluate_ablation.py` counts once per world.
- **`test_post_training.py` can't load the trained models:** it uses `PPO.load`, which doesn't fit them, and it imports `scipy`, which isn't in `requirements.txt`.
- **No-comm training mode isn't a flag:** the no-comm arm's zeroed-message training mode is not exposed as a `train.py` flag. `--communication` always delivers messages. The no-comm model is evaluated with zeroed slots via `evaluate_ablation.py --variant nocomm`.

**Documentation:**
- **Parts of `repo_docs/` are out of date** (files 1, 2, 5, 6 and 8). They describe a continuous action space, a different observation layout, 40 LiDAR rays over 360° and different reward values. This walkthrough and `env.py` are authoritative.
- **Stale comments:** comments in `env.py`, `training.py` and `train.py` still quote the specification's 0.9 / 0.1 split and a −5 shared collision. The code uses 0.8 / 0.2 and −1 shared / −5 individual.

**Scope limits:**
- **Motion:** grid-based (cell moves, 90° turns) with interpolated kinematic motion, not torque control.
- **Critic:** decentralised (per-agent observation), not a centralised MAPPO critic.
- **Statistics:** one training run per arm.

---

## 23. Presentation material

| File | Content |
|---|---|
| `presentation-final-content.md` | slide-by-slide content |
| `presentation.md` | long-form reference, including the full reward mathematics |
| `HiveMind_presentation.pdf` | rendered 16:9 deck |
| `presentation_graphs/architecture_diagram.{png,svg}` | verified architecture diagram |
| `presentation_graphs/env_topdown.png` | top-down warehouse screenshot |
| `presentation_graphs/t1–t10_*_compare.png` | training comparisons: success, episode length, reward, entropy, KL, explained variance, curriculum, total loss, value loss, policy-gradient loss |
| `presentation_graphs/e1–e6_*.png` | success by carton count, steps to finish, collisions, intervention test, token usage, token vs robot state |
| `presentation_graphs/1–6_*.png` | earlier no-comm-only graphs |
| `presentation_videos/demo_comm.mp4`, `demo_nocomm.mp4` | full 12-carton episodes, top-down, same layout |

---

## 24. Git branches and contributors

**Branches:**

| Branch | Content |
|---|---|
| `multi-agent-rl` | This branch: the 4-robot environment, shared-policy RecurrentPPO, and the communication ablation |
| `single-agent-rl` | The default branch on origin; the original single-robot project (CNN grid observation, obstacle-level curriculum) from which the training utilities were ported |
| `multi-agent-v2` (origin) | A parallel multi-agent exploration branched on 2026-09-01: modular semantic extractor, anti-thrashing curriculum safeguards, heading-aware potential experiments |

- **History:** the repository began on 2026-08-27; this branch has 57 commits.
- **Contributors:** karmanyaiitj, Udayrajsinh Vala, codr-shiv and Het Thakkar.
- **Specification:** `legacy_archive/old_reference_and_media/MAWC_Technical_Specification.pdf` defines the reward table and the LiDAR / vocabulary targets the environment follows.
