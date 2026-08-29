# HiveMind Multi-Agent Warehouse

A Gymnasium environment built with PyBullet for experimenting with robot navigation and carton pickup in a generated warehouse. The current environment creates four differential-drive bots, shelf obstacles, two rows of carton resources, a depot, boundary walls, and a custom floor.

## Current Status

- Four bots are spawned in the warehouse.
- The warehouse is a 13 x 13 grid with 1 m cells.
- Six shelf rows are generated at grid rows 1, 3, 5, 7, 9, and 11.
- Shelves contain cartons on the lower two rows. The top shelf row is empty.
- Shelf segment lengths are randomized on every reset while preserving two gaps per shelf row.
- Twelve carton resources are placed in the gaps.
- The depot is at grid cell `(0, 0)` and is shown in semi-transparent black.
- The floor is light brown and the boundary walls are brown.
- Bot chassis and arm booms are metallic black; grippers and lidar are metallic grey; wheels retain their dark colour.
- Cartons are 0.5 m x 0.5 m x 0.5 m.
- The lidar starts at its initial height and raises to 0.5 m while a carton is carried.
- Pickup and drop are implemented as environment actions.

The environment is a complete RL problem and the training pipeline runs end to end. It
reports a pinned 177-float observation including a 72-ray LiDAR sweep, pays the reward
structure from `MAWC_Technical_Specification.pdf` §3, treats shelves and other robots as
solid obstacles that cost `-5.0` on contact, and ends episodes on completion or the step
limit. `train.py` trains a shared policy across all four robots and the saved checkpoint
loads back into the evaluation harness.

**The greedy baseline is the number to beat: makespan 98, 100% completion over 30 fixed
seeds.** No learned policy has been run against it yet, so no policy result exists.

The training scaffolding in `hivemind_env/training.py` and the evaluation harness in
`scripts/run_evaluation.py` are ported and adapted for four agents, but they cannot train
or score a policy until observations and rewards exist. `smoke_test.py` reports exactly
which of these are still outstanding. See `CLAUDE.md` for the ordered roadmap.

## Setup

The project runs from a virtual environment at the repository root. There is no activate
step in the documented workflow -- call the interpreter by path:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Then run everything as:

```powershell
.venv\Scripts\python.exe <script>
```

`hivemind_env` is installed editable, so imports resolve without setting `PYTHONPATH`.

The stack this is currently developed and tested against:

| Package | Version |
| --- | --- |
| Python | 3.14 |
| pybullet | 3.2.7 |
| gymnasium | 1.3.0 |
| numpy | 2.5.2 |
| torch | 2.13.0 |
| stable-baselines3 / sb3-contrib | 2.9.0 |

The core runtime dependencies are Gymnasium, NumPy, and PyBullet. `requirements.txt` also
contains the training and utility packages.

### Conda alternative

`environment.yml` is kept for anyone who prefers Conda. It is not the workflow in use and
pins an older Python (3.10), so it is not verified against the versions in the table above:

```bash
conda env create -f environment.yml
conda activate hivemind
```

### Smoke test

Confirm the environment imports, constructs, resets, and steps:

```powershell
.venv\Scripts\python.exe smoke_test.py
```

It reports three states -- `PASS` (works now), `TODO` (a roadmap item that is genuinely
not built yet), and `FAIL` (a real breakage). Only `FAIL` sets a non-zero exit code, so
the `TODO` lines double as a progress tracker for the roadmap in `CLAUDE.md`.

## Run The Pickup Demo

Run the demo from the repository root:

```powershell
.venv\Scripts\python.exe play_multi.py
```

The demo opens PyBullet in GUI mode, resets a randomized world, reads the actual resource positions from the environment, and uses bot 0 to:

1. Navigate to the first available carton using a grid path that avoids shelf cells.
2. Face the carton and execute pickup.
3. Stop after the pickup so the carried carton and raised lidar can be inspected.

The other three bots remain in their initial positions. Drop and depot delivery are not currently part of this demo. Despite the historical filename, `play_multi.py` is not currently a multi-bot choreography or an all-resource delivery script.

For a headless smoke test, create the environment with `render_mode=None` and call `reset()` and `step()` from Python:

```python
from hivemind_env.env import HiveMindMultiAgentEnv

env = HiveMindMultiAgentEnv(render_mode=None)
try:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step([6, 6, 6, 6])
finally:
    env.close()
```

On the current code that yields a `(4, 105)` float32 `observation`, four per-agent
rewards, `terminated`/`truncated` flags, and an `info` dict carrying robot positions,
remaining resources, delivery count, collisions, per-agent pickup/delivery/invalid-action
flags, and a full `reward_breakdown`.
Note that `env.close()` is not yet idempotent -- a second call raises
`pybullet.error("Not connected to physics server.")`.

## Environment API

The environment uses a four-element `MultiDiscrete` action space. One action is supplied for each bot:

| Action | Meaning |
| --- | --- |
| `0` | Move forward one grid cell |
| `1` | Move backward one grid cell |
| `2` | Turn left |
| `3` | Turn right |
| `4` | Pick up the nearest carton within range |
| `5` | Drop the carried carton near the depot |
| `6` | Stay in place |

Each action is executed with physics substeps for smooth motion. The environment returns
an observation, a per-agent reward list, termination flags, and an info dictionary
carrying robot positions, remaining resources, delivery count and the observation width.

### Observation

`observation_space` is `Box(-1.0, 1.0, (4, 177), float32)` — one row per robot, every
value normalised into `[-1, 1]`. **The width is pinned at 177 and must not change**: it is
baked into a trained policy's input layer, so moving it invalidates every saved
checkpoint. Adding a component means a new `OBS_DIM_V4`, never an edit to V3.

| slice | size | component |
| --- | --- | --- |
| `[ 0: 3]` | 3 | own pose (x, y, heading) |
| `[ 3: 5]` | 2 | own velocity, cells per step |
| `[ 5: 6]` | 1 | own carrying flag |
| `[ 6:15]` | 9 | other robots' poses |
| `[15:18]` | 3 | other robots' carrying flags |
| `[18:30]` | 12 | carton status (available / mine / other's / delivered) |
| `[30:54]` | 24 | carton positions — 12 × (x, y), same index as the status slots |
| `[54:56]` | 2 | depot direction |
| `[56:57]` | 1 | elapsed time |
| `[57:129]` | 72 | LiDAR — 270° arc, 0.1–10 m, Gaussian noise |
| `[129:177]` | 48 | message slots — reserved, all zero until communication lands |

The message slots are reserved now precisely so that adding communication later does not
change the width. The full component table, the encoding of each field and the reasoning
behind every choice live at the top of `hivemind_env/env.py`.

Check it end to end — this drives a robot through a full pickup and delivery and verifies
each component against PyBullet ground truth:

```powershell
.venv\Scripts\python.exe scripts/verify_observations.py
```

### Rewards

Transcribed from `MAWC_Technical_Specification.pdf` §3. Shared rewards (90% weight) are
identical for all four agents; individual rewards (10%) are per agent.

| Scope | Event | Reward |
| --- | --- | --- |
| Shared | All 12 delivered | `+100.0` once per episode |
| Shared | Per delivery (any bot) | `+10.0` |
| Shared | Makespan bonus | `+50 × (T_max − T_actual) / T_max`, once per episode |
| Shared | Collision (any robot pair) | `−5.0` per event |
| Shared | Time | `−0.05` per step |
| Individual | Own pickup | `+1.0` |
| Individual | Own delivery | `+2.0` |
| Individual | Idle (not moving, not at depot) | `−0.02` per step |
| Individual | Invalid action | `−0.5` |

`R_total_i = 0.90 × R_shared + 0.10 × R_individual_i`

An episode `terminated`s when all 12 cartons are delivered and `truncated`s at
`max_steps` (2000). `T_max` is `max_steps` — this environment counts steps, not seconds.

The spec's replanning penalty is **not** implemented: it fires when A* re-runs, and this
environment has no planner (see `CLAUDE.md`, decision 3). The constant is defined and
left unused so the omission is visible.

Check it end to end — this drives a robot through a full delivery, printing the reward
and its breakdown every step, then verifies each term against the spec:

```powershell
.venv\Scripts\python.exe scripts/verify_rewards.py
```

## Greedy baseline

A scripted controller — each robot claims the nearest unclaimed carton, delivers it, and
repeats. Its makespan is the reference every learned policy is quoted against, and it is
scored through the same harness, seeds and metrics as a policy so the comparison is
valid.

```powershell
.venv\Scripts\python.exe scripts/run_evaluation.py --baseline greedy --episodes 30
```

| Metric | Value |
| --- | --- |
| Makespan | mean 97.6, median 96.5, range 82–123 |
| Completion | 30/30 seeds |
| Distance | 230.9 m per episode |
| Collisions | 6.3 per episode |

Results are written to `docs_analysis/greedy_baseline.json`.

## Training

All four robots share one set of weights. `HiveMindSharedPolicyVecEnv` presents each
four-robot world as four policy-facing slots, so PPO trains on the pooled experience of
every robot in every world.

```powershell
# smoke run - ~4k steps, exercises the whole pipeline in about two minutes
.venv\Scripts\python.exe train.py --smoke

# a real run
.venv\Scripts\python.exe train.py --timesteps 5000000 --worlds 8
```

Throughput is roughly 65 robot-steps/s, or 34/s with PPO updates included, so a 2M-step
run is around 16 hours. The 30 physics substeps per environment step dominate — the LiDAR
sweep is only about 4 ms of a 61 ms step. Note that `--worlds` increases batch diversity
but **not** throughput: worlds are stepped sequentially in one process.

TensorBoard is optional. If it is not installed, training runs and simply writes no
curves rather than failing.

This is parameter sharing with a decentralised critic, not MAPPO — `hivemind_env/vec_env.py`
explains what that trade buys and when to replace it.

## Files And Assets

```text
.
├── environment.yml             Conda environment definition (alternative, not in use)
├── requirements.txt            Python dependencies
├── pyproject.toml              Package metadata and core dependencies
├── CLAUDE.md                   Decisions already made, current state, and the roadmap
├── play_multi.py               Single-bot navigation, pickup, and depot-drop demo
├── smoke_test.py               Import / device / reset / step check; PASS-TODO-FAIL report
├── train.py                    Shared-policy PPO training entry point
├── scripts/
│   ├── run_evaluation.py       Fixed-seed evaluation harness; makespan is the headline
│   ├── verify_observations.py  Drives a full delivery, checking every observation field
│   └── verify_rewards.py       Prints reward per step and checks it against the spec
└── hivemind_env/
    ├── env.py                  Gymnasium environment and warehouse generation
    ├── greedy.py               Scripted baseline controller; the makespan to beat
    ├── vec_env.py              4 robots -> 4 policy slots sharing one set of weights
    ├── models.py               HiveMindExtractor: MLP + 1-D CNN over the LiDAR sweep
    ├── training.py             Shared scaffolding: curriculum callback, LR schedules,
    │                           env factory, device probe, version-safe policy loader
    └── assets/
        ├── carton.urdf         0.5 m carton resource
        ├── diff_drive_bot.urdf Shared four-bot model
        ├── generate_shelves.py Shelf URDF generator
        ├── shelf_1m.urdf       Generated 1 m shelf
        ├── shelf_2m.urdf       Generated 2 m shelf
        ├── shelf_3m.urdf       Generated 3 m shelf
        ├── shelf_4m.urdf       Generated 4 m shelf
        ├── shelf_5m.urdf       Generated 5 m shelf
        ├── shelf_6m.urdf       Generated 6 m shelf
        ├── shelf_7m.urdf       Generated 7 m shelf
        └── shelf.urdf          Legacy shelf asset, not used by env.py
```

To regenerate the generated shelf URDFs after changing the shelf or carton layout:

```powershell
.venv\Scripts\python.exe hivemind_env/assets/generate_shelves.py
```

## Development Notes

- Run commands from the repository root so relative asset paths resolve correctly.
- Invoke the interpreter as `.venv\Scripts\python.exe`. There is no activate step; do not
  create a second environment or install packages without checking first.
- The environment seeds Python's `random` module and NumPy when `reset(seed=...)` is called, making shelf segmentation reproducible for a given seed.
- PyBullet GUI shutdown with `Ctrl+C` is expected to produce exit code 130.
- The shelf URDFs inline the carton geometry and visual details. PyBullet may emit inertial warnings if older generated assets are present; regenerate the shelves after modifying the generator.
