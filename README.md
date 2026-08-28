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

The repository is currently a simulation and environment prototype: the world is built,
the learning system is not. Specifically, `_get_obs()` returns an empty list and no
`observation_space` is declared, rewards are hard-coded `[0, 0, 0, 0]`, `terminated` and
`truncated` are hard-coded `False` (so the 2000-step limit is not enforced by the env),
there is no LiDAR ray-casting or collision detection, and `train.py` and
`hivemind_env/models.py` are empty placeholders.

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

On the current code that yields `observation == []`, `reward == [0, 0, 0, 0]`,
`terminated is False`, `truncated is False`, and
`info == {"robot_pos": [...4 (x, y, z) tuples...], "remaining_resources": 12}`.
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
an observation placeholder (an empty list), a per-agent reward list, termination flags,
and an info dictionary containing robot positions and the number of remaining resources.
Observations, rewards and termination are roadmap steps 3 and 4 in `CLAUDE.md`.

## Files And Assets

```text
.
├── environment.yml             Conda environment definition (alternative, not in use)
├── requirements.txt            Python dependencies
├── pyproject.toml              Package metadata and core dependencies
├── CLAUDE.md                   Decisions already made, current state, and the roadmap
├── play_multi.py               Single-bot navigation, pickup, and depot-drop demo
├── smoke_test.py               Import / device / reset / step check; PASS-TODO-FAIL report
├── train.py                    Reserved for a future training pipeline; currently empty
├── scripts/
│   └── run_evaluation.py       Fixed-seed evaluation harness; makespan is the headline
└── hivemind_env/
    ├── env.py                  Gymnasium environment and warehouse generation
    ├── models.py               Reserved model module; currently empty
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
