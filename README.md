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

The repository is currently a simulation and environment prototype. `train.py` and `hivemind_env/models.py` are empty placeholders; no training pipeline is included yet.

## Setup

The recommended environment is Conda:

```bash
conda env create -f environment.yml
conda activate hivemind
```

To install the project dependencies with pip instead:

```bash
python -m pip install -r requirements.txt
```

The main runtime dependencies are Gymnasium, NumPy, and PyBullet. `requirements.txt` also contains utility packages for future experiments.

## Run The Pickup Demo

Run the demo from the repository root:

```bash
conda activate hivemind
python play_multi.py
```

The demo opens PyBullet in GUI mode, resets a randomized world, reads the actual resource positions from the environment, and uses bot 0 to:

1. Navigate to the first available carton using a grid path that avoids shelf cells.
2. Face the carton and execute pickup.
3. Return to a cell adjacent to the depot.
4. Face the depot and execute drop without entering the depot cell.

The other three bots remain in their initial positions. Despite the historical filename, `play_multi.py` is not currently a multi-bot choreography or an all-resource delivery script.

For a headless smoke test, create the environment with `render_mode=None` and call `reset()` and `step()` from Python:

```python
from hivemind_env.env import HiveMindMultiAgentEnv

env = HiveMindMultiAgentEnv(render_mode=None)
try:
    env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step([6, 6, 6, 6])
finally:
    env.close()
```

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

Each action is executed with physics substeps for smooth motion. The environment returns an observation placeholder, a per-agent reward list, termination flags, and an info dictionary containing robot positions and the number of remaining resources.

## Files And Assets

```text
.
├── environment.yml             Conda environment definition
├── requirements.txt            Python dependencies
├── pyproject.toml              Package metadata and core dependencies
├── play_multi.py               Single-bot navigation, pickup, and depot-drop demo
├── train.py                    Reserved for a future training pipeline
└── hivemind_env/
    ├── env.py                  Gymnasium environment and warehouse generation
    ├── models.py               Reserved model module; currently empty
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

```bash
conda activate hivemind
python hivemind_env/assets/generate_shelves.py
```

## Development Notes

- Run commands from the repository root so relative asset paths resolve correctly.
- Use `conda activate hivemind` before running Python commands in this project.
- The environment seeds Python's `random` module and NumPy when `reset(seed=...)` is called, making shelf segmentation reproducible for a given seed.
- PyBullet GUI shutdown with `Ctrl+C` is expected to produce exit code 130.
- The shelf URDFs inline the carton geometry and visual details. PyBullet may emit inertial warnings if older generated assets are present; regenerate the shelves after modifying the generator.
