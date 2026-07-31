# HiveMind: Single-Agent RL Environment

This directory contains the single-agent reinforcement learning environment for the HiveMind project. The environment is built on top of Gymnasium and PyBullet, serving as the foundational stepping stone before scaling to the full multi-agent setup.

## Project Structure

The project follows this structure to ensure easy installation and modularity:

```
Single-Agent-implementation/
├── .env                  # Environment configuration variables
├── .gitignore            # Git ignore list
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Package definition for pip installation
├── scripts/
│   └── test_env.py       # Script to test if the environment loads correctly
└── hivemind_env/
    ├── __init__.py       # Registers the Gym environment
    └── env.py            # The core Reinforcement Learning environment class
```

## Setup & Installation

To install the environment so it can be imported from anywhere on your machine, navigate to this directory and run:

```bash
pip install -r requirements.txt
pip install -e .
```

You can verify the installation by running the test script:

```bash
python scripts/test_env.py
```

## Next Steps

This repository provides a clean API skeleton. The team needs to build the following components into `env.py`:
1. **PyBullet Physics**: Connect to PyBullet and spawn the robot, obstacles, resource, and depot.
2. **Action Kinematics**: Map the discrete actions (0-5) to differential drive wheel velocities.
3. **LiDAR & Observation**: Implement raycasting to construct the `15x15x5` egocentric occupancy grid.
4. **Reward Logic**: Implement distance shaping and pickup/delivery logic.
