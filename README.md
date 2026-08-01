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

## Installation (Important)

Due to PyBullet's heavy C++ physics engine requirements, a standard `pip install` may fail on Windows if you do not have Microsoft Visual Studio C++ compilers correctly configured. To guarantee cross-platform compatibility without compiler issues, this project uses **Miniconda** to fetch pre-compiled binaries.

1. **Install Miniconda:** If you don't have it, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Single-Agent-implementation
   ```
3. **Create the isolated environment:**
   ```bash
   conda env create -f environment.yml
   ```
4. **Activate the environment:**
   ```bash
   conda activate hivemind
   ```

## Running the Test Environment

To verify that the PyBullet physics engine is loading correctly:
```bash
python test_env.py
```

## Next Steps

This repository provides a clean API skeleton. The team needs to build the following components into `env.py`:
1. **PyBullet Physics**: Connect to PyBullet and spawn the robot, obstacles, resource, and depot.
2. **Action Kinematics**: Map the discrete actions (0-5) to differential drive wheel velocities.
3. **LiDAR & Observation**: Implement raycasting to construct the `15x15x5` egocentric occupancy grid.
4. **Reward Logic**: Implement distance shaping and pickup/delivery logic.
