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
├── train.py              # Master PPO training script using Stable-Baselines3
└── hivemind_env/
    ├── __init__.py       # Registers the Gym environment
    ├── models.py         # Custom CNN PyTorch feature extractor
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
4. **Activate the environment and install ML libraries:**
   ```bash
   conda activate hivemind
   conda install -y -c conda-forge stable-baselines3 tensorboard pytorch
   ```

## Running the Test Environment

To verify that the PyBullet physics engine is loading correctly:
```bash
python test_env.py
```

To verify full environment mechanics (smooth substep navigation, 360deg LiDAR occupancy grid, pickup & delivery logic):

```bash
python test_run.py
```

## Training the AI (Phase 2)

The environment integrates with **Stable-Baselines3** and **PyTorch** to train a Proximal Policy Optimization (PPO) agent. The training script uses a Custom CNN to process the 15x15 LiDAR grid, and a `CurriculumCallback` to automatically increase the difficulty as the AI learns.

To begin training across 4 parallel CPU cores:
```bash
python train.py
```

### Monitoring Training
To visualize the AI's learning curve (rewards, episode length, etc.), open a second terminal and run TensorBoard:
```bash
tensorboard --logdir=./tensorboard_logs/
```
Navigate to `http://localhost:6006` in your browser.

## Next Steps (Phase 3 & 4)

With the Single-Agent environment completely operational (perfect physics, LiDAR, and PBRS) and the PPO architecture established, future phases will focus on:
1. **Multi-Agent Scaling**: Transitioning the single robot to a true "HiveMind" swarm of independent agents.
2. **Advanced Physics Integration**: Introducing surface friction variations (mud, ice) to test robust routing policies.
