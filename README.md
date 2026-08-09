# HiveMind: Single-Agent RL Environment

This directory contains the single-agent reinforcement learning environment for the HiveMind project. The environment is built on top of Gymnasium and PyBullet, serving as the foundational stepping stone before scaling to the full multi-agent setup.

## Project Structure

The project follows this structure to ensure easy installation and modularity:

```
HiveMind_MidEval/
├── hivemind_env/                 # the package
│   ├── env.py                    # the RL environment (physics, LiDAR, reward)
│   ├── models.py                 # custom CNN feature extractor
│   ├── training.py               # shared schedule / curriculum / env factory / loader
│   ├── __init__.py               # registers HiveMind-SingleAgent-v0
│   └── assets/diff_drive_bot.urdf
├── models/
│   ├── ppo_hivemind_v1_final.zip # the trained policy everything loads (10M steps)
│   └── checkpoints/              # one RecurrentPPO checkpoint, kept as the failed baseline
├── scripts/
│   ├── run_evaluation.py         # scored evaluation -> docs_analysis/evaluation_results.json
│   └── test_*.py                 # perception / arm / end-to-end mechanics checks
├── docs_analysis/                # numbered analysis docs + evaluation_results.json
├── tensorboard_logs*/            # the runs behind the training figures
├── play_demo.py                  # GUI demo of the trained policy  <- start here
├── train.py / train_ppo_v2.py / train_v2.py
├── smoke_test.py
├── pyproject.toml · requirements.txt · environment.yml
└── .venv/                        # ready-to-use virtualenv (not committed)
```

## Installation (Important)

Due to PyBullet's heavy C++ physics engine requirements, a standard `pip install` may fail on Windows if you do not have Microsoft Visual Studio C++ compilers correctly configured. To guarantee cross-platform compatibility without compiler issues, this project uses **Miniconda** to fetch pre-compiled binaries.

1. **Install Miniconda:** If you don't have it, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd HiveMind_MidEval
   ```
3. **Create the isolated environment:**
   ```bash
   conda env create -f environment.yml
   ```
   `environment.yml` now includes the full ML stack (PyTorch, Stable-Baselines3,
   sb3-contrib, TensorBoard), so no separate `conda install` step is needed.
4. **Activate it and install the package itself:**
   ```bash
   conda activate hivemind
   pip install -e .
   ```

## Observation window sizes (important)

The observation grid size is a **per-run choice**, and a saved model can only be loaded
back into an env built at the same size — the CNN's flatten width is baked into the
weights (15x15 → 578 features, 21x21 → 1602).

| Constant | Value | Used by |
|---|---|---|
| `OBS_SIZE_V1` | 15 | every model currently in `models/`, `train.py`, `train_v2.py` |
| `OBS_SIZE_V2` | 21 | `train_ppo_v2.py` — the wider window that addresses level-4 collisions |

Import them from `hivemind_env.env` and pass `obs_size=` explicitly rather than relying
on the default.

## Quick start — watch the trained robot

A working `.venv` already exists in the repo root. Every command below uses it directly,
so **no activation step is needed**.

```powershell
.venv\Scripts\python.exe play_demo.py
```

That opens a PyBullet window and runs 3 episodes at difficulty 3 with the trained v1
policy — the robot drives to the green block, picks it up with the arm, and delivers it
to the red depot. Coloured LiDAR rays and the grid overlay are drawn live.

Expected output:

```
Loading model: models/ppo_hivemind_v1_final.zip
Starting 3 episodes of Level 3...
Episode 1 finished! Steps: 39, Reward: 33.76
Episode 2 finished! Steps: 23, Reward: 28.63
Episode 3 finished! Steps: 48, Reward: 43.10
Demo complete!
```

### Other things you can run

| Command | What it does | Time |
|---|---|---|
| `.venv\Scripts\python.exe smoke_test.py` | Headless health check: imports, device, reset/step, extractor. Exits non-zero on failure. | ~5 s |
| `.venv\Scripts\python.exe scripts\run_evaluation.py --episodes 30` | Full scored evaluation, 4 levels, writes `docs_analysis/evaluation_results.json` | ~37 min |
| `.venv\Scripts\python.exe scripts\run_evaluation.py --episodes 5 --out docs_analysis/eval_smoke.json` | Quick scored run. **Use `--out`** so it doesn't replace the authoritative results. | ~1 min |
| `.venv\Scripts\python.exe test_run.py` | Scripted BFS mission in the GUI — no policy, verifies mechanics | ~1 min |
| `.venv\Scripts\python.exe scripts\test_robot_mechanics_end_to_end.py --seed 42` | A* navigation + rigorous LiDAR global-transform assertions | ~1 min |
| `.venv\Scripts\python.exe test_recurrent_demo.py` | The failed LSTM baseline — scores 0% everywhere, which is the point | ~10 min |
| `.venv\Scripts\python.exe extract_tb_data.py` | Re-extracts the PPO_v1_5 TensorBoard scalars to a text report | ~5 s |

> The slide deck is maintained outside this repo. `docs_analysis/evaluation_results.json`
> is the authoritative source for every number quoted in it — regenerate that with
> `run_evaluation.py` and copy the figures across by hand.

### Troubleshooting the GUI demo

**"Not connected to physics server" / the window vanishes mid-episode.** Closing the
PyBullet viewer terminates its physics server, so the next step raises. This is now
reported as a clean message and the demo stops early rather than dumping a traceback —
just re-run it.

If the viewer dies *on its own* before finishing all three episodes, it is usually the
GPU driver struggling with the live LiDAR ray overlay (36 debug lines redrawn every
step; a 500-step timeout episode redraws ~18,000 times). Turn the overlay off:

```python
env = HiveMindSingleAgentEnv(render_mode="human", difficulty_level=3,
                             obs_size=OBS_SIZE_V1, show_lidar=False)
```

The policy behaves identically — `show_lidar` only controls the debug-line rendering.

### If you need to rebuild the environment from scratch

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The `-e .` step is what makes `import hivemind_env` work from the `scripts/` directory.

## Training the AI (Phase 2)

The environment integrates with **Stable-Baselines3** and **PyTorch** to train a Proximal Policy Optimization (PPO) agent. The training script uses a Custom CNN to process the 15x15 LiDAR grid, and a `CurriculumCallback` to automatically increase the difficulty as the AI learns.

Training spawns up to 16 parallel environments (`min(16, os.cpu_count())`):
```powershell
.venv\Scripts\python.exe train.py          # v1 config: 15x15 window, 256-dim, 10M steps
.venv\Scripts\python.exe train_ppo_v2.py   # v2 config: 21x21 window, 512-dim, 20M steps
```

### Monitoring Training
To visualize the learning curve, open a second terminal and point TensorBoard at the log
directory for the run you started — `train.py` writes to `tensorboard_logs_ppo_v1/`,
`train_ppo_v2.py` to `tensorboard_logs_ppo_v2/`, and `train_v2.py` (RecurrentPPO) to
`tensorboard_logs/`:
```powershell
.venv\Scripts\python.exe -m tensorboard.main --logdir=./tensorboard_logs_ppo_v1/
```
Navigate to `http://localhost:6006` in your browser. The completed 10M-step run is
already there as `PPO_v1_5`.

## Next Steps (Phase 3 & 4)

With the Single-Agent environment completely operational (perfect physics, LiDAR, and PBRS) and the PPO architecture established, future phases will focus on:
1. **Multi-Agent Scaling**: Transitioning the single robot to a true "HiveMind" swarm of independent agents.
2. **Advanced Physics Integration**: Introducing surface friction variations (mud, ice) to test robust routing policies.
