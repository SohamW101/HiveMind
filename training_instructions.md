# Training Instructions: Communication Run

## 1. Get the code

```bash
git clone https://github.com/SohamW101/HiveMind.git
cd HiveMind
git checkout multi-agent-rl
git pull
git log -1 --oneline        # must show: c3eab42 fixed collisions
```

If the repo is already on the server, just `cd` into it and run the last three commands.

## 2. Create the venv (once)

```bash
python3.12 -m venv venv     # baseline was trained on Python 3.12
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt tensorboard
```

## 3. Check the setup

```bash
source venv/bin/activate
nproc                       # must be >= 8
python -c "import torch, pybullet, stable_baselines3, sb3_contrib, tensorboard; print('ok', torch.cuda.is_available())"
python -m scripts.train --smoke --communication --num-cartons 1
```

The smoke run must end with a `saved :` line.

## 4. Train (from the repo root)

```bash
tmux new -s comm
source venv/bin/activate
python -m scripts.train --communication --curriculum --num-cartons 1 \
  --worlds 8 --timesteps 30000000 --seed 0 --run-name ppo_recurrent_comm_s0
```

- Detach: `Ctrl+B`, then `D`
- Reattach: `tmux attach -t comm`

Rules:
- Use `python -m scripts.train`, not `python scripts/train.py` (the latter fails with `No module named 'hivemind_env'`).
- Do not change or drop any flag. `--num-cartons 1` and `--timesteps 30000000` must match the baseline.

## 5. Monitor

```bash
source venv/bin/activate
tensorboard --logdir tensorboard_logs --port 6006
```

On your laptop: `ssh -L 6006:localhost:6006 <user>@<server>`, then open `http://localhost:6006`.

## 6. Output

| What | Where |
|---|---|
| Checkpoints (every 249,984 steps) | `models/checkpoints/ppo_recurrent_comm_s0_<steps>_steps.zip` |
| Final model | `models/ppo_recurrent_comm_s0_final.zip` |
| Logs | `tensorboard_logs/ppo_recurrent_comm_s0_1/` |

Compare against the baseline (`models/ppo_recurrent_final.zip`, 13,499,136 steps) using `models/checkpoints/ppo_recurrent_comm_s0_13499136_steps.zip`, which is the same step count.

`Ctrl+C` stops training and still saves the final model.
