# Communication Ablation

## Summary
Both arms of the communication ablation now run from the same code. A single flag, `--mute-messages`, picks the arm. You never check out an old commit to switch.

| Arm | Flags | What the robots see |
|---|---|---|
| **no-comm** (baseline) | `--communication --mute-messages` | message slots are always zero |
| **comm** | `--communication` | one-hot tokens from the 3 other robots |

In both arms every robot still picks a token each step, so the action space is the same, `[7 moves, 16 tokens]`. The network, rewards and hyperparameters are also identical. The **only** difference is whether tokens reach the other robots' observations (obs slots `[129:177]`).

## Why the baseline is "muted", not "communication off"
The existing 13.5M-step model (`models/ppo_recurrent_final.zip`) was trained with the `[7, 16]` action space, but its message slots were always zero. A check on that model confirmed it:

- Its tokens are close to random: 3.93 bits of entropy, where 4.0 is fully random.
- Zeroing the message slots doesn't change its results. It completed 4/4 episodes at 12 cartons either way.

`--mute-messages` reproduces exactly that condition. The existing model therefore stays a valid no-comm arm, and the comm arm differs from it in one variable only.

## What changed in the code
- `hivemind_env/env.py`: new `mute_messages=False` argument. When it is `True`, tokens are still decoded but never written into the message slots.
- `scripts/train.py`: new `--mute-messages` flag, which is refused without `--communication`. Default run names now end in `_comm` or `_nocomm`.
- `scripts/inference.py` and `scripts/evaluate_all.py`: new `--mute-messages` flag, passed through to the env.

Defaults are unchanged, so existing commands behave exactly as before.

Before this, the "fixed collisions" commit (71f4a4c) was reverted in `b9d707f`. Its safety shield in `inference.py` stopped robots from completing tasks: the model succeeded in 1 of 5 episodes with it and 5 of 5 without it.

## Checks run
- With the channel open, each robot sees 3 tokens. Muted, it sees 0. Everything else in the observation is identical between the two.
- The 11 tests in `tests/test_environment.py` pass.
- A short smoke-training run completes for both arms.
- The existing model with `--mute-messages` completed 3/3 episodes at 12 cartons, averaging about 141 steps.

## How to run

Run training as a module (`python -m scripts.train`). Running `python scripts/train.py` directly fails to find `hivemind_env`.

**Train the comm arm.** The settings below match the baseline's saved configuration:
```bash
python -m scripts.train --communication --curriculum --num-cartons 1 \
  --worlds 8 --n-steps 512 --batch-size 4096 \
  --timesteps 13500000 --seed 0 --run-name ppo_recurrent_comm_s0
```

This is a **full 12-carton run**, not a partial one. With `--curriculum`, `--num-cartons 1` is only the starting level. Training then moves up 1 → 2 → 4 → 8 → 12 cartons whenever the rolling success rate reaches 85%. The baseline's TensorBoard log (`tensorboard_logs/ppo_recurrent_30M_1`) shows the same ladder:

| Step | Cartons |
|---|---|
| 0 | 1 |
| 1.25M | 2 |
| 1.51M | 4 |
| 1.70M | 8 |
| **2.56M → 13.5M** | **12** |

About 11M of the baseline's 13.5M steps were at 12 cartons. The comm arm must use the same curriculum. Starting at 12 without it would change the training schedule as well as communication, and the ablation would no longer isolate a single variable.

**Optional, recommended:** retrain the no-comm arm from this same code by adding `--mute-messages`, with the same seed. The existing baseline was trained from code that was never committed.

**Evaluate both arms** with the same script:
```bash
python scripts/evaluate_all.py --episodes 50 --model-path models/ppo_recurrent_final.zip --mute-messages
python scripts/evaluate_all.py --episodes 50 --model-path models/ppo_recurrent_comm_s0_final.zip
```

Two things to keep in mind when comparing:
- **Collision numbers are 4× too high.** `inference.py` adds up one world's collisions once for each of the 4 robots. Both arms are affected equally, so comparisons between them are still fair.
- **Checking that the comm arm really uses its messages.** Zeroing its message slots should make it perform worse. Its token entropy should also end up well below 4 bits.
