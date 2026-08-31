"""
Behaviour cloning from the greedy controller - a warm start for PPO.

READ THIS FIRST: THE PREMISE OF THIS SCRIPT WAS WRONG

It was written on the theory that PPO could not *find* a delivery, and that a warm start
would skip the discovery problem. That theory was tested on 2026-08-31 and is false.

A uniformly random policy already achieves **3.0 pickups and 1.3 deliveries per 400-step
episode** at 4 cartons. Delivery was never hard to stumble into. What made it worthless
was the price of the stumbling: 105.8 collision events per episode, 93.8 of them against
shelving, at -4.5 shared each. The reward was reachable and simply outweighed 476 to 12.

Cloning greedy does not change that ratio, which is why this script did not work either:

    BC, 76.4% action match  ->  0/10 completed
      deterministic: fwd 51% / back 49%, oscillating on the spot
      stochastic   : 2.1/4 delivered, never finished

The oscillation is not a bug in the clone. Greedy uses a single `backward` action for a
180-degree turn, so from one observation the demonstrator sometimes says forward and
sometimes backward; cross-entropy averages the two modes and the argmax flips between
them. **If you use this script, evaluate the clone stochastically.**

The actual fixes were to the environment and the shaping - shelves became unenterable and
the potential stopped punishing pickups. See CLAUDE.md step 6 and TRAINING.md section 11.

WHEN THIS IS STILL WORTH RUNNING

As step 4 of the escalation ladder, after raising `shaping_scale`, starting the curriculum
lower, and raising `ent_coef` have all failed - not before. A warm start is the right tool
for a genuine exploration problem, and this may yet become one at 12 cartons where a
random policy's delivery rate is far lower than at 4.

WHAT THIS DOES

Rolls out the greedy controller, records (observation, action) for every robot at every
step, and fits the policy's action distribution to those actions by cross-entropy. The
result is a checkpoint in exactly the format train.py loads, so fine-tuning is:

    .venv\\Scripts\\python.exe scripts/pretrain_bc.py --episodes 60 --num-cartons 4
    .venv\\Scripts\\python.exe train.py --init-from models/bc_pretrained.zip --num-cartons 4

WHAT IT IS NOT

Cloning greedy caps the policy at greedy's behaviour, not above it - the point is to
start PPO somewhere the reward gradient is informative, not to hand it the answer. A
policy that merely matches greedy has demonstrated nothing; the claim only starts when
fine-tuning beats 98 (or 23 at four cartons). Report the BC score and the fine-tuned
score separately, or the improvement is unreadable.

The critic is NOT cloned - there is nothing to clone it from. It starts fresh and PPO
fits it during fine-tuning, which is why the first few fine-tuning iterations will look
worse before they look better.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from stable_baselines3 import PPO

from hivemind_env.env import NUM_AGENTS, HiveMindMultiAgentEnv
from hivemind_env.greedy import GreedyController
from hivemind_env.models import DEFAULT_POLICY_KWARGS
from hivemind_env.training import get_device
from hivemind_env.vec_env import HiveMindSharedPolicyVecEnv

ACTION_NAMES = ["fwd", "back", "turnL", "turnR", "PICKUP", "DROP", "stay"]


def collect(episodes: int, num_cartons: int, seed0: int, verbose: bool = True):
    """Roll out greedy, returning (observations, actions) over every robot-step."""
    obs_buf, act_buf = [], []
    completed, makespans = 0, []

    for ep in range(episodes):
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons)
        obs, _ = env.reset(seed=seed0 + ep)
        ctrl = GreedyController(env)
        terminated = False

        for _ in range(env.max_steps):
            actions = ctrl.act()
            # Record the state the controller saw, paired with what it chose.
            for i in range(NUM_AGENTS):
                obs_buf.append(obs[i].copy())
                act_buf.append(int(actions[i]))
            obs, _, terminated, truncated, _ = env.step(actions)
            ctrl.sync_after_step()
            if terminated or truncated:
                break

        if terminated:
            completed += 1
            makespans.append(env.current_step)
        env.close()

        if verbose and (ep + 1) % 10 == 0:
            print(f"  {ep + 1:3d}/{episodes} episodes, {len(obs_buf):,} samples", flush=True)

    ms = float(np.mean(makespans)) if makespans else float("nan")
    return (np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.int64),
            completed, ms)


def main():
    ap = argparse.ArgumentParser(description="Behaviour-clone the greedy controller")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--num-cartons", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=5000,
                    help="Demo seeds start here. Kept clear of the 1000-1029 evaluation "
                         "seeds so the policy is not cloned on the seeds it is scored on.")
    ap.add_argument("--out", default="models/bc_pretrained.zip")
    args = ap.parse_args()

    device = get_device()
    print("=" * 74)
    print("  Behaviour cloning from the greedy controller")
    print("=" * 74)
    print(f"  episodes    : {args.episodes} at {args.num_cartons} cartons")
    print(f"  demo seeds  : {args.seed}..{args.seed + args.episodes - 1} "
          f"(evaluation uses 1000-1029, kept separate)")
    print(f"  device      : {device}\n", flush=True)

    t0 = time.time()
    obs, acts, completed, ms = collect(args.episodes, args.num_cartons, args.seed)
    print(f"\n  collected {len(obs):,} robot-steps in {time.time() - t0:.0f}s")
    print(f"  demonstrator: {completed}/{args.episodes} episodes completed, "
          f"mean makespan {ms:.1f}")
    counts = np.bincount(acts, minlength=7)
    print("  action mix  : " + "  ".join(
        f"{ACTION_NAMES[i]} {counts[i] / len(acts):.1%}" for i in range(7)))
    if completed < args.episodes:
        print("  WARNING: the demonstrator did not finish every episode - cloning a "
              "controller that fails will teach the policy to fail too.")

    # A throwaway env just to give PPO the right spaces; it is never stepped.
    vec = HiveMindSharedPolicyVecEnv(num_worlds=1, num_cartons=args.num_cartons)
    model = PPO("MlpPolicy", vec, policy_kwargs=DEFAULT_POLICY_KWARGS,
                device=device, verbose=0)

    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    obs_t = torch.as_tensor(obs, device=policy.device)
    act_t = torch.as_tensor(acts, device=policy.device)
    n = len(obs_t)

    print(f"\n  fitting {sum(p.numel() for p in policy.parameters()):,} parameters")
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=policy.device)
        tot_loss, tot_correct = 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            dist = policy.get_distribution(obs_t[idx])
            logits = dist.distribution.logits
            loss = F.cross_entropy(logits, act_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(1) == act_t[idx]).sum().item()
        print(f"    epoch {epoch + 1:2d}/{args.epochs}  loss {tot_loss / n:.4f}  "
              f"action match {tot_correct / n:.1%}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    vec.close()

    print(f"\n  saved {args.out}")
    print(f"  wall clock {(time.time() - t0) / 60:.1f} min")
    print("\n  Fine-tune with:")
    print(f"    .venv\\Scripts\\python.exe train.py --init-from {args.out} "
          f"--num-cartons {args.num_cartons}")
    print("\n  Score the clone BEFORE fine-tuning too - the improvement is only")
    print("  readable if both numbers are reported.")


if __name__ == "__main__":
    main()
