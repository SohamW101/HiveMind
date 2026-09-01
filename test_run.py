"""
Watch a trained policy drive the warehouse, in the GUI.

WHY THERE IS A --stochastic FLAG

This used to call `model.predict(obs, deterministic=True)` and nothing else, which is
the argmax of the action distribution. On this branch argmax and sampling disagree
sharply, and argmax is the one that lies:

    mini_final.zip, 1 carton, 10 episodes
      deterministic:  0/10 complete, 0 pickups, and in 760 robot-steps spent standing
                      within reach of a carton it pressed DROP 96% of the time
      stochastic   :  4/10 complete, 0.7 pickups per episode - the same weights

A policy whose argmax collapses onto one action looks exactly like a robot that drives
to a carton and freezes beside it. So test both before concluding anything about a
checkpoint, and treat a deterministic run alone as inconclusive.

The counts printed at the end are the ones that matter for that failure: pickups and
deliveries, not just whether the episode was a success. An episode ends only when every
carton is delivered, so `is_success` is a conjunction over four robots and reports False
for a policy that delivered eleven of twelve just as loudly as for one that never moved.

    .venv\\Scripts\\python.exe test_run.py models/my_run_final.zip --num-cartons 12
    .venv\\Scripts\\python.exe test_run.py models/my_run_final.zip --stochastic --episodes 3

For numbers rather than a picture, use scripts/probe_pickup.py (does it press PICKUP
when a carton is in reach?) and scripts/probe_policy.py (what does it do overall?) -
both run headless and report deterministic and stochastic side by side.
"""
import argparse
import time

import numpy as np
import pybullet as pb
from stable_baselines3 import PPO

from hivemind_env.env import NUM_AGENTS, HiveMindMultiAgentEnv
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS

ACTION_NAMES = ["fwd", "back", "turnL", "turnR", "PICKUP", "DROP", "stay"]


def main():
    p = argparse.ArgumentParser(description="Watch a trained policy in the GUI")
    p.add_argument("model", help="path to a saved checkpoint, e.g. models/run_final.zip")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the action distribution instead of taking the "
                        "argmax. Run both - see the module docstring.")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--num-cartons", type=int, default=None,
                   help="cartons in play (env default 12). Use the count the policy "
                        "was trained at, or the comparison means nothing.")
    p.add_argument("--seed", type=int, default=None,
                   help="fixed seed for episode 1; later episodes step upward from it")
    p.add_argument("--delay", type=float, default=0.03,
                   help="seconds to sleep per step so it is watchable; 0 for full speed")
    p.add_argument("--headless", action="store_true",
                   help="no GUI - just the numbers")
    args = p.parse_args()

    deterministic = not args.stochastic
    print(f"Loading model weights from: {args.model}")
    model = PPO.load(args.model, custom_objects=INFERENCE_CUSTOM_OBJECTS, device="cpu")

    print("Initializing environment...")
    env = HiveMindMultiAgentEnv(
        render_mode=None if args.headless else "human",
        num_cartons=args.num_cartons,
    )
    print(f"  mode      : {'deterministic (argmax)' if deterministic else 'stochastic (sampled)'}")
    print(f"  cartons   : {env.active_cartons if hasattr(env, 'active_cartons') else args.num_cartons or 12}")
    print(f"  max steps : {env.max_steps}")

    counts = np.zeros(7, dtype=int)
    completed = 0
    totals = {"pickups": 0, "deliveries": 0, "steps": 0}

    try:
        for ep in range(args.episodes):
            seed = None if args.seed is None else args.seed + ep
            obs, info = env.reset(seed=seed)

            if not args.headless:
                # Top-down view, as in play_multi.py.
                pb.resetDebugVisualizerCamera(
                    cameraDistance=16.0,
                    cameraYaw=0,
                    cameraPitch=-89.9,
                    cameraTargetPosition=[0, 0, 0],
                )

            done = False
            steps = 0
            pickups = 0
            deliveries = 0
            print(f"\nEpisode {ep + 1}/{args.episodes}"
                  + (f" (seed {seed})" if seed is not None else "") + " ...")

            while not done:
                # obs is (4, 177) - one row per robot. PPO reads that as a batch of 4
                # observations and returns 4 actions, which is exactly the joint action
                # the env wants.
                actions, _ = model.predict(obs, deterministic=deterministic)
                actions = np.asarray(actions).reshape(-1)[:NUM_AGENTS].astype(int)
                for a in actions:
                    counts[int(a)] += 1

                obs, rewards, terminated, truncated, info = env.step(actions)
                pickups += sum(1 for x in info["pickups"] if x)
                deliveries += sum(1 for x in info["deliveries"] if x)
                done = terminated or truncated
                steps += 1
                if args.delay:
                    time.sleep(args.delay)

            completed += int(terminated)
            totals["pickups"] += pickups
            totals["deliveries"] += deliveries
            totals["steps"] += steps
            print(f"  {steps} steps | delivered {info['delivered']}/{env.active_cartons}"
                  f" | pickups {pickups} | deliveries {deliveries}"
                  f" | {'COMPLETE' if terminated else 'timed out'}")

    except KeyboardInterrupt:
        print("\nTest run stopped by user.")
    finally:
        env.close()

    n = max(args.episodes, 1)
    total_actions = max(counts.sum(), 1)
    mix = "  ".join(f"{name} {100.0 * c / total_actions:.0f}%"
                    for name, c in zip(ACTION_NAMES, counts) if c)
    print("\n" + "=" * 70)
    print(f"  {'deterministic (argmax)' if deterministic else 'stochastic (sampled)'}"
          f" over {args.episodes} episode(s)")
    print(f"  completed  {completed}/{args.episodes}")
    print(f"  pickups    {totals['pickups'] / n:.1f}/ep      "
          f"deliveries {totals['deliveries'] / n:.1f}/ep      "
          f"ep_len {totals['steps'] / n:.1f}")
    print(f"  actions    {mix}")
    if totals["pickups"] == 0:
        print("\n  Zero pickups. If this was a deterministic run, try --stochastic "
              "before\n  concluding the policy is broken, then use "
              "scripts/probe_pickup.py to see\n  whether it is standing in reach "
              "of cartons without grabbing them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
