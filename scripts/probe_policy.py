"""
What is a checkpoint actually doing? Action mix, pickups, deliveries, collisions.

WHY THIS AND NOT THE REWARD CURVE

`ep_rew_mean` cannot distinguish "learned to stand still" from "working but not
finishing", and both look like a smoothly rising curve. Three runs were misread that way.
The distinction is visible in one place: what the policy actually does.

    stay-like     : stay >> everything, 0 pickups, 0 deliveries
    thrashing     : fwd ~= back, 0 net progress          <- the BC clone's failure
    working       : PICKUP and DROP non-zero, deliveries > 0

Deterministic and stochastic are both reported because they can disagree sharply. The
behaviour-cloned policy scored 0.0 deliveries under argmax and 2.1 under sampling: greedy
uses a single `backward` for a 180-degree turn, so the demonstrator is bimodal from one
observation and the argmax flips between the modes. A policy is not broken merely because
its argmax is.

    .venv\\Scripts\\python.exe scripts/probe_policy.py models/canary_final.zip --num-cartons 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from stable_baselines3 import PPO  # noqa: E402

from hivemind_env.env import NUM_AGENTS, HiveMindMultiAgentEnv  # noqa: E402
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS, get_device  # noqa: E402

ACTION_NAMES = ["fwd", "back", "turnL", "turnR", "PICKUP", "DROP", "stay"]


def probe(model, episodes, num_cartons, deterministic, seed0=1000):
    counts = np.zeros(7, dtype=int)
    pickups = deliveries = collisions = completed = 0
    lengths, rewards, delivered = [], [], []

    for ep in range(episodes):
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons)
        obs, _ = env.reset(seed=seed0 + ep)
        total = np.zeros(NUM_AGENTS)
        terminated = False

        for _ in range(env.max_steps):
            actions, _ = model.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=deterministic)
            actions = np.asarray(actions).reshape(-1)[:NUM_AGENTS]
            for a in actions:
                counts[int(a)] += 1
            obs, r, terminated, truncated, info = env.step(actions)
            total += np.asarray(r)
            pickups += sum(info["pickups"])
            deliveries += sum(info["deliveries"])
            collisions += info["collisions"]
            if terminated or truncated:
                break

        if terminated:
            completed += 1
        lengths.append(env.current_step)
        rewards.append(float(total.mean()))
        delivered.append(sum(info["delivered_flags_total"][:env.active_cartons])
                         if isinstance(info.get("delivered_flags_total"), (list, tuple))
                         else info["delivered"])
        env.close()

    n = max(episodes, 1)
    return dict(
        counts=counts, completed=completed, episodes=episodes,
        length=float(np.mean(lengths)), reward=float(np.mean(rewards)),
        pickups=pickups / n, deliveries=deliveries / n, collisions=collisions / n,
        delivered=float(np.mean(delivered)),
    )


def report(label, r, num_cartons):
    total = max(r["counts"].sum(), 1)
    mix = "  ".join(f"{ACTION_NAMES[i]} {r['counts'][i] / total:.0%}" for i in range(7))
    print(f"  {label:>13}: completed {r['completed']}/{r['episodes']}   "
          f"ep_len {r['length']:6.1f}   reward {r['reward']:+8.1f}   "
          f"delivered {r['delivered']:.1f}/{num_cartons}")
    print(f"  {'':>13}  pickups {r['pickups']:.1f}/ep   "
          f"deliveries {r['deliveries']:.1f}/ep   collisions {r['collisions']:.1f}/ep")
    print(f"  {'':>13}  {mix}")

    # The diagnosis, stated rather than left to the reader.
    c = r["counts"] / total
    if c[6] > 0.5:
        verdict = "STANDING STILL - the reward is being maximised by doing nothing"
    elif abs(c[0] - c[1]) < 0.1 and c[0] + c[1] > 0.6:
        verdict = "THRASHING - forward and backward cancel; no net progress"
    elif r["deliveries"] == 0:
        verdict = "MOVING BUT NOT WORKING - never completes a delivery"
    elif r["completed"] == 0:
        verdict = "WORKING BUT NOT FINISHING - deliveries happen, episodes do not close"
    else:
        verdict = "COMPLETING EPISODES"
    print(f"  {'':>13}  -> {verdict}\n")


def main():
    ap = argparse.ArgumentParser(description="What is this checkpoint actually doing?")
    ap.add_argument("model")
    ap.add_argument("--num-cartons", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    model = PPO.load(args.model, device=get_device(),
                     custom_objects=INFERENCE_CUSTOM_OBJECTS)

    print("=" * 78)
    print(f"  {os.path.basename(args.model)} at {args.num_cartons} cartons, "
          f"{args.episodes} episodes (seeds 1000-{1000 + args.episodes - 1})")
    print("=" * 78)
    for label, det in (("deterministic", True), ("stochastic", False)):
        report(label, probe(model, args.episodes, args.num_cartons, det),
               args.num_cartons)

    ref = {4: 23, 8: 58, 12: 97}.get(args.num_cartons)
    if ref:
        print(f"  greedy reference at {args.num_cartons} cartons: "
              f"makespan {ref}, 30/30 complete")


if __name__ == "__main__":
    main()
