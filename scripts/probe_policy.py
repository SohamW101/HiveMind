"""
What is a checkpoint actually doing? Action mix, pickups, deliveries, collisions.

    scripts/probe_policy.py models/run_final.zip --num-cartons 4

`ep_rew_mean` cannot distinguish "learned to stand still" from "working but not
finishing", and both look like a smoothly rising curve - three runs were misread that
way. The distinction is only visible in what the policy does:

    stay-like  stay >> everything, 0 pickups, 0 deliveries
    thrashing  fwd ~= back, no net progress          <- the BC clone's failure
    working    PICKUP and DROP non-zero, deliveries > 0

Deterministic and stochastic are both reported because they disagree sharply: greedy
uses a single `backward` for a 180-degree turn, so a cloned demonstrator is bimodal from
one observation and its argmax flips between the modes. A policy is not broken merely
because its argmax is.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from stable_baselines3 import PPO

from hivemind_env.env import (
    ACTION_NAMES,
    NUM_AGENTS,
    HiveMindMultiAgentEnv,
    joint_from_slot_actions,
    policy_uses_comms,
)
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS, get_device



def probe(model, episodes, num_cartons, deterministic, seed0=1000, message_mode="learned"):
    # A comms checkpoint needs a comms env or its own actions are rejected. Read off
    # the policy rather than asking for a flag.
    comms = policy_uses_comms(model)
    counts = np.zeros(7, dtype=int)
    pickups = deliveries = collisions = completed = 0
    lengths, rewards, delivered = [], [], []

    for ep in range(episodes):
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons,
                                    comms=comms, message_mode=message_mode,
                                    msg_dropout=0.0)
        obs, _ = env.reset(seed=seed0 + ep)
        total = np.zeros(NUM_AGENTS)
        terminated = False

        for _ in range(env.max_steps):
            raw, _ = model.predict(np.asarray(obs, dtype=np.float32),
                                   deterministic=deterministic)
            actions = joint_from_slot_actions(raw, NUM_AGENTS)
            moves = actions[:, 0] if actions.ndim == 2 else actions
            for a in moves:
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
