"""
When a carton is in reach, does the policy actually press PICKUP?

    scripts/probe_pickup.py models/run_final.zip --num-cartons 12
    scripts/probe_pickup.py --baseline greedy --num-cartons 12

probe_policy.py answers "what does this checkpoint do?" across a whole episode, which
separates standing still from thrashing but cannot see the failure this was written for:
the robots drive to the carton and then stop. A policy that spends 95% of its steps
driving and 5% standing beside a carton looks healthy in an episode-wide histogram
whatever it does in the 5% that decide the task.

So this conditions on the one state that matters - not carrying, and some available
carton inside the env's own pickup radius, which is exactly what step() checks.

THE HEADLINE IS OPPORTUNITY-STEPS PER PICKUP, NOT CONVERSION. Raw conversion is diluted
by robots standing in range of a carton somebody else is claiming: greedy scores only
55.6% at 1 carton because three of the four are bystanders correctly standing off. Run
`--baseline greedy` at your carton count before reading anything into a policy's number.

WHAT THE ANSWER MEANS

  no opportunities         the policy never reaches a carton - probe_policy.py instead.
  argmax ~0, sampled >0    ARGMAX COLLAPSE. The weights know something, the greedy read
                           of them does not. mini_final.zip is DROP 76% under argmax and
                           completes 0/10, while the same weights sampled complete 4/10.
  both ~0                  the policy cannot tell it is in range. Note the observation
                           asymmetry: the depot arrives as a RELATIVE offset (54:56) but
                           carton positions are ABSOLUTE (30:54), and approaching needs
                           neither - the shaping term computes the distance and pays it
                           out - so a policy can learn the approach and never the
                           trigger. Fixing that is a V4 observation, not a reward change.
  both high, no delivery   the pickup fires and the freeze is later - look at what
                           happens while carrying.

The reward is not on the list of suspects, measured rather than assumed: beside a carton
PICKUP pays +9.8 to +14.4 against -0.047 for standing still.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pybullet as pb

from hivemind_env.env import (
    ACTION_NAMES,
    NUM_AGENTS,
    HiveMindMultiAgentEnv,
    joint_from_slot_actions,
    policy_uses_comms,
)
from hivemind_env.greedy import GreedyController
from hivemind_env.training import get_device, is_maskable, load_policy

PICKUP = 4

# The environment grants a pickup when the nearest carton is within 1.5 cells, measured
# body to body from the snapped pose. Read from the env at runtime rather than restated -
# a constant copied here is a constant that drifts.
REACH_CELLS = 1.5


def pickup_opportunity(env, agent_idx):
    """
    Is this robot-step one where `step()` would grant a pickup if PICKUP were pressed?

    Mirrors the env's own test exactly: not already carrying, and the nearest carton
    still on the floor within REACH_CELLS. Returns (is_opportunity, distance_or_None).
    """
    if env.is_carrying[agent_idx]:
        return False, None
    x, y, _ = env._canonical_pose(agent_idx)
    best = None
    for rid in env.resource_ids:
        p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=env.client_id)
        d = math.hypot(p[0] - x, p[1] - y)
        if best is None or d < best:
            best = d
    if best is None:
        return False, None
    return best <= env.cell_size * REACH_CELLS, best


class Stats:
    def __init__(self):
        self.opportunities = 0
        self.converted = 0
        self.in_range_actions = np.zeros(7, dtype=int)
        self.out_range_actions = np.zeros(7, dtype=int)
        self.pickup_out_of_range = 0     # PICKUP pressed with nothing in reach
        self.distances = []
        self.episodes_with_opportunity = 0
        self.episodes = 0
        self.completed = 0
        self.pickups = 0
        self.deliveries = 0
        self.lengths = []

    @property
    def conversion(self):
        return self.converted / self.opportunities if self.opportunities else float("nan")

    @property
    def steps_per_pickup(self):
        """
        Robot-steps spent standing in reach per pickup produced. The headline.

        Conversion alone is diluted by bystanders - a robot in range of a carton another
        robot is claiming should NOT press PICKUP. This ratio is not: a policy that picks
        up promptly keeps it near 1, and a policy that stands beside cartons without ever
        grabbing one sends it to infinity.
        """
        if self.pickups == 0:
            return float("inf")
        return self.opportunities / self.pickups


def run(actor, mode, episodes, num_cartons, deterministic, seed0=1000):
    """
    `mode` is "policy" or "greedy". Returns a Stats.

    The env is rebuilt per episode rather than reset in place, matching probe_policy.py
    - the greedy controller captures the shelf map at construction, so a fresh one is
    needed anyway and this keeps the two paths identical.
    """
    st = Stats()
    # A comms checkpoint emits a token alongside every movement and needs an env that
    # accepts one. Read it off the policy; the greedy controller never speaks.
    comms = mode == "policy" and policy_uses_comms(actor)
    masked = mode == "policy" and is_maskable(actor)

    for ep in range(episodes):
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons,
                                    comms=comms, msg_dropout=0.0)
        obs, _ = env.reset(seed=seed0 + ep)
        controller = GreedyController(env) if mode == "greedy" else None
        st.episodes += 1
        saw_opportunity = False
        steps = 0

        for _ in range(env.max_steps):
            # Sample the opportunity state BEFORE stepping - it is a property of the
            # state the action was chosen in, not of the one it produced.
            flags = [pickup_opportunity(env, i) for i in range(NUM_AGENTS)]

            if mode == "greedy":
                actions = np.asarray(controller.act(), dtype=int)
            else:
                kw = {"action_masks": env.action_masks()} if masked else {}
                raw, _ = actor.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=deterministic, **kw)
                actions = joint_from_slot_actions(raw, NUM_AGENTS)

            # Movement only. The conversion this script measures is about PICKUP, and
            # reading column 1 of a comms action would be counting tokens as actions.
            moves = actions[:, 0] if actions.ndim == 2 else actions

            for i, (in_range, dist) in enumerate(flags):
                a = int(moves[i])
                if in_range:
                    st.opportunities += 1
                    saw_opportunity = True
                    st.in_range_actions[a] += 1
                    st.distances.append(dist)
                    if a == PICKUP:
                        st.converted += 1
                else:
                    st.out_range_actions[a] += 1
                    if a == PICKUP and not env.is_carrying[i]:
                        st.pickup_out_of_range += 1

            obs, _, terminated, truncated, info = env.step(actions)
            if mode == "greedy":
                controller.sync_after_step()
            st.pickups += sum(1 for p in info["pickups"] if p)
            st.deliveries += sum(1 for d in info["deliveries"] if d)
            steps += 1
            if terminated or truncated:
                break

        if terminated:
            st.completed += 1
        if saw_opportunity:
            st.episodes_with_opportunity += 1
        st.lengths.append(steps)
        env.close()

    return st


def report(label, st):
    print(f"  {label}")
    if st.opportunities == 0:
        print(f"     opportunities  0  <- never got a carton within reach in "
              f"{st.episodes} episodes")
        print(f"     ep_len {np.mean(st.lengths):.1f}   completed {st.completed}/{st.episodes}")
        print()
        return

    pct = 100.0 * st.conversion
    spp = st.steps_per_pickup
    spp_s = "never picks up" if spp == float("inf") else f"{spp:.1f}"
    print(f"     opportunities {st.opportunities:>6}  in {st.episodes_with_opportunity}"
          f"/{st.episodes} episodes   mean distance {np.mean(st.distances):.2f} m")
    print(f"     PICKUP taken  {st.converted:>6}  -> conversion {pct:5.1f}%")
    print(f"     in-reach steps per pickup: {spp_s}"
          f"   <- the headline; compare with --baseline greedy")
    total = st.in_range_actions.sum()
    mix = "  ".join(f"{n} {100.0 * c / total:.0f}%"
                    for n, c in zip(ACTION_NAMES, st.in_range_actions) if c)
    print(f"     what it does instead, in range: {mix}")
    om = st.out_range_actions.sum()
    omix = "  ".join(f"{n} {100.0 * c / om:.0f}%"
                     for n, c in zip(ACTION_NAMES, st.out_range_actions) if c)
    print(f"     for contrast, out of range    : {omix}")
    print(f"     PICKUP pressed with nothing in reach: {st.pickup_out_of_range}"
          f"  (costs -0.5 individual each)")
    print(f"     pickups {st.pickups / st.episodes:.1f}/ep   "
          f"deliveries {st.deliveries / st.episodes:.1f}/ep   "
          f"ep_len {np.mean(st.lengths):.1f}   completed {st.completed}/{st.episodes}")
    print()


def verdict(det, sto):
    """One stated conclusion, so the table is not left to interpretation."""
    print("  verdict")
    print("  " + "-" * 70)
    if det.opportunities == 0 and sto.opportunities == 0:
        print("  NEVER REACHES A CARTON - nothing here is about the pickup action.")
        print("  Run probe_policy.py; the problem is navigation or standing still.")
        return
    # Judged on whether pickups happen at all, not on conversion - see Stats.
    d = det.pickups / max(det.episodes, 1)
    s = sto.pickups / max(sto.episodes, 1)
    print(f"  pickups per episode: argmax {d:.2f}   sampled {s:.2f}")
    if d < 0.05 and s >= 0.20:
        print("  ARGMAX COLLAPSE - sampled play presses PICKUP, argmax does not.")
        print("  The weights are not the problem; the deterministic read of them is.")
        print("  Check that the test harness is not passing deterministic=True.")
    elif d < 0.05 and s < 0.05:
        print("  THE TRIGGER WAS NEVER LEARNED - PICKUP is not pressed in either mode,")
        print("  though the robot is standing in range. The reward pays ~+10 for it, so")
        print("  this is a perception problem, not an incentive one: carton positions")
        print("  are ABSOLUTE in the observation while the depot is relative, so 'am I")
        print("  in range' is a subtraction the network has to learn for 12 cartons at")
        print("  every point in the arena. See the module docstring.")
    elif d >= 0.5 or s >= 0.5:
        print("  PICKUP FIRES - the freeze, if any, is not at the pickup.")
        print("  Look after the pickup instead, or at navigation to the depot.")
    else:
        print("  PARTIAL - PICKUP is pressed sometimes but far below greedy's 100%.")
        print("  Treat this as an under-trained trigger rather than a broken one.")


def main():
    p = argparse.ArgumentParser(
        description="Does the policy press PICKUP when a carton is in reach?")
    p.add_argument("model", nargs="?", default=None, help="path to a saved checkpoint")
    p.add_argument("--baseline", choices=["greedy"], default=None,
                   help="score the scripted controller instead of a model - the 100% "
                        "reference this number is read against")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--num-cartons", type=int, default=None,
                   help="cartons in play (env default 12). Use the count the policy "
                        "was trained at.")
    p.add_argument("--seed0", type=int, default=1000)
    args = p.parse_args()

    if args.baseline is None and args.model is None:
        p.error("give a checkpoint path, or --baseline greedy")

    cartons = args.num_cartons
    label = f"{cartons if cartons else 12} cartons, {args.episodes} episodes " \
            f"(seeds {args.seed0}-{args.seed0 + args.episodes - 1})"

    print("=" * 78)
    if args.baseline:
        print(f"  greedy baseline at {label}")
        print("=" * 78)
        st = run(None, "greedy", args.episodes, cartons, deterministic=True,
                 seed0=args.seed0)
        report("greedy", st)
        return

    device = get_device()
    model, _ = load_policy(args.model, device=device)

    print(f"  {os.path.basename(args.model)} at {label}")
    print("=" * 78)
    det = run(model, "policy", args.episodes, cartons, deterministic=True,
              seed0=args.seed0)
    report("deterministic (argmax)", det)
    sto = run(model, "policy", args.episodes, cartons, deterministic=False,
              seed0=args.seed0)
    report("stochastic (sampled)", sto)
    verdict(det, sto)


if __name__ == "__main__":
    main()
