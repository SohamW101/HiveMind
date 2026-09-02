"""
What is the reward actually paying for? Answered in seconds, not in a training run.

    scripts/diagnose_incentives.py --num-cartons 4

Three training runs were spent discovering reward bugs a thirty-second measurement would
have caught:

    5,013,504 steps, 12 cartons -> 0 completions, reward -103 (below the -94 floor)
      409,600 steps,  4 cartons -> 0 completions, reward -16.5 (stay scores -17.2)

Both failed for reasons that are arithmetic, not stochastic. A random policy took 105.8
collision events per episode - 93.8 against shelving, which greedy never touches once -
at -4.5 each: -1.19 reward per agent per step for moving against -0.045 for standing
still. No amount of exploration beats a 26x tax. Separately, the shaping term made every
pickup score NEGATIVE, punishing the single most important action in the task.

Neither is visible in a TensorBoard curve. Both are obvious in this table.

For `stay`, `random` and `greedy` on the same seeds it prints reward per agent per
episode and per step, collision events split robot-robot vs obstacle, pickups and
deliveries, the total reward of a pickup and a delivery step with shaping broken out,
and the per-cell shaping gain.

The gates at the bottom are what matter. A pickup that does not pay, or a random policy
scoring far below a do-nothing policy, means the reward points somewhere other than the
task - and training against it produces a confident, well-converged policy that stands
still. Passing them does NOT mean the policy will learn; it means the incentives are not
the reason if it does not.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from hivemind_env.env import (
    NUM_AGENTS,
    R_TIME_PENALTY,
    SHAPING_SCALE_DEFAULT,
    SHARED_WEIGHT,
    HiveMindMultiAgentEnv,
)
from hivemind_env.greedy import GreedyController

STAY_ACTION = 6


class Tally:
    """Per-policy accumulator. Everything here is per episode unless named otherwise."""

    def __init__(self, name):
        self.name = name
        self.reward = []          # mean over agents, summed over the episode
        self.length = []
        self.completed = 0
        self.cc_robot = 0
        self.cc_obstacle = 0
        self.pickups = 0
        self.deliveries = 0
        self.invalid = 0
        self.episodes = 0
        self.pickup_rewards = []   # total reward on the step an agent picked up
        self.pickup_shaping = []
        self.deliver_rewards = []
        self.deliver_shaping = []

    def row(self):
        n = max(self.episodes, 1)
        steps = max(sum(self.length), 1)
        return dict(
            name=self.name,
            reward=float(np.mean(self.reward)) if self.reward else float("nan"),
            per_step=(float(np.sum(self.reward)) / steps) if self.reward else float("nan"),
            length=float(np.mean(self.length)) if self.length else float("nan"),
            completed=self.completed,
            episodes=self.episodes,
            cc_robot=self.cc_robot / n,
            cc_obstacle=self.cc_obstacle / n,
            collision_cost=(self.cc_robot + self.cc_obstacle) / n * -5.0 * SHARED_WEIGHT,
            pickups=self.pickups / n,
            deliveries=self.deliveries / n,
            invalid=self.invalid / n,
        )


def rollout(policy, seeds, num_cartons, shaping, scale):
    """One tally over `seeds`. `policy` is 'stay', 'random' or 'greedy'."""
    t = Tally(policy)
    rng = np.random.default_rng(12345)

    for s in seeds:
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons,
                                    shaping=shaping, shaping_scale=scale)
        env.reset(seed=s)
        ctrl = GreedyController(env) if policy == "greedy" else None
        ep = np.zeros(NUM_AGENTS)
        terminated = False

        for _ in range(env.max_steps):
            if policy == "stay":
                actions = np.full(NUM_AGENTS, STAY_ACTION)
            elif policy == "random":
                actions = rng.integers(0, 7, size=NUM_AGENTS)
            else:
                actions = ctrl.act()

            _, rewards, terminated, truncated, info = env.step(actions)
            if ctrl is not None:
                ctrl.sync_after_step()

            ep += np.asarray(rewards)
            for pair in info["collision_pairs"]:
                if pair[0] == "robot":
                    t.cc_robot += 1
                else:
                    t.cc_obstacle += 1
            t.pickups += sum(info["pickups"])
            t.deliveries += sum(info["deliveries"])
            t.invalid += sum(info["invalid_actions"])

            # The two transitions the whole task turns on. Record what they actually pay.
            terms = info["reward_breakdown"]["individual_terms"]
            for i in range(NUM_AGENTS):
                sh = terms[i].get("shaping (unweighted)", 0.0)
                if info["pickups"][i]:
                    t.pickup_rewards.append(rewards[i])
                    t.pickup_shaping.append(sh)
                if info["deliveries"][i]:
                    t.deliver_rewards.append(rewards[i])
                    t.deliver_shaping.append(sh)

            if terminated or truncated:
                break

        t.episodes += 1
        t.length.append(env.current_step)
        t.reward.append(float(ep.mean()))
        if terminated:
            t.completed += 1
        env.close()

    return t


def approach_gain(num_cartons, shaping, scale):
    """
    What one cell of progress toward the objective is worth, measured not derived.

    Drives a single robot one cell closer to its target and reads the shaping term off
    the breakdown. This is the number that has to beat the time penalty for approaching
    to be worth doing at all.
    """
    env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons,
                               shaping=shaping, shaping_scale=scale)
    env.reset(seed=1000)
    ctrl = GreedyController(env)
    gains = []
    for _ in range(min(env.max_steps, 40)):
        actions = ctrl.act()
        _, _, term, trunc, info = env.step(actions)
        ctrl.sync_after_step()
        terms = info["reward_breakdown"]["individual_terms"]
        # Skip the whole step if ANY agent picked up or delivered, not just this one.
        # Phi's n_undelivered term is global, so a teammate's delivery raises every
        # agent's potential by a full carton - and an earlier version of this filter,
        # which only excluded the acting agent, reported +0.931 per cell at 12 cartons
        # against a true value near +0.15. A per-cell figure inflated 6x is worse than
        # no figure at all, because it is the number the shaping scale is tuned against.
        if any(info["pickups"]) or any(info["deliveries"]):
            if term or trunc:
                break
            continue
        for i in range(NUM_AGENTS):
            if int(actions[i]) == 0:
                sh = terms[i].get("shaping (unweighted)", 0.0)
                if sh > 0:
                    gains.append(sh)
        if term or trunc:
            break
    env.close()
    return float(np.mean(gains)) if gains else float("nan")


def collision_per_move(num_cartons, seeds):
    """
    P(a robot-robot collision event | one forward or backward action), under random play.

    Only movement can collide - turning, staying, PICKUP and DROP never can - so this is
    the risk premium attached to the one action class that makes progress. Measured under
    a random policy, which is deliberately pessimistic: a directed policy collides less
    (greedy manages 2.8 per episode), so the margin widens as the policy improves. That
    is the bootstrapping property we want.
    """
    rng = np.random.default_rng(7)
    moves = events = 0
    for s in seeds:
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons)
        env.reset(seed=s + 3000)
        for _ in range(env.max_steps):
            actions = rng.integers(0, 7, size=NUM_AGENTS)
            moves += int(sum(1 for a in actions if int(a) in (0, 1)))
            _, _, term, trunc, info = env.step(actions)
            events += sum(1 for p in info["collision_pairs"] if p[0] == "robot")
            if term or trunc:
                break
        env.close()
    return events / max(moves, 1)


def main():
    ap = argparse.ArgumentParser(description="Print the reward budget for each policy")
    ap.add_argument("--num-cartons", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--shaping-scale", type=float, default=SHAPING_SCALE_DEFAULT,
                    help="Sweep this to retune the shaping without editing env.py. "
                         "Raising it must not break the pickup gate below.")
    ap.add_argument("--no-shaping", action="store_true",
                    help="Measure the specification's reward exactly, shaping removed.")
    args = ap.parse_args()

    shaping = not args.no_shaping
    seeds = list(range(args.seed, args.seed + args.episodes))

    print("=" * 92)
    print(f"  Incentive diagnostic - {args.num_cartons} cartons, {args.episodes} seeds "
          f"({seeds[0]}-{seeds[-1]}), shaping "
          f"{('scale ' + str(args.shaping_scale)) if shaping else 'OFF'}")
    print("=" * 92)

    tallies = [rollout(p, seeds, args.num_cartons, shaping, args.shaping_scale)
               for p in ("stay", "random", "greedy")]
    rows = [t.row() for t in tallies]

    print(f"\n{'policy':>8} {'reward':>9} {'/step':>8} {'ep_len':>7} {'done':>7} "
          f"{'coll rr':>8} {'coll obs':>9} {'coll cost':>10} {'pickup':>7} {'deliv':>7} {'invalid':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:>8} {r['reward']:>9.1f} {r['per_step']:>8.3f} "
              f"{r['length']:>7.1f} {r['completed']:>3}/{r['episodes']:<3} "
              f"{r['cc_robot']:>8.1f} {r['cc_obstacle']:>9.1f} {r['collision_cost']:>10.1f} "
              f"{r['pickups']:>7.1f} {r['deliveries']:>7.1f} {r['invalid']:>8.1f}")

    stay, rnd, greedy = rows

    print("\n  the two transitions the task turns on (greedy, per event)")
    print("  " + "-" * 62)
    g = tallies[2]
    for label, rew, sh in (("PICKUP", g.pickup_rewards, g.pickup_shaping),
                           ("DELIVER", g.deliver_rewards, g.deliver_shaping)):
        if rew:
            print(f"  {label:>8}: total reward {np.mean(rew):+8.3f}   "
                  f"(shaping {np.mean(sh):+7.3f})   over {len(rew)} events   "
                  f"min {min(rew):+.3f}")
        else:
            print(f"  {label:>8}: never happened")

    time_cost = R_TIME_PENALTY * SHARED_WEIGHT
    gain = approach_gain(args.num_cartons, shaping, args.shaping_scale)

    # THE NUMBER THAT DECIDES WHETHER THE POLICY WILL MOVE AT ALL.
    #
    # An earlier version of this section compared the shaping gain to the time penalty
    # and called 3x healthy. That was the wrong competitor. The time penalty is charged
    # whether or not the robot moves, so it cannot make moving unattractive - only the
    # collision risk can, and it is charged ONLY when the robot moves.
    #
    # Measured on the canary that failed: at scale 6.0 a move was worth -0.227 and the
    # policy went to `stay` 100% deterministic, with fwd 2% / back 1% under sampling. It
    # still turned (41%) and still pressed PICKUP (21%) - because turning and grabbing
    # cannot collide. It had learned precisely the right lesson from the wrong reward.
    p_coll = collision_per_move(args.num_cartons, seeds[:4])
    coll_cost = -5.0 * SHARED_WEIGHT
    ev_move = gain + p_coll * abs(coll_cost) * -1 + time_cost

    print("\n  per-step economics")
    print("  " + "-" * 62)
    print(f"  one cell of approach     : {gain:+8.3f}")
    print(f"  time penalty per step    : {time_cost:+8.3f}")
    print(f"  P(collision | one move)  : {p_coll:8.4f}   (random policy)")
    print(f"  expected collision cost  : {p_coll * coll_cost:+8.3f}")
    print(f"  EXPECTED VALUE OF MOVING : {ev_move:+8.3f}   <- must be positive")
    if gain > 0:
        print(f"  one collision event      : {coll_cost:+8.3f}"
              f"  = {abs(coll_cost / gain):.0f} cells of progress")

    # ---- gates -----------------------------------------------------------------
    print("\n  gates")
    print("  " + "-" * 62)
    fails = []

    worst_pickup = min(g.pickup_rewards) if g.pickup_rewards else float("nan")
    ok = bool(g.pickup_rewards) and worst_pickup > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] a pickup pays  "
          f"(worst {worst_pickup:+.3f}, needs > 0)")
    if not ok:
        fails.append("pickup does not pay - the policy is punished for the key action")

    ok = bool(g.deliver_rewards) and min(g.deliver_rewards) > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] a delivery pays "
          f"(worst {min(g.deliver_rewards) if g.deliver_rewards else float('nan'):+.3f}, needs > 0)")
    if not ok:
        fails.append("delivery does not pay")

    # Random must not be catastrophically worse than doing nothing, or the fastest
    # descent for PPO is to stop moving - which is exactly what it did, twice.
    margin = rnd["reward"] - stay["reward"]
    ok = margin > -5.0 * abs(stay["reward"]) - 50.0
    print(f"  [{'PASS' if ok else 'FAIL'}] exploring is survivable "
          f"(random {rnd['reward']:+.1f} vs stay {stay['reward']:+.1f}, "
          f"gap {margin:+.1f})")
    if not ok:
        fails.append("random is far below stay - standing still is the easy optimum")

    ok = greedy["reward"] > stay["reward"] and greedy["completed"] == greedy["episodes"]
    print(f"  [{'PASS' if ok else 'FAIL'}] the good policy wins "
          f"(greedy {greedy['reward']:+.1f}, {greedy['completed']}/{greedy['episodes']} complete)")
    if not ok:
        fails.append("greedy does not dominate - something is wrong beyond the reward")

    ok = greedy["cc_obstacle"] < 0.5
    print(f"  [{'PASS' if ok else 'FAIL'}] the good policy does not hit shelves "
          f"({greedy['cc_obstacle']:.1f} per episode)")
    if not ok:
        fails.append("greedy hits shelving - the blocked-cell set disagrees with physics")

    # The gate the first canary would have failed before it was ever launched.
    ok = ev_move > 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}] MOVING IS WORTH IT "
          f"(expected value {ev_move:+.3f}, needs > +0.05)")
    if not ok:
        fails.append(
            f"a movement action is worth {ev_move:+.3f} - the policy will learn to "
            f"turn, grab and stand still, because those cannot collide. Raise "
            f"--shaping-scale until this clears.")

    print()
    if fails:
        print("  NOT READY TO TRAIN:")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("  Incentives are sane. This does not promise the policy will learn - it means")
    print("  the reward is not the reason if it does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
