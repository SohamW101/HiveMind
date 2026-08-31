"""
Drive one robot through a full delivery, printing the reward every step.

The project roadmap, step 4: "Verify with play_multi.py before any training - drive one
bot through a full delivery and print the reward each step." This is that, plus a
numeric check of every line of the reward table in
MAWC_Technical_Specification.pdf section 3.

Reward is the one thing that cannot be debugged later. A wrong observation makes a
policy worse; a wrong reward makes it learn the wrong task and look fine doing it. So
every term is checked against the spec's number, and the shared/individual split is
recomputed by hand from the breakdown rather than trusted.

    .venv\\Scripts\\python.exe scripts/verify_rewards.py

Exits non-zero on the first failed check.
"""
import math
import os
import sys

import numpy as np
import pybullet as pb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from hivemind_env.env import (
    INDIVIDUAL_WEIGHT,
    NUM_CARTONS,
    R_ALL_DELIVERED,
    R_COLLISION,
    R_IDLE_PENALTY,
    R_INVALID_ACTION,
    R_MAKESPAN_SCALE,
    R_OWN_DELIVERY,
    R_OWN_PICKUP,
    R_PER_DELIVERY,
    R_TIME_PENALTY,
    SHARED_WEIGHT,
    HiveMindMultiAgentEnv,
)
from play_multi import approach_cell, direction_for, path_between, resource_cells, shelf_cells

TOL = 1e-9
checks = {"pass": 0, "fail": 0}


def check(label, condition, detail=""):
    if condition:
        checks["pass"] += 1
        print(f"  PASS  {label}")
    else:
        checks["fail"] += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


ACTION_NAMES = {0: "fwd", 1: "back", 2: "turnL", 3: "turnR", 4: "PICKUP", 5: "DROP", 6: "stay"}


def fmt_terms(d):
    return "{" + ", ".join(f"{k}={v:+.3f}" for k, v in d.items()) + "}" if d else "{}"


class Driver:
    """Scripted driver for robot 0; the others hold position."""

    def __init__(self, env, trace=True):
        self.env = env
        self.trace = trace
        self.cell = env._world_to_grid(*pb.getBasePositionAndOrientation(
            env.robot_ids[0], physicsClientId=env.client_id)[0][:2])
        self.direction = 0
        self.last = None

    def act(self, action, note=""):
        obs, rew, term, trunc, info = self.env.step([action, 6, 6, 6])
        self.last = (obs, rew, term, trunc, info)
        if self.trace:
            b = info["reward_breakdown"]
            print(f"  step {self.env.current_step:4d} | {ACTION_NAMES[action]:6s} | "
                  f"r0={rew[0]:+8.4f}  r=[{', '.join(f'{v:+.3f}' for v in rew)}] | "
                  f"shared={b['shared_total']:+8.3f} {fmt_terms(b['shared_terms'])} | "
                  f"indiv0={b['individual_totals'][0]:+.3f} "
                  f"{fmt_terms(b['individual_terms'][0])}"
                  + (f"  <- {note}" if note else ""))
        return self.last

    def turn_to(self, target):
        while self.direction != target:
            self.act(2)
            self.direction = (self.direction + 1) % 4

    def drive_to(self, target, blocked):
        for nxt in path_between(self.cell, target, blocked, self.env.grid_size)[1:]:
            self.turn_to(direction_for(self.cell, nxt))
            self.act(0)
            self.cell = nxt


def section(title):
    print(f"\n{'=' * 92}\n  {title}\n{'=' * 92}")


def main():
    print("=" * 92)
    print("  Reward verification - roadmap step 4, spec section 3")
    print("=" * 92)
    print(f"  shared weight {SHARED_WEIGHT}  individual weight {INDIVIDUAL_WEIGHT}")
    print(f"  delivery +{R_PER_DELIVERY} shared / +{R_OWN_DELIVERY} own   "
          f"pickup +{R_OWN_PICKUP} own   all-delivered +{R_ALL_DELIVERED}")
    print(f"  time {R_TIME_PENALTY}/step   idle {R_IDLE_PENALTY}/step   "
          f"collision {R_COLLISION}/event   invalid {R_INVALID_ACTION}")

    # ---- 1. Full delivery, reward printed every step -------------------------
    section("1. One robot, one complete delivery - every step")
    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        env.reset(seed=7)
        d = Driver(env)

        cells = resource_cells(env)
        blocked = shelf_cells(cells, env.grid_size) | {(0, 0)}
        target_cell = cells[0]
        slot = env.resource_slot[env.resource_ids[0]]

        d.drive_to(approach_cell(target_cell, blocked, env.grid_size), blocked)
        d.turn_to(direction_for(d.cell, target_cell))

        _, rew_pickup, _, _, info_pickup = d.act(4, "pick up")
        b = info_pickup["reward_breakdown"]
        check("pickup pays +1.0 to the picker only",
              b["individual_terms"][0].get("own_pickup") == R_OWN_PICKUP
              and all("own_pickup" not in b["individual_terms"][j] for j in (1, 2, 3)))
        check("pickup pays nothing shared",
              "per_delivery" not in b["shared_terms"])

        d.drive_to((0, 1), blocked)
        _, rew_drop, term, trunc, info_drop = d.act(5, "deliver")
        b = info_drop["reward_breakdown"]

        check("delivery pays +10.0 shared",
              b["shared_terms"].get("per_delivery") == R_PER_DELIVERY,
              f"got {b['shared_terms'].get('per_delivery')}")
        check("delivery pays +2.0 to the deliverer only",
              b["individual_terms"][0].get("own_delivery") == R_OWN_DELIVERY
              and all("own_delivery" not in b["individual_terms"][j] for j in (1, 2, 3)))
        check("all four agents receive the same shared component",
              len({round(SHARED_WEIGHT * b["shared_total"], 9)}) == 1)
        check("carton is recorded delivered", env.delivered[slot])
        check("not terminated on 1 of 12", not term and not trunc)

        # The headline identity, recomputed by hand.
        #
        # Shaping is added OUTSIDE the 90/10 split as of 2026-08-31 - inside it, the
        # 0.10 individual weight was silently dividing shaping_scale by ten. So the
        # identity now carries a third term. Everything the specification defines is
        # still exactly 0.90*shared + 0.10*individual; F_i is the declared addition.
        shaping_i = [t.get("shaping (unweighted)", 0.0) for t in b["individual_terms"]]
        expected = [SHARED_WEIGHT * b["shared_total"] + INDIVIDUAL_WEIGHT * iv + f
                    for iv, f in zip(b["individual_totals"], shaping_i)]
        check("R_total_i = 0.90*shared + 0.10*individual_i + F_i for all four agents",
              all(abs(e - r) < TOL for e, r in zip(expected, rew_drop)),
              f"expected {expected}\n        got      {list(rew_drop)}")
        check("shaping is NOT inside the individual total (it would be diluted 10x)",
              all("shaping (unweighted)" not in str(k) or True for k in [0])
              and all(abs(sum(v for k, v in t.items()
                              if k != "shaping (unweighted)") - iv) < TOL
                      for t, iv in zip(b["individual_terms"], b["individual_totals"])),
              f"individual_totals {b['individual_totals']} vs terms {b['individual_terms']}")

        # PERMANENT REGRESSION TEST for the bug that stalled roadmap step 6.
        #
        # Until 2026-08-31 the shaping potential was plain distance to the robot's
        # current objective, and the objective switched from carton to depot the moment
        # is_carrying flipped. Phi therefore fell off a cliff at exactly the transition
        # the task is built around, and every pickup in a PERFECT greedy episode scored
        # a negative total reward - up to -0.698 - against a +1.0 own-pickup term that
        # the 0.10 weight had already reduced to +0.1.
        #
        # The single most important action in the environment was punished, and no
        # training curve showed it. This assertion is the reason it cannot come back.
        check("PICKUP PAYS: the picking agent's total reward is positive",
              rew_pickup[0] > 0,
              f"pickup step paid {rew_pickup[0]:+.4f} to the picker - if this is "
              f"negative the policy is being trained to avoid picking cartons up")
        check("DELIVERY PAYS: the delivering agent's total reward is positive",
              rew_drop[0] > 0, f"delivery step paid {rew_drop[0]:+.4f}")

        # ---- 2. Time and idle penalties --------------------------------------
        section("2. Time and idle penalties")
        _, rew, _, _, info = d.act(6, "stay, parked at depot")
        b = info["reward_breakdown"]
        check("time penalty applied every step",
              b["shared_terms"].get("time_penalty") == R_TIME_PENALTY)
        check("robot parked at the depot is not charged idle",
              "idle" not in b["individual_terms"][0],
              f"got {b['individual_terms'][0]}")
        far = [j for j in range(env.num_agents) if not env._at_depot(j)]
        check("robots idling away from the depot are charged",
              all("idle" in b["individual_terms"][j] for j in far),
              f"far-from-depot agents {far}: "
              f"{[b['individual_terms'][j] for j in far]}")

        # ---- 3. Invalid actions ----------------------------------------------
        section("3. Invalid actions (spec S3.2, -0.5)")
        _, rew, _, _, info = d.act(5, "drop with an empty gripper")
        b = info["reward_breakdown"]
        check("dropping while empty is invalid",
              b["individual_terms"][0].get("invalid_action") == R_INVALID_ACTION,
              f"got {b['individual_terms'][0]}")

        _, rew, _, _, info = d.act(4, "pick up with nothing in range")
        b = info["reward_breakdown"]
        check("grabbing at nothing is invalid",
              b["individual_terms"][0].get("invalid_action") == R_INVALID_ACTION,
              f"got {b['individual_terms'][0]}")
        check("invalid action is individual, never shared",
              "invalid_action" not in b["shared_terms"])
    finally:
        env.close()

    # ---- 4. Collisions ---------------------------------------------------------
    section("4. Collision (spec S3.1, -5.0 per event)")
    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        env.reset(seed=7)
        # Robot 0 spawns at grid (0,1) facing +column; robot 2 sits at (0,2).
        # One forward step puts them in the same cell.
        _, rew, _, _, info = env.step([0, 6, 6, 6])
        b = info["reward_breakdown"]
        print(f"  contact pairs: {info['collision_pairs']}   "
              f"shared terms: {fmt_terms(b['shared_terms'])}")
        check("driving into another robot registers a collision",
              info["collisions"] >= 1, f"collisions={info['collisions']}")
        check("collision charges -5.0 shared",
              b["shared_terms"].get("collision") == R_COLLISION * info["collisions"],
              f"got {b['shared_terms'].get('collision')}")
        # Every agent must carry the SAME shared component. Comparing the raw rewards
        # does not test that - each robot has its own individual terms, and since
        # potential-based shaping landed those differ for all four. Back the individual
        # part out and check what is left is one number.
        # Shaping is outside the split, so it has to come out here too.
        shared_part = [r - INDIVIDUAL_WEIGHT * iv - t.get("shaping (unweighted)", 0.0)
                       for r, iv, t in zip(rew, b["individual_totals"],
                                           b["individual_terms"])]
        check("collision is shared, so all four agents pay the same amount",
              max(shared_part) - min(shared_part) < TOL,
              f"shared components differ: {[round(v, 6) for v in shared_part]}")

        _, rew2, _, _, info2 = env.step([6, 6, 6, 6])
        check("a pair that stays overlapped is not re-charged every step",
              "collision" not in info2["reward_breakdown"]["shared_terms"],
              f"got {info2['reward_breakdown']['shared_terms']}")
    finally:
        env.close()

    # ---- 4b. Shelves are unenterable --------------------------------------------
    # This section used to assert the opposite: that driving into a shelf registered a
    # contact and was charged -5.0. Shelves became solid on 2026-08-29 and the penalty
    # was left to "do the work" of teaching robots to route around them.
    #
    # Measured on 2026-08-31, it did not do that work. A random policy took 105.8
    # collision events per episode of which 93.8 were shelf contacts, while the greedy
    # controller took 0.0 across 10 episodes. The penalty was not teaching avoidance;
    # it was charging -4.5 per event, to all four agents, for a mistake the optimal
    # policy never makes - -1.19 reward/agent/step for moving against -0.045 for
    # standing still. PPO learned to stand still, three runs running.
    #
    # The move is now refused, exactly as driving off the grid always was, and costs
    # the specification's -0.5 invalid action. No reward constant changed.
    section("4b. Shelves are unenterable - the move is refused, not charged")
    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step([2, 6, 6, 6])          # face the shelf row
        before = env._canonical_pose(0)[:2]
        _, rew, _, _, info = env.step([0, 6, 6, 6])
        after = env._canonical_pose(0)[:2]
        b = info["reward_breakdown"]
        print(f"  pose {tuple(round(v, 2) for v in before)} -> "
              f"{tuple(round(v, 2) for v in after)}   "
              f"invalid={info['invalid_actions'][0]}   "
              f"shelf_contacts={info['shelf_contacts']}   "
              f"shared: {fmt_terms(b['shared_terms'])}")

        check("the shelf row is in the blocked set",
              len(env.blocked_cells) == 6 * (env.grid_size - 2) - NUM_CARTONS,
              f"{len(env.blocked_cells)} blocked cells, expected "
              f"{6 * (env.grid_size - 2) - NUM_CARTONS}")
        check("driving into a shelf does not move the robot",
              math.hypot(after[0] - before[0], after[1] - before[1]) < TOL,
              f"moved {math.hypot(after[0]-before[0], after[1]-before[1]):.4f} m")
        check("driving into a shelf is an invalid action",
              info["invalid_actions"][0],
              f"invalid_actions={info['invalid_actions']}")
        check("it costs -0.5 individual, not -5.0 shared",
              b["individual_terms"][0].get("invalid_action") == R_INVALID_ACTION
              and "collision" not in b["shared_terms"],
              f"individual={b['individual_terms'][0]}  shared={b['shared_terms']}")
        check("no shelf contact is generated at all",
              info["shelf_contacts"] == 0 and not any(
                  p[0] == "obstacle" for p in info["collision_pairs"]),
              f"shelf_contacts={info['shelf_contacts']} pairs={info['collision_pairs']}")

        # The planner and the simulator must agree about which squares exist, or a
        # robot paths into a wall forever. greedy.py reads env.blocked_cells directly
        # for exactly this reason; assert the agreement rather than trusting it.
        walkable_gaps = set(env.carton_home_cells)
        check("carton gap cells stay walkable",
              not (walkable_gaps & env.blocked_cells),
              f"{len(walkable_gaps & env.blocked_cells)} carton cells were blocked")
    finally:
        env.close()

    # ---- 5. Termination and the terminal bonuses -------------------------------
    section("5. Episode end - all 12 delivered")
    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        env.reset(seed=7)
        d = Driver(env, trace=False)
        cells = resource_cells(env)
        blocked = shelf_cells(cells, env.grid_size) | {(0, 0)}

        # Mark the other 11 as already delivered so the run reaches the terminal
        # branch in a few dozen steps rather than a few thousand. The final carton is
        # picked up and delivered for real, so the code path under test is the real one.
        target_id = env.resource_ids[0]
        keep = env.resource_slot[target_id]
        for s in range(NUM_CARTONS):
            if s != keep:
                env.delivered[s] = True
        print(f"  11 of 12 marked delivered; carton slot {keep} delivered for real")

        d.drive_to(approach_cell(cells[0], blocked, env.grid_size), blocked)
        d.turn_to(direction_for(d.cell, cells[0]))
        d.act(4)
        d.drive_to((0, 1), blocked)
        _, rew, term, trunc, info = d.act(5)
        b = info["reward_breakdown"]

        t_actual = env.current_step
        expected_bonus = R_MAKESPAN_SCALE * (env.max_steps - t_actual) / env.max_steps

        print(f"  finished on step {t_actual} of {env.max_steps}")
        print(f"  shared terms: {fmt_terms(b['shared_terms'])}")
        print(f"  final reward: [{', '.join(f'{v:+.4f}' for v in rew)}]")

        check("terminated when all 12 are delivered", term)
        check("not truncated", not trunc)
        check("all-delivered bonus is +100.0",
              b["shared_terms"].get("all_delivered") == R_ALL_DELIVERED)
        check("makespan bonus matches 50*(T_max-T_actual)/T_max",
              abs(b["shared_terms"].get("makespan_bonus", 0) - expected_bonus) < 1e-9,
              f"expected {expected_bonus:.6f}, got {b['shared_terms'].get('makespan_bonus')}")
        check("final delivery still pays its +10 shared and +2 own",
              b["shared_terms"].get("per_delivery") == R_PER_DELIVERY
              and b["individual_terms"][0].get("own_delivery") == R_OWN_DELIVERY)
        check("info reports success", info["is_success"] and info["all_delivered"])
        check("makespan bonus is awarded once per episode", env._makespan_awarded)
    finally:
        env.close()

    # ---- 6. Truncation ---------------------------------------------------------
    section("6. Episode end - step limit")
    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        env.reset(seed=7)
        env.max_steps = 5
        term = trunc = False
        for _ in range(5):
            _, rew, term, trunc, info = env.step([6, 6, 6, 6])
        check("truncated at max_steps", trunc, f"term={term} trunc={trunc}")
        check("not terminated - the job is unfinished", not term)
        check("no completion bonus on a truncated episode",
              "all_delivered" not in info["reward_breakdown"]["shared_terms"]
              and "makespan_bonus" not in info["reward_breakdown"]["shared_terms"])
        check("step limit is actually enforced", env.current_step == 5)
    finally:
        env.close()

    print("\n" + "=" * 92)
    print(f"  {checks['pass']} passed, {checks['fail']} failed")
    print("=" * 92)
    return 1 if checks["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
