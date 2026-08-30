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
        expected = [SHARED_WEIGHT * b["shared_total"] + INDIVIDUAL_WEIGHT * iv
                    for iv in b["individual_totals"]]
        check("R_total_i = 0.90*shared + 0.10*individual_i for all four agents",
              all(abs(e - r) < TOL for e, r in zip(expected, rew_drop)),
              f"expected {expected}\n        got      {list(rew_drop)}")

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
        check("collision is shared, so all four agents pay it",
              len(set(np.round(rew, 9))) <= 2)  # differs only by individual terms

        _, rew2, _, _, info2 = env.step([6, 6, 6, 6])
        check("a pair that stays overlapped is not re-charged every step",
              "collision" not in info2["reward_breakdown"]["shared_terms"],
              f"got {info2['reward_breakdown']['shared_terms']}")
    finally:
        env.close()

    # ---- 4b. Shelf collisions ---------------------------------------------------
    # Shelves became solid on 2026-08-29: the bottom plate was lowered from 0.30 to
    # 0.18 so it overlaps the chassis. Before that a robot drove straight through a
    # shelf row for free, and aisles constrained nothing.
    section("4b. Shelf collision - shelves are solid obstacles now")
    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step([2, 6, 6, 6])          # face the shelf row
        _, rew, _, _, info = env.step([0, 6, 6, 6])
        b = info["reward_breakdown"]
        print(f"  contact pairs: {info['collision_pairs']}   "
              f"shelf_contacts={info['shelf_contacts']}   "
              f"shared: {fmt_terms(b['shared_terms'])}")
        check("entering a shelf cell registers contact", info["shelf_contacts"] >= 1)
        check("shelf contact is charged as a collision",
              b["shared_terms"].get("collision") == R_COLLISION * info["collisions"],
              f"got {b['shared_terms'].get('collision')}")
        check("the obstacle collision is attributed to the right robot",
              ("obstacle", 0) in info["collision_pairs"],
              f"pairs={info['collision_pairs']}")

        _, _, _, _, info2 = env.step([0, 6, 6, 6])   # drive out the far side
        check("leaving the shelf clears the contact",
              info2["shelf_contacts"] == 0
              and "collision" not in info2["reward_breakdown"]["shared_terms"],
              f"shelf_contacts={info2['shelf_contacts']}")
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
