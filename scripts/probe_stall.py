"""
Why does an episode never finish? Is the remaining carton reachable at all?

    scripts/probe_stall.py models/nocomm3_final.zip --num-cartons 4

WHY THIS EXISTS

nocomm3 delivered ~2.4 cartons per episode and completed 0 of 20, and quadrupling the
episode cap from 150 to 600 steps changed nothing. More time not helping rules out
slowness: whatever stops the episode is a state the policy cannot escape, not a job it
has not finished yet.

There is a documented mechanism that would produce exactly that. Cartons are solid
bodies, and a robot that drives into one PUSHES it - one carton was measured travelling
1.93 m out of its starting cell. A carton shoved into the interior of a shelf row can end
up with no enterable cell within the gripper's 1.5-cell reach, and then no robot can ever
pick it up. The episode becomes unwinnable, and every remaining step is wasted.

This script runs episodes to their end and, for each carton still undelivered, reports:

    moved        how far it is from where the generator put it
    reachable    is there an enterable cell within reach of it, from which a robot
                 could legally pick it up
    connected    can a robot actually GET to such a cell from the depot

The distinction between the last two matters: a carton can sit beside a legal standing
cell that is itself walled off from the rest of the arena.

    UNREACHABLE CARTONS  -> the world was broken by the robots. An environment fix.
    ALL REACHABLE        -> the world is fine and the policy simply stops trying.
                            A policy problem, and the next question is what it does
                            with those wasted steps (see probe_policy.py).

Greedy is the control: run --baseline greedy and confirm it strands nothing.
"""
from __future__ import annotations

import argparse
import math
from collections import deque

import numpy as np
import pybullet as pb

from hivemind_env.env import (
    NUM_AGENTS,
    HiveMindMultiAgentEnv,
    joint_from_slot_actions,
    policy_uses_comms,
)
from hivemind_env.greedy import GreedyController
from hivemind_env.training import get_device, is_maskable, load_policy

REACH_CELLS = 1.5


def reachable_cells(env, world_xy):
    """Enterable cells from which a robot could legally pick this carton up."""
    out = []
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r, c) in env.blocked_cells:
                continue
            x, y = env._grid_to_world(r, c)
            if math.hypot(x - world_xy[0], y - world_xy[1]) <= env.cell_size * REACH_CELLS:
                out.append((r, c))
    return out


def connected_from(env, start):
    """Every cell a robot can walk to from `start`."""
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in seen or nb in env.blocked_cells:
                continue
            if not (0 <= nb[0] < env.grid_size and 0 <= nb[1] < env.grid_size):
                continue
            seen.add(nb)
            q.append(nb)
    return seen


def run(actor, mode, episodes, num_cartons, deterministic, seed0, max_steps):
    comms = mode == "policy" and policy_uses_comms(actor)
    masked = mode == "policy" and is_maskable(actor)

    totals = {"episodes": 0, "completed": 0, "stranded_eps": 0,
              "undelivered": 0, "unreachable": 0, "disconnected": 0}
    shoves = []
    strand_detail = []

    for ep in range(episodes):
        env = HiveMindMultiAgentEnv(render_mode=None, num_cartons=num_cartons,
                                    comms=comms, msg_dropout=0.0,
                                    max_steps=max_steps)
        obs, _ = env.reset(seed=seed0 + ep)
        controller = GreedyController(env) if mode == "greedy" else None
        terminated = False

        for _ in range(env.max_steps):
            if mode == "greedy":
                action = np.asarray(controller.act(), dtype=int)
            else:
                kw = {"action_masks": env.action_masks()} if masked else {}
                raw, _ = actor.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=deterministic, **kw)
                action = joint_from_slot_actions(raw, NUM_AGENTS)

            obs, _, terminated, truncated, info = env.step(action)
            if mode == "greedy":
                controller.sync_after_step()
            if terminated or truncated:
                break

        totals["episodes"] += 1
        if terminated:
            totals["completed"] += 1

        # Where did every still-undelivered carton end up?
        depot_cell = env.depot_pos_grid
        walkable = connected_from(env, depot_cell)
        stranded_here = 0

        for slot in range(env.active_cartons):
            if env.delivered[slot]:
                continue
            totals["undelivered"] += 1
            rid = env.all_resource_ids[slot]
            carried = rid not in env.resource_ids
            p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=env.client_id)
            home = env.carton_home_cells[slot]
            hx, hy = env._grid_to_world(*home)
            moved = math.hypot(p[0] - hx, p[1] - hy)
            shoves.append(moved)

            if carried:
                continue   # in a gripper; reachability is not the question

            spots = reachable_cells(env, p)
            usable = [s for s in spots if s in walkable]
            if not spots:
                totals["unreachable"] += 1
                stranded_here += 1
                strand_detail.append((seed0 + ep, slot, moved, "no cell within reach"))
            elif not usable:
                totals["disconnected"] += 1
                stranded_here += 1
                strand_detail.append((seed0 + ep, slot, moved, "reach cells walled off"))

        if stranded_here:
            totals["stranded_eps"] += 1
        env.close()

    return totals, shoves, strand_detail


def report(label, totals, shoves, detail):
    n = max(totals["episodes"], 1)
    print(f"\n  {label}")
    print(f"  {'-' * 68}")
    print(f"     completed              {totals['completed']}/{totals['episodes']}")
    print(f"     cartons left undelivered {totals['undelivered']} "
          f"({totals['undelivered'] / n:.1f} per episode)")
    if shoves:
        print(f"     how far they were shoved  mean {np.mean(shoves):.2f} m   "
              f"max {np.max(shoves):.2f} m")
    print(f"     no standing cell in reach {totals['unreachable']}")
    print(f"     reach cells walled off    {totals['disconnected']}")
    print(f"     episodes with a stranded carton  {totals['stranded_eps']}/{totals['episodes']}")
    for seed, slot, moved, why in detail[:8]:
        print(f"         seed {seed} carton {slot}: shoved {moved:.2f} m - {why}")
    if len(detail) > 8:
        print(f"         ... and {len(detail) - 8} more")


def main():
    ap = argparse.ArgumentParser(
        description="Are the undelivered cartons still reachable when an episode ends?")
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--baseline", choices=["greedy"], default=None)
    ap.add_argument("--num-cartons", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override the episode cap. The default is the env's own.")
    args = ap.parse_args()

    if (args.model is None) == (args.baseline is None):
        ap.error("pass exactly one of a model path or --baseline greedy")

    print("=" * 78)
    print(f"  Stall diagnosis at {args.num_cartons} cartons, {args.episodes} episodes "
          f"(seeds {args.seed0}-{args.seed0 + args.episodes - 1})")
    print("=" * 78)

    if args.baseline:
        totals, shoves, detail = run(None, "greedy", args.episodes, args.num_cartons,
                                     True, args.seed0, args.max_steps)
        report("greedy baseline", totals, shoves, detail)
    else:
        model, _ = load_policy(args.model, device=get_device())
        for label, det in (("deterministic (argmax)", True), ("stochastic (sampled)", False)):
            totals, shoves, detail = run(model, "policy", args.episodes, args.num_cartons,
                                         det, args.seed0, args.max_steps)
            report(label, totals, shoves, detail)

    print("\n  verdict")
    print(f"  {'-' * 68}")
    print("     stranded cartons > 0  -> the robots broke the world by shoving cartons")
    print("                              out of reach. An ENVIRONMENT fix.")
    print("     stranded cartons = 0  -> the world stayed solvable and the policy simply")
    print("                              stopped making progress. A POLICY problem;")
    print("                              probe_policy.py says what it does instead.")


if __name__ == "__main__":
    main()
