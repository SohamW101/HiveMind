"""
Scripted greedy controller - roadmap step 5.

Each robot claims the nearest unclaimed carton, drives to it, delivers it to the depot,
and repeats until the warehouse is empty. No learning, no communication, no lookahead
beyond the current target.

WHY THIS EXISTS

    Build it before training so "it trains" is never mistaken for "it works".

Its makespan is the number a learned policy has to beat. A policy that cannot beat a
robot driving straight at the closest box has not demonstrated anything, however
smoothly its reward curve rises.

It is also the first real test of the world. A greedy run that fails on a particular
seed says the warehouse generator produced something unsolvable - a fact no amount of
staring at training curves would ever reveal.

MAKING IT A FAIR OPPONENT

A baseline that is weaker than it needs to be flatters whatever it is compared against,
so the obvious efficiencies are taken:

  - A 180-degree turn is one `backward` action, not two turns. Costing it two would
    inflate every makespan by roughly the number of reversals in a route.
  - Pickup and drop do not require facing the target - the environment checks range
    only - so no turn is ever spent lining up.
  - Every unblocked cell within drop range of the depot is a valid destination, and a
    carrying robot heads for the nearest one, so deliveries do not queue on one square.
  - Robots stop as soon as the target is within range rather than driving onto it.

WHAT IT DELIBERATELY DOES NOT DO

No task reallocation once a claim is made, no lookahead beyond the next target, and no
attempt to sequence deliveries to minimise makespan globally. Contention is handled by
strict priority plus a progress watchdog - enough to keep it moving, nowhere near
optimal. Those are the things a learned policy is supposed to discover; if
the baseline did them too, beating it would stop meaning anything.
"""
from __future__ import annotations

import math
from collections import deque

import pybullet as pb

# Action ids, from the environment's MultiDiscrete([7] * 4).
FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, PICKUP, DROP, STAY = range(7)

# (row, column) deltas for the four headings, matching the environment's yaw
# convention: heading 0 is +column (+x), and a left turn advances the index.
HEADING_DELTA = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}
DELTA_HEADING = {v: k for k, v in HEADING_DELTA.items()}

# A robot may pick up or drop when the target is within 1.5 cells, so it never has to
# stand on the target square.
REACH_CELLS = 1.5

# How many consecutive blocked steps before a robot stops yielding and moves anyway.
# Without this two robots can stand politely waiting for each other forever.
MAX_WAIT = 3


def bfs(start, goals, blocked, grid_size):
    """
    Shortest 4-connected path from `start` to the nearest cell in `goals`.

    Returns the full path including `start`, or None if no goal is reachable.
    `goals` is a set so "get within range of X" and "reach any depot slot" are the
    same query.
    """
    if start in goals:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        r, c = cur
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in prev:
                continue
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                continue
            if nxt in blocked:
                continue
            prev[nxt] = cur
            if nxt in goals:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            q.append(nxt)
    return None


class GreedyController:
    """
    Drives all four robots. Call `act(env)` once per environment step to get the joint
    action, then step the environment with it.

    Rebuild (or call `reset`) whenever the environment is reset - the shelf layout and
    carton slots are captured once per episode.
    """

    def __init__(self, env):
        self.env = env
        self.reset()

    # -- episode setup ---------------------------------------------------------
    def reset(self):
        env = self.env
        self.n = env.num_agents

        # Cells a robot cannot occupy. This used to be rebuilt here from
        # env.carton_home_cells; the env now owns the same set and enforces it - a move
        # into one is refused as an invalid action - so reading it is both shorter and
        # the only way to be certain the planner and the simulator agree about which
        # squares exist. A disagreement here is a robot pathing into a wall forever.
        self.blocked = set(env.blocked_cells)

        # Carton positions are NOT cached. Robots shove cartons: they are solid bodies
        # and a robot teleporting into one pushes it. In one seed a carton travelled
        # 1.93 m out of its starting cell, and a controller working from the cell it
        # started in drove to the wrong square, grabbed at nothing, and burned the rest
        # of the episode paying the invalid-action penalty. Read them live instead.

        # Squares a robot can drop from: within the environment's drop radius of the
        # actual depot body, and not shelf.
        #
        # These are computed, not hard-coded. The obvious four - the depot cell and its
        # three nearest neighbours - include (1, 1), which sits in shelf row 1 and is
        # usually solid. Listing it sent a carrying robot towards a square it could
        # never reach: it delivered 11 of 12 and then stalled for 1,997 steps waiting
        # for a path that did not exist.
        self.depot_pos, _ = pb.getBasePositionAndOrientation(
            env.depot_id, physicsClientId=env.client_id
        )
        self.drop_cells = set()
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if (r, c) in self.blocked:
                    continue
                wx, wy = env._grid_to_world(r, c)
                if math.hypot(wx - self.depot_pos[0],
                              wy - self.depot_pos[1]) <= env.cell_size * REACH_CELLS:
                    self.drop_cells.add((r, c))
        if not self.drop_cells:
            raise RuntimeError(
                "No unblocked cell lies within drop range of the depot - the generated "
                "warehouse is unsolvable, which is exactly the kind of thing this "
                "baseline exists to surface."
            )

        self.claim = {i: None for i in range(self.n)}   # robot -> carton slot
        self.claimed_by = {}                            # carton slot -> robot
        self.waits = {i: 0 for i in range(self.n)}
        # Progress watchdog: how far each robot was from its goal, and for how many
        # steps it has failed to get closer. See _step_towards.
        self.best_dist = {i: None for i in range(self.n)}
        self.no_progress = {i: 0 for i in range(self.n)}

    # -- geometry helpers ------------------------------------------------------
    def _cell(self, i):
        x, y, _ = self.env._canonical_pose(i)
        return self.env._world_to_grid(x, y)

    def _heading(self, i):
        _, _, yaw = self.env._canonical_pose(i)
        return int(round(yaw / (math.pi / 2.0))) % 4

    def _carton_xy(self, slot):
        """Live world position of a carton. Only valid while its body exists."""
        rid = self.env.all_resource_ids[slot]
        pos, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.env.client_id)
        return pos[0], pos[1]

    def _carton_cell(self, slot):
        return self.env._world_to_grid(*self._carton_xy(slot))

    def _can_pick(self, i, slot):
        """Measured body-to-body, exactly as the environment's pickup check does."""
        x, y, _ = self.env._canonical_pose(i)
        cx, cy = self._carton_xy(slot)
        return math.hypot(cx - x, cy - y) <= self.env.cell_size * REACH_CELLS

    def _within_reach(self, i, cell):
        """Is `cell`'s centre within the environment's pickup/drop radius of robot i?"""
        x, y, _ = self.env._canonical_pose(i)
        tx, ty = self.env._grid_to_world(*cell)
        return math.hypot(tx - x, ty - y) <= self.env.cell_size * REACH_CELLS

    def _can_drop(self, i):
        """Measured against the depot body itself, exactly as the environment does."""
        x, y, _ = self.env._canonical_pose(i)
        return math.hypot(x - self.depot_pos[0],
                          y - self.depot_pos[1]) <= self.env.cell_size * REACH_CELLS

    def _neighbourhood(self, cell):
        """Cells from which `cell` is reachable - itself plus its 8 neighbours."""
        r, c = cell
        out = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr, c + dc)
                if (0 <= n[0] < self.env.grid_size and 0 <= n[1] < self.env.grid_size
                        and n not in self.blocked):
                    out.add(n)
        return out

    # -- claiming --------------------------------------------------------------
    def _available_slots(self):
        """Carton slots still on the floor and not claimed by anyone."""
        env = self.env
        on_floor = set()
        for slot, rid in enumerate(env.all_resource_ids):
            if not env.delivered[slot] and rid in env.resource_ids:
                on_floor.add(slot)
        return on_floor - set(self.claimed_by)

    def _claim_nearest(self, i, reserved):
        """Give robot i the reachable unclaimed carton with the shortest path."""
        start = self._cell(i)
        best, best_len = None, None
        for slot in self._available_slots():
            goals = self._neighbourhood(self._carton_cell(slot))
            if not goals:
                continue          # shoved somewhere with no free cell beside it
            path = bfs(start, goals,
                       self.blocked | (reserved - {start}), self.env.grid_size)
            if path is not None and (best_len is None or len(path) < best_len):
                best, best_len = slot, len(path)
        if best is not None:
            self.claim[i] = best
            self.claimed_by[best] = i
        return best

    def _release(self, i):
        slot = self.claim[i]
        if slot is not None:
            self.claimed_by.pop(slot, None)
        self.claim[i] = None

    # -- movement --------------------------------------------------------------
    def _step_towards(self, i, goals, reserved):
        """
        One action towards the nearest cell in `goals`, plus the cell it moves into.

        `reserved` holds the cells already spoken for this step by higher-priority
        robots - see `act` for why priority rather than mutual avoidance.

        A 180-degree turn is a single backward move rather than two turns, which is the
        difference between a fair baseline and a hobbled one.
        """
        start = self._cell(i)
        path = bfs(start, goals, self.blocked | (reserved - {start}), self.env.grid_size)

        # Progress watchdog. Priority ordering removes mutual yielding, but a robot can
        # still be walled in by a lower-priority robot that has nowhere useful to go. If
        # it has not got closer to its goal for a while, plan straight through everyone:
        # a collision costs -5.0, standing still costs the rest of the episode.
        clear = bfs(start, goals, self.blocked, self.env.grid_size)
        dist = len(clear) if clear else None
        if dist is not None and (self.best_dist[i] is None or dist < self.best_dist[i]):
            self.best_dist[i] = dist
            self.no_progress[i] = 0
        else:
            self.no_progress[i] += 1

        if path is None or self.no_progress[i] > MAX_WAIT:
            if path is None:
                self.waits[i] += 1
                if self.waits[i] <= MAX_WAIT:
                    return STAY, start
            path = clear
        else:
            self.waits[i] = 0

        if path is None or len(path) < 2:
            return STAY, start

        nxt = path[1]
        want = DELTA_HEADING[(nxt[0] - start[0], nxt[1] - start[1])]
        turn = (want - self._heading(i)) % 4
        if turn == 0:
            return FORWARD, nxt
        if turn == 2:
            return BACKWARD, nxt     # one action, not two turns
        # Turning in place: the robot stays where it is this step.
        return (TURN_LEFT if turn == 1 else TURN_RIGHT), start

    def _reset_progress(self, i):
        self.best_dist[i] = None
        self.no_progress[i] = 0

    # -- the policy ------------------------------------------------------------
    def act(self, env=None):
        """
        Joint action for one environment step, as a list of `num_agents` ints.

        Robots are planned in index order and each one avoids only the cells already
        spoken for by robots planned before it. That strict priority is what stops the
        livelock: when every robot avoided every other robot's *current* cell, two
        robots would each step aside, each see the other move into the square it had
        just vacated, and swap back - forward, backward, forward, backward for the rest
        of the episode. One seed spent 1,400 steps doing exactly that with two robots
        holding cartons a dozen squares from the depot.
        """
        env = env or self.env
        actions = [STAY] * self.n
        reserved = set()

        for i in range(self.n):
            here = self._cell(i)

            if env.is_carrying[i]:
                # Carrying: head for the nearest drop square, then drop.
                self._release(i)
                if self._can_drop(i):
                    actions[i] = DROP
                    reserved.add(here)
                    self._reset_progress(i)
                    continue
                actions[i], nxt = self._step_towards(i, self.drop_cells, reserved)
                reserved.add(nxt)
                continue

            # Not carrying: make sure the claim is still live.
            slot = self.claim[i]
            if slot is not None and (env.delivered[slot]
                                     or env.all_resource_ids[slot] not in env.resource_ids):
                self._release(i)
                self._reset_progress(i)
                slot = None
            if slot is None:
                slot = self._claim_nearest(i, reserved)
                self._reset_progress(i)

            if slot is None:
                actions[i] = STAY        # nothing left to fetch
                reserved.add(here)
                continue

            if self._can_pick(i, slot):
                actions[i] = PICKUP
                reserved.add(here)
                continue

            goals = self._neighbourhood(self._carton_cell(slot))
            if not goals:
                # Shoved somewhere with no free cell beside it. Drop the claim so
                # another robot can try from a different approach.
                self._release(i)
                actions[i] = STAY
                reserved.add(here)
                continue

            actions[i], nxt = self._step_towards(i, goals, reserved)
            reserved.add(nxt)

        return actions

    def sync_after_step(self):
        """
        Re-read what each robot is actually carrying.

        The environment's pickup grabs the *nearest* carton in range, which is not
        necessarily the claimed one when two sit close together. Rather than assume,
        re-derive the claim from what is really in the gripper and free the other.
        """
        env = self.env
        for i in range(self.n):
            rid = env.carried_resource_ids[i]
            if rid is None:
                continue
            actual = env.resource_slot.get(rid)
            if actual is not None and self.claim[i] != actual:
                self._release(i)
                self.claim[i] = actual
                self.claimed_by[actual] = i
