"""
Drive one robot through a full pickup and delivery, checking the observation at
every stage. Roadmap step 3's verification, in the spirit of step 4's "verify with
play_multi.py before any training".

This exists because an observation bug is silent. A wrong reward makes training
diverge visibly; a carton status stuck on "available" just makes the policy a bit
worse, and nothing ever points at the cause. So each component is checked against
ground truth read straight from PyBullet rather than trusted.

    .venv\\Scripts\\python.exe scripts/verify_observations.py

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
    CARTON_AVAILABLE,
    CARTON_CLAIMED_BY_ME,
    CARTON_CLAIMED_BY_OTHER,
    CARTON_DELIVERED,
    MSG_TOKENS,
    NUM_CARTONS,
    LIDAR_BEAM_Z,
    LIDAR_MAX_RANGE,
    LIDAR_NUM_RAYS,
    OBS_DIM_V1,
    OBS_DIM_V2,
    OBS_DIM_V3,
    OBS_SLICES,
    OBS_WORLD_DIM,
    HiveMindMultiAgentEnv,
    describe_observation_layout,
)
from play_multi import (
    approach_cell,
    direction_for,
    path_between,
    resource_cells,
    shelf_cells,
)

TOL = 1e-4
checks = {"pass": 0, "fail": 0}


def check(label, condition, detail=""):
    if condition:
        checks["pass"] += 1
        print(f"  PASS  {label}")
    else:
        checks["fail"] += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def part(row, name):
    return np.asarray(row[OBS_SLICES[name]], dtype=float)


class Driver:
    """Minimal scripted driver for robot 0. The other three hold position."""

    def __init__(self, env):
        self.env = env
        self.cell = env._world_to_grid(*pb.getBasePositionAndOrientation(
            env.robot_ids[0], physicsClientId=env.client_id)[0][:2])
        self.direction = 0  # spawn yaw is 0, which is +column
        self.obs = None

    def act(self, action):
        self.obs, _, _, _, _ = self.env.step([action, 6, 6, 6])
        return self.obs

    def turn_to(self, target):
        for _ in range((target - self.direction) % 4):
            self.act(2)
            self.direction = (self.direction + 1) % 4

    def drive_to(self, target, blocked):
        for nxt in path_between(self.cell, target, blocked, self.env.grid_size)[1:]:
            self.turn_to(direction_for(self.cell, nxt))
            self.act(0)
            self.cell = nxt


def main():
    print("=" * 78)
    print("  Observation verification - roadmap step 3")
    print("=" * 78)
    print(describe_observation_layout())

    # -- The pin defences --------------------------------------------------------
    print("\n[1] Pinned-dimension defences")
    try:
        HiveMindMultiAgentEnv(render_mode=None, obs_dim=64).close()
        check("mismatched obs_dim is rejected", False, "constructor accepted obs_dim=64")
    except ValueError:
        check("mismatched obs_dim is rejected", True)
    try:
        HiveMindMultiAgentEnv(render_mode=None, obs_size=21).close()
        check("stale obs_size is rejected", False, "constructor accepted obs_size=21")
    except TypeError:
        check("stale obs_size is rejected", True)

    try:
        HiveMindMultiAgentEnv(render_mode=None, obs_dim=OBS_DIM_V1).close()
        check("superseded V1 width is rejected", False, "constructor accepted V1")
    except ValueError as e:
        check("superseded V1 width is rejected with an explanation",
              "superseded" in str(e), str(e)[:140])
    try:
        HiveMindMultiAgentEnv(render_mode=None, obs_dim=OBS_DIM_V2).close()
        check("superseded V2 width is rejected", False, "constructor accepted V2")
    except ValueError as e:
        check("superseded V2 width is rejected with an explanation",
              "superseded" in str(e), str(e)[:140])

    widths = sum(sl.stop - sl.start for sl in OBS_SLICES.values())
    check(f"slices tile the vector exactly ({widths} == {OBS_DIM_V3})", widths == OBS_DIM_V3)
    check("message slots are last (world indices stay stable in step 7)",
          OBS_SLICES["messages"].stop == OBS_DIM_V3
          and OBS_SLICES["messages"].start == OBS_WORLD_DIM)

    env = HiveMindMultiAgentEnv(render_mode=None)
    try:
        # -- Reset -------------------------------------------------------------
        print("\n[2] At reset")
        obs, info = env.reset(seed=7)
        space = env.observation_space

        check(f"shape is {space.shape}", obs.shape == (env.num_agents, OBS_DIM_V3),
              f"got {obs.shape}")
        check("dtype is float32", obs.dtype == np.float32, f"got {obs.dtype}")
        check("observation is inside observation_space", space.contains(obs))
        check("all values finite", bool(np.isfinite(obs).all()))
        check("velocity is zero on the first observation",
              np.allclose(part(obs[0], "own_velocity"), 0.0))
        check("elapsed time is zero", abs(part(obs[0], "elapsed_time")[0]) < TOL)
        check("nobody is carrying", not part(obs[0], "own_carrying").any()
              and not part(obs[0], "other_carrying").any())
        check("all 12 cartons read available",
              np.allclose(part(obs[0], "carton_status"), CARTON_AVAILABLE))

        # Carton positions (new in V2) against PyBullet ground truth.
        cpos = part(obs[0], "carton_positions").reshape(NUM_CARTONS, 2)
        truth = []
        for rid in env.all_resource_ids:
            cp, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=env.client_id)
            truth.append((cp[0], cp[1]))
        ok = all(abs(cpos[k][0] * env._arena_half_extent - truth[k][0]) < TOL
                 and abs(cpos[k][1] * env._arena_half_extent - truth[k][1]) < TOL
                 for k in range(NUM_CARTONS))
        check("all 12 carton positions match PyBullet", ok,
              f"slot 0 obs=({cpos[0][0]*env._arena_half_extent:.3f}, "
              f"{cpos[0][1]*env._arena_half_extent:.3f}) truth={truth[0]}")
        check("every robot sees the same carton positions",
              all(np.allclose(part(obs[j], "carton_positions"),
                              part(obs[0], "carton_positions")) for j in (1, 2, 3)))
        check("message slots are all zero", np.allclose(part(obs[0], "messages"), 0.0))

        # LiDAR (new in V3).
        lidar = part(obs[0], "lidar")
        check(f"lidar occupies {LIDAR_NUM_RAYS} slots", lidar.shape == (LIDAR_NUM_RAYS,))
        check("lidar readings are normalised into [0, 1]",
              float(lidar.min()) >= 0.0 and float(lidar.max()) <= 1.0,
              f"range [{lidar.min():.3f}, {lidar.max():.3f}]")
        check("info reports lidar distances in metres",
              len(info["lidar_distances"]) == env.num_agents
              and len(info["lidar_distances"][0]) == LIDAR_NUM_RAYS)
        check("lidar is not uniformly max range (it sees the warehouse)",
              float(np.ptp(lidar)) > 0.01, f"ptp={np.ptp(lidar):.4f}")

        # Own pose against PyBullet ground truth.
        pos, orn = pb.getBasePositionAndOrientation(env.robot_ids[0],
                                                    physicsClientId=env.client_id)
        pose = part(obs[0], "own_pose")
        check("own pose matches PyBullet",
              abs(pose[0] * env._arena_half_extent - pos[0]) < TOL
              and abs(pose[1] * env._arena_half_extent - pos[1]) < TOL,
              f"obs says ({pose[0]*env._arena_half_extent:.3f}, "
              f"{pose[1]*env._arena_half_extent:.3f}), pybullet says "
              f"({pos[0]:.3f}, {pos[1]:.3f})")

        # Robot 1's view of robot 0 must equal robot 0's view of itself. Robot 1's
        # "others" are [0, 2, 3], so robot 0 occupies the first pose triple.
        check("robot 1 sees robot 0 at the same pose robot 0 reports",
              np.allclose(part(obs[1], "other_poses")[0:3], pose, atol=TOL))

        # Depot direction against ground truth.
        depot, _ = pb.getBasePositionAndOrientation(env.depot_id,
                                                    physicsClientId=env.client_id)
        dd = part(obs[0], "depot_direction")
        check("depot direction points at the depot",
              abs(dd[0] * env._arena_span - (depot[0] - pos[0])) < TOL
              and abs(dd[1] * env._arena_span - (depot[1] - pos[1])) < TOL)

        # -- Motion ------------------------------------------------------------
        print("\n[3] After one forward move")
        driver = Driver(env)
        obs = driver.act(0)
        vel = part(obs[0], "own_velocity")
        check("velocity magnitude is one cell", abs(np.hypot(*vel) - 1.0) < 1e-3,
              f"got {vel} (magnitude {np.hypot(*vel):.4f})")
        check("elapsed time advanced",
              abs(part(obs[0], "elapsed_time")[0] - 1.0 / env.max_steps) < TOL)
        check("observation still inside observation_space", space.contains(obs))

        obs = driver.act(6)  # stay
        check("velocity returns to zero when stationary",
              np.allclose(part(obs[0], "own_velocity"), 0.0, atol=1e-3))
        driver.cell = env._world_to_grid(*pb.getBasePositionAndOrientation(
            env.robot_ids[0], physicsClientId=env.client_id)[0][:2])

        # -- Pickup ------------------------------------------------------------
        print("\n[4] After pickup")
        cells = resource_cells(env)
        blocked = shelf_cells(cells, env.grid_size) | {(0, 0)}
        target_cell = cells[0]
        target_id = env.resource_ids[0]
        slot = env.resource_slot[target_id]

        driver.drive_to(approach_cell(target_cell, blocked, env.grid_size), blocked)
        driver.turn_to(direction_for(driver.cell, target_cell))
        obs = driver.act(4)

        check("robot 0 is carrying", env.is_carrying[0])
        check("own carrying flag is set", part(obs[0], "own_carrying")[0] == 1.0)
        check("robots 1-3 see robot 0 carrying",
              all(part(obs[j], "other_carrying")[0] == 1.0 for j in (1, 2, 3)))
        check(f"carton slot {slot} reads claimed-by-me for robot 0",
              abs(part(obs[0], "carton_status")[slot] - CARTON_CLAIMED_BY_ME) < TOL,
              f"got {part(obs[0], 'carton_status')[slot]:.4f}")
        check(f"carton slot {slot} reads claimed-by-other for robots 1-3",
              all(abs(part(obs[j], "carton_status")[slot] - CARTON_CLAIMED_BY_OTHER) < TOL
                  for j in (1, 2, 3)))
        carried = part(obs[0], "carton_positions").reshape(NUM_CARTONS, 2)[slot]
        rp, _ = pb.getBasePositionAndOrientation(env.robot_ids[0],
                                                 physicsClientId=env.client_id)
        check("a carried carton reports a position near its carrier",
              np.hypot(carried[0] * env._arena_half_extent - rp[0],
                       carried[1] * env._arena_half_extent - rp[1]) < 1.0,
              f"carton at {carried * env._arena_half_extent}, robot at {rp[:2]}")
        others_available = [v for i, v in enumerate(part(obs[0], "carton_status"))
                            if i != slot]
        check("the other 11 cartons still read available",
              np.allclose(others_available, CARTON_AVAILABLE))
        check("observation still inside observation_space", space.contains(obs))

        # -- Delivery ----------------------------------------------------------
        print("\n[5] After delivery")
        driver.drive_to((0, 1), blocked)
        obs = driver.act(5)

        check("robot 0 is no longer carrying", not env.is_carrying[0])
        check("own carrying flag cleared", part(obs[0], "own_carrying")[0] == 0.0)
        dep, _ = pb.getBasePositionAndOrientation(env.depot_id,
                                                  physicsClientId=env.client_id)
        dpos = part(obs[0], "carton_positions").reshape(NUM_CARTONS, 2)[slot]
        check("a delivered carton reports the depot position",
              abs(dpos[0] * env._arena_half_extent - dep[0]) < TOL
              and abs(dpos[1] * env._arena_half_extent - dep[1]) < TOL)
        check(f"carton slot {slot} reads delivered for every robot",
              all(abs(part(obs[j], "carton_status")[slot] - CARTON_DELIVERED) < TOL
                  for j in range(env.num_agents)),
              f"robot 0 reads {part(obs[0], 'carton_status')[slot]:.4f}")
        check("info reports one delivery", env._get_info()["delivered"] == 1)
        check("remaining_resources dropped to 11",
              env._get_info()["remaining_resources"] == 11)
        check("message slots are still zero after a full cycle",
              np.allclose(part(obs[0], "messages"), 0.0))
        check("observation still inside observation_space", space.contains(obs))

        # -- Message wiring ----------------------------------------------------
        print("\n[6] Message slot wiring (step 7 rehearsal, not step 7)")
        env.messages[1] = np.linspace(0.1, 0.9, MSG_TOKENS, dtype=np.float32)
        obs = driver.act(6)
        heard = part(obs[0], "messages")[0:MSG_TOKENS]
        check("robot 0 hears robot 1's message in the first token block",
              np.allclose(heard, env.messages[1], atol=1e-6),
              f"heard {heard[:3]}... expected {env.messages[1][:3]}...")
        check("robot 1 does not hear itself",
              not np.allclose(part(obs[1], "messages")[0:MSG_TOKENS], env.messages[1]))
        check("dimension unchanged by writing messages", obs.shape[1] == OBS_DIM_V3)
        env.messages[1] = 0.0

    finally:
        env.close()

    # -- The V1 regression, kept as a permanent test ---------------------------
    # V1 carried carton status but no carton positions, so five different warehouse
    # layouts produced one byte-identical observation. That is what retired it. This
    # check exists so the blindness cannot come back unnoticed.
    print("\n[7] Layout awareness (the defect that retired V1)")
    seen_obs, seen_layouts = set(), set()
    for seed in range(5):
        e = HiveMindMultiAgentEnv(render_mode=None)
        o, _ = e.reset(seed=seed)
        seen_obs.add(o.tobytes())
        seen_layouts.add(tuple(sorted(
            e._world_to_grid(*pb.getBasePositionAndOrientation(
                r, physicsClientId=e.client_id)[0][:2]) for r in e.resource_ids)))
        e.close()
    print(f"  5 seeds -> {len(seen_layouts)} distinct layouts, "
          f"{len(seen_obs)} distinct observations")
    check("distinct warehouse layouts produce distinct observations",
          len(seen_obs) == len(seen_layouts) == 5,
          "V1 collapsed 5 layouts into 1 observation because it carried no carton "
          "positions. If this fails again, the same blindness has returned.")

    # -- Perception must agree with collision ----------------------------------
    # The invariant that makes LiDAR worth having: the beam has to see the obstacles
    # the chassis actually hits. It sits at a fixed LIDAR_BEAM_Z chosen to fall inside
    # both the chassis band and the bottom shelf plate. Reading the live chassis z
    # instead let the beam sink below the plate over a few hundred steps, and a robot
    # one cell from a shelf reported 2.5 m of clear space.
    print("\n[8] LiDAR agrees with collision geometry")
    env = HiveMindMultiAgentEnv(render_mode=None, lidar_noise=False)
    try:
        env.reset(seed=0)
        check("beam height sits inside the bottom shelf plate (0.14 - 0.22)",
              0.14 < LIDAR_BEAM_Z < 0.22, f"LIDAR_BEAM_Z={LIDAR_BEAM_Z}")

        # Drive robot 0 into the aisle one cell south of shelf row 1, then face it.
        for _ in range(3):
            env.step([2, 6, 6, 6])
        env.step([0, 6, 6, 6])            # into the shelf cell - should collide
        _, _, _, _, hit_info = env.step([6, 6, 6, 6])
        env.step([0, 6, 6, 6])            # out the far side into the aisle
        for _ in range(2):
            env.step([2, 6, 6, 6])        # turn back to face the shelf row
        _, _, _, _, info = env.step([6, 6, 6, 6])

        d = np.asarray(info["lidar_distances"][0])
        forward = float(d[LIDAR_NUM_RAYS // 2])
        print(f"  forward ray reads {forward:.3f} m; the shelf face is 0.5 m away")
        check("forward ray detects the shelf one cell ahead",
              abs(forward - 0.5) < 0.05,
              f"got {forward:.3f} m - if this is ~2.5 m the beam is passing under "
              f"the plate again")
        check("a robot standing in a shelf cell registers contact",
              hit_info["shelf_contacts"] >= 1,
              f"shelf_contacts={hit_info['shelf_contacts']}")

        # Chassis must not sink: the beam height is a constant, but a sinking chassis
        # would still break pickup ranges and wheel contacts.
        z0 = pb.getBasePositionAndOrientation(
            env.robot_ids[0], physicsClientId=env.client_id)[0][2]
        for _ in range(300):
            env.step([6, 6, 6, 6])
        z1 = pb.getBasePositionAndOrientation(
            env.robot_ids[0], physicsClientId=env.client_id)[0][2]
        print(f"  chassis z after 300 idle steps: {z0:.5f} -> {z1:.5f}")
        check("chassis does not sink over 300 steps", abs(z1 - z0) < 1e-3,
              f"drifted {z1 - z0:+.5f} m; unfixed this was -0.051 m per 300 steps")
    finally:
        env.close()

    print("\n" + "=" * 78)
    print(f"  {checks['pass']} passed, {checks['fail']} failed")
    print("=" * 78)
    return 1 if checks["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
