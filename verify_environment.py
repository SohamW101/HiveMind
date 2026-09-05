"""Hard-coded integration checks for the HiveMind warehouse environment.

Run from the repository root with:

    .venv/bin/python verify_environment.py

The checks deliberately use a deterministic, noise-free sensor mode. They validate
sensor and state contracts, not learning performance.
"""

import math
import sys
import argparse
from collections import deque

import numpy as np
import pybullet as pb

from hivemind_env.env import (
    LIDAR_MAX_RANGE,
    LIDAR_BEAM_Z,
    LIDAR_MIN_RANGE,
    LOCAL_MAP_CHANNELS,
    OBS_DIM_V4,
    HiveMindMultiAgentEnv,
)

FIXED_SEED = 2026
FIXED_RESOURCE_CELLS = [
    (1, 3), (1, 5), (3, 3), (3, 10), (5, 8), (5, 10),
    (7, 8), (7, 10), (9, 3), (9, 10), (11, 2), (11, 7),
]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def teleport(env, agent, cell, yaw=0.0):
    x, y = env._grid_to_world(*cell)
    pb.resetBasePositionAndOrientation(
        env.robot_ids[agent],
        [x, y, env._spawn_z],
        pb.getQuaternionFromEuler([0.0, 0.0, yaw]),
        physicsClientId=env.client_id,
    )


def route(env, start, goal):
    """Return a fixed-seed legal grid route using only cardinal moves."""
    blocked = set(env.blocked_cells)
    queue = deque([start])
    previous = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        r, c = cell
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in previous or neighbor in blocked:
                continue
            if 0 <= neighbor[0] < env.grid_size and 0 <= neighbor[1] < env.grid_size:
                previous[neighbor] = cell
                queue.append(neighbor)
    check(goal in previous, f"no legal route from {start} to {goal}")
    cells = []
    cell = goal
    while cell is not None:
        cells.append(cell)
        cell = previous[cell]
    return list(reversed(cells))


def hardcoded_gui_run(render_mode="human"):
    """Visually solve the fixed warehouse using scripted actions."""
    env = HiveMindMultiAgentEnv(
        render_mode=render_mode, lidar_noise=False,
        num_cartons=12, max_steps=1000, shaping=False,
    )
    try:
        _, info = env.reset(seed=FIXED_SEED)
        actual_resources = [
            env._world_to_grid(*pb.getBasePositionAndOrientation(
                resource, physicsClientId=env.client_id
            )[0][:2]) for resource in env.resource_ids
        ]
        check(actual_resources == FIXED_RESOURCE_CELLS, "fixed seed resource layout changed")
        print(f"GUI demo seed={FIXED_SEED}; resources={actual_resources}")

        current = env.spawn_cells[0]
        heading = 0  # east; environment starts every robot at this cardinal heading
        for resource_number, resource_cell in enumerate(FIXED_RESOURCE_CELLS, start=1):
            approach = (resource_cell[0] - 1, resource_cell[1])
            path = route(env, current, approach)
            for next_cell in path[1:]:
                dr, dc = next_cell[0] - current[0], next_cell[1] - current[1]
                desired = {(0, 1): 0, (1, 0): -math.pi / 2,
                           (0, -1): math.pi, (-1, 0): math.pi / 2}[(dr, dc)]
                while abs(math.atan2(math.sin(desired - heading), math.cos(desired - heading))) > 1e-6:
                    delta = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
                    action = 2 if delta > 0 else 3
                    env.step([action, 6, 6, 6])
                    heading += math.pi / 2 if action == 2 else -math.pi / 2
                    heading = (heading + math.pi) % (2 * math.pi) - math.pi
                _, _, terminated, truncated, step_info = env.step([0, 6, 6, 6])
                check(step_info["collisions"] == 0, "GUI route caused a collision")
                check(not terminated and not truncated, "GUI route ended before pickup")
                current = next_cell

            _, _, _, _, step_info = env.step([[4, 0], [6, 0], [6, 0], [6, 0]]) if env.comms else env.step([4, 6, 6, 6])
            check(step_info["pickups"][0], f"resource {resource_number} pickup failed")
            print(f"resource {resource_number}/12 picked at {resource_cell}")

            drop_cell = (0, 1)
            path = route(env, current, drop_cell)
            for next_cell in path[1:]:
                dr, dc = next_cell[0] - current[0], next_cell[1] - current[1]
                desired = {(0, 1): 0, (1, 0): -math.pi / 2,
                           (0, -1): math.pi, (-1, 0): math.pi / 2}[(dr, dc)]
                while abs(math.atan2(math.sin(desired - heading), math.cos(desired - heading))) > 1e-6:
                    delta = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
                    action = 2 if delta > 0 else 3
                    env.step([action, 6, 6, 6])
                    heading += math.pi / 2 if action == 2 else -math.pi / 2
                    heading = (heading + math.pi) % (2 * math.pi) - math.pi
                _, _, _, _, step_info = env.step([0, 6, 6, 6])
                check(step_info["collisions"] == 0, "GUI return route caused a collision")
                current = next_cell
            _, _, terminated, truncated, step_info = env.step([5, 6, 6, 6])
            check(step_info["deliveries"][0], f"resource {resource_number} drop failed")
            print(f"resource {resource_number}/12 delivered at depot")
        check(terminated and not truncated, "GUI demo did not terminate after all deliveries")
        print("PASS hardcoded GUI warehouse run")
    finally:
        env.close()


def test_world_observation_and_lidar():
    default_env = HiveMindMultiAgentEnv(render_mode=None, lidar_noise=False)
    try:
        default_observation, _ = default_env.reset(seed=123)
        check(default_observation.shape == (4, 177), "plan observation shape is wrong")
        check(default_env.observation_space.shape == (4, 177), "plan observation space is wrong")
    finally:
        default_env.close()

    env = HiveMindMultiAgentEnv(
        render_mode=None, lidar_noise=False, obs_dim=OBS_DIM_V4, decentralized=True
    )
    try:
        observation, info = env.reset(seed=123)
        check(observation.shape == (4, OBS_DIM_V4), "V4 observation shape is wrong")
        check(np.all(observation >= -1.0) and np.all(observation <= 1.0), "observation bounds violated")
        check(info["spawn_cells"] == [(0, 0), (0, 12), (12, 0), (12, 12)], "corner spawns are wrong")
        check(len(env.obstacle_ids) == 18, "expected 6 rows x 3 shelf segments")
        check(len(env.all_resource_ids) == 12, "expected 12 resources")
        check(info["shelf_contacts"] == 0, "a robot starts inside shelf geometry")

        scans = info["lidar_distances"]
        check(len(scans) == 4 and all(len(scan) == 72 for scan in scans), "LiDAR ray count is wrong")
        for scan in scans:
            check(np.all(np.asarray(scan) >= LIDAR_MIN_RANGE), "LiDAR returned below minimum range")
            check(np.all(np.asarray(scan) <= LIDAR_MAX_RANGE), "LiDAR returned above maximum range")
            check(np.any(np.asarray(scan) < LIDAR_MAX_RANGE), "LiDAR detected no warehouse geometry")

        before = env.local_maps.copy()
        env.step([6, 6, 6, 6])
        check(np.any(env.local_maps[:, 1:] > 0), "LiDAR did not write map evidence")
        check(np.array_equal(before[1], env.local_maps[1]) is False, "agent 1 map did not update")
        untouched = env.local_maps[0].copy()
        env.local_maps[1, 0, 0, 0] = 0.0
        check(env.local_maps[0, 0, 0, 0] == untouched[0, 0, 0], "maps are not independent")
        check(env.local_maps[0, 0, 6, 6] == 0.0, "spawn origin was not observed")
        check(np.sum(env.local_maps[0, 2]) > 0, "no static obstacle was written to map")
    finally:
        env.close()


def test_obstacles_and_robot_arbitration():
    env = HiveMindMultiAgentEnv(render_mode=None, lidar_noise=False)
    try:
        env.reset(seed=0)
        # Agent 0 faces east from the corner. The boundary makes backward invalid.
        _, _, _, _, info = env.step([1, 6, 6, 6])
        check(info["invalid_actions"][0], "boundary move was not rejected")
        check(info["collisions"] == 0, "boundary rejection caused a collision")

        # Two robots deliberately target the same free cell from opposite sides.
        teleport(env, 0, (0, 0), 0.0)
        teleport(env, 1, (0, 2), math.pi)
        _, _, _, _, info = env.step([0, 0, 6, 6])
        check(info["blocked_by_agent"] == [[0, 1]], "same-cell conflict was not arbitrated")
        check(info["collisions"] == 0, "arbitration allowed a physical collision")
    finally:
        env.close()


def test_pickup_drop_and_communication():
    env = HiveMindMultiAgentEnv(
        render_mode=None, comms=True, msg_dropout=0.0, lidar_noise=False, substeps=20
    )
    try:
        observation, info = env.reset(seed=9)
        check(np.all(observation[:, -48:] == 0.0), "initial communication state is not silent")

        resource_id = env.resource_ids[0]
        lidar_joint = next(
            joint for joint in range(pb.getNumJoints(env.robot_ids[0], physicsClientId=env.client_id))
            if pb.getJointInfo(env.robot_ids[0], joint, physicsClientId=env.client_id)[1].decode() == "lidar_joint"
        )
        lidar_before = pb.getLinkState(
            env.robot_ids[0], lidar_joint, physicsClientId=env.client_id
        )[0][2]
        lidar_joint_before = pb.getJointState(
            env.robot_ids[0], lidar_joint, physicsClientId=env.client_id
        )[0]
        lidar_type = pb.getJointInfo(
            env.robot_ids[0], lidar_joint, physicsClientId=env.client_id
        )[2]
        check(lidar_type == pb.JOINT_FIXED, "URDF LiDAR joint is not fixed")
        arm_joint = next(
            joint for joint in range(pb.getNumJoints(env.robot_ids[0], physicsClientId=env.client_id))
            if pb.getJointInfo(env.robot_ids[0], joint, physicsClientId=env.client_id)[1].decode() == "arm_yaw_joint"
        )
        arm_info = pb.getJointInfo(env.robot_ids[0], arm_joint, physicsClientId=env.client_id)
        lift_joint = next(
            joint for joint in range(pb.getNumJoints(env.robot_ids[0], physicsClientId=env.client_id))
            if pb.getJointInfo(env.robot_ids[0], joint, physicsClientId=env.client_id)[1].decode() == "arm_lift_joint"
        )
        lift_info = pb.getJointInfo(env.robot_ids[0], lift_joint, physicsClientId=env.client_id)
        lidar_post = next(
            joint for joint in range(pb.getNumJoints(env.robot_ids[0], physicsClientId=env.client_id))
            if pb.getJointInfo(env.robot_ids[0], joint, physicsClientId=env.client_id)[1].decode() == "lidar_post_joint"
        )
        check(lift_info[16] == lidar_post, "arm lift is not attached to the LiDAR mast rod")
        check(arm_info[16] == lift_joint, "arm yaw is not attached to the arm lift")
        check(lift_info[13][2] > 0.0, "arm lift axis is not vertical")
        check(lift_info[8] == 0.0 and lift_info[9] > 0.0, "arm lift limits are invalid")
        arm_z = pb.getLinkState(env.robot_ids[0], arm_joint, physicsClientId=env.client_id)[0][2]
        lidar_z = pb.getLinkState(env.robot_ids[0], lidar_joint, physicsClientId=env.client_id)[0][2]
        check(arm_z > lidar_z, "arm is not physically above the LiDAR link")
        resource_pos, _ = pb.getBasePositionAndOrientation(resource_id, physicsClientId=env.client_id)
        resource_cell = env._world_to_grid(resource_pos[0], resource_pos[1])
        approach_cell = (resource_cell[0] - 1, resource_cell[1])
        approach_x, approach_y = env._grid_to_world(*approach_cell)
        approach_yaw = math.atan2(resource_pos[1] - approach_y, resource_pos[0] - approach_x)
        teleport(env, 0, approach_cell, approach_yaw)
        _, _, _, _, info = env.step([[6, 5], [6, 5], [6, 7], [6, 9]])
        check(np.any(env.local_maps[0, 3] > 0), "resource was not represented in local map")
        teleport(env, 0, resource_cell, approach_yaw)
        _, _, _, _, info = env.step([[4, 3], [6, 5], [6, 7], [6, 9]])
        check(not info["pickups"][0] and info["invalid_actions"][0], "pickup was allowed on resource cell")
        teleport(env, 0, (resource_cell[0] - 2, resource_cell[1]), approach_yaw)
        _, _, _, _, info = env.step([[4, 3], [6, 5], [6, 7], [6, 9]])
        check(not info["pickups"][0] and info["invalid_actions"][0], "pickup was allowed from two cells away")
        # Face east so the pickup must visibly rotate toward the resource below.
        pickup_trace = []
        original_set_arm = env._set_arm_and_finger_joints

        def trace_arm(agent_idx, arm_yaw, finger_pos, arm_lift):
            if agent_idx == 0:
                pickup_trace.append((arm_yaw, arm_lift))
            return original_set_arm(agent_idx, arm_yaw, finger_pos, arm_lift)

        env._set_arm_and_finger_joints = trace_arm
        teleport(env, 0, approach_cell, 0.0)
        _, _, _, _, info = env.step([[4, 3], [6, 5], [6, 7], [6, 9]])
        check(info["pickups"][0], "pickup action failed within reach")
        check(info["is_carrying"][0], "pickup did not set carrying state")
        carried_pos, _ = pb.getBasePositionAndOrientation(
            env.carried_resource_ids[0], physicsClientId=env.client_id
        )
        lidar_after = pb.getLinkState(
            env.robot_ids[0], lidar_joint, physicsClientId=env.client_id
        )[0][2]
        lidar_joint_after = pb.getJointState(
            env.robot_ids[0], lidar_joint, physicsClientId=env.client_id
        )[0]
        check(abs(lidar_joint_after - lidar_joint_before) < 1e-9, "LiDAR joint moved during pickup")
        check(abs(pb.getJointState(env.robot_ids[0], lift_joint, physicsClientId=env.client_id)[0] - env.arm_lift_carried) < 1e-6, "arm did not finish raised")
        check(any(abs(yaw) > 0.5 for yaw, _ in pickup_trace), "arm did not rotate toward resource")
        check(any(lift < 1e-6 for _, lift in pickup_trace), "arm did not lower to resource height")
        check(any(lift > env.arm_lift_carried * 0.9 for _, lift in pickup_trace), "arm did not raise above LiDAR")
        check(abs(pickup_trace[-1][0]) < 1e-6, "arm did not return to forward position")
        check(abs(lidar_after - lidar_before) < 0.01, "LiDAR moved independently of chassis")
        check(
            carried_pos[2] - env.carton_size / 2.0 > LIDAR_BEAM_Z,
            "carried resource does not clear the fixed LiDAR plane",
        )
        robot_x, robot_y, robot_yaw = env._canonical_pose(0)
        front_x = robot_x + env.gripper_reach * math.cos(robot_yaw)
        front_y = robot_y + env.gripper_reach * math.sin(robot_yaw)
        check(
            math.hypot(carried_pos[0] - front_x, carried_pos[1] - front_y) < 0.03,
            "carried resource is not on the robot forward centreline",
        )

        teleport(env, 0, (0, 1))
        _, _, _, _, info = env.step([[5, 4], [6, 5], [6, 7], [6, 9]])
        check(info["deliveries"][0], "drop action failed at depot")
        check(info["delivered"] == 1, "delivery was not recorded")
        check(not info["is_carrying"][0], "drop did not clear carrying state")

        # A drop from inside the depot or two cells away is invalid.
        teleport(env, 0, env.depot_pos_grid)
        resource_id = env.resource_ids[0]
        resource_pos, _ = pb.getBasePositionAndOrientation(resource_id, physicsClientId=env.client_id)
        resource_cell = env._world_to_grid(resource_pos[0], resource_pos[1])
        teleport(env, 0, (resource_cell[0] - 1, resource_cell[1]))
        _, _, _, _, info = env.step([[4, 4], [6, 5], [6, 7], [6, 9]])
        check(info["pickups"][0], "second pickup setup failed")
        teleport(env, 0, env.depot_pos_grid)
        _, _, _, _, info = env.step([[5, 4], [6, 5], [6, 7], [6, 9]])
        check(not info["deliveries"][0], "drop was allowed from inside depot")
        check(info["invalid_actions"][0], "inside-depot drop was not invalid")
        teleport(env, 0, (0, 2))
        _, _, _, _, info = env.step([[5, 4], [6, 5], [6, 7], [6, 9]])
        check(not info["deliveries"][0], "drop was allowed from two cells away")
        check(info["invalid_actions"][0], "two-cell drop was not invalid")

        received = observation[0, -48:]
        check(np.all(received == 0.0), "first observation was not silent")
        observation, _, _, _, info = env.step([[6, 1], [6, 2], [6, 3], [6, 4]])
        check(np.sum(observation[0, -48:]) == 3.0, "broadcast tokens were not delivered to listeners")
        check(info["messages_dropped"] == 0, "dropout occurred despite zero dropout")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="run the fixed-seed visual demo")
    parser.add_argument("--headless-demo", action="store_true", help="run the same demo without a display")
    args = parser.parse_args()
    if args.gui:
        hardcoded_gui_run()
        return
    if args.headless_demo:
        hardcoded_gui_run(render_mode=None)
        return
    tests = [
        test_world_observation_and_lidar,
        test_obstacles_and_robot_arbitration,
        test_pickup_drop_and_communication,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PASS all environment integration checks")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise