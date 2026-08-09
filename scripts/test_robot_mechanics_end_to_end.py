import os
import sys
import time
import math
import heapq
import argparse
import numpy as np
import pybullet as pb

# Add workspace directory to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from hivemind_env.env import HiveMindSingleAgentEnv

def astar_grid_path(grid, start, goal, forbidden_cells=None):
    """
    Finds shortest 4-connected grid path from start (r,c) to goal (r,c) avoiding obstacles (1)
    and any forbidden cells (e.g. resource or depot cell).
    """
    rows, cols = grid.shape
    if forbidden_cells is None:
        forbidden_cells = set()
    else:
        forbidden_cells = set(forbidden_cells)

    if grid[start] == 1 or grid[goal] == 1 or start in forbidden_cells or goal in forbidden_cells:
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        cr, cc = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in forbidden_cells:
                tentative_g = g_score[current] + 1
                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score[(nr, nc)] = tentative_g + (abs(nr - goal[0]) + abs(nc - goal[1]))
                    heapq.heappush(open_set, (f_score[(nr, nc)], (nr, nc)))
    return None

def normalize_angle(angle):
    """Normalizes angle to range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

def path_to_actions(env, grid_path):
    """Translates a sequence of grid cells into discrete environment actions."""
    actions = []
    r_pos, r_orn = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
    cur_yaw = normalize_angle(pb.getEulerFromQuaternion(r_orn)[2])

    for i in range(len(grid_path) - 1):
        cr, cc = grid_path[i]
        nr, nc = grid_path[i + 1]

        dr, dc = nr - cr, nc - cc
        if dr == -1 and dc == 0:
            target_yaw = math.pi / 2.0   # Up (+Y)
        elif dr == 1 and dc == 0:
            target_yaw = -math.pi / 2.0  # Down (-Y)
        elif dr == 0 and dc == 1:
            target_yaw = 0.0             # Right (+X)
        elif dr == 0 and dc == -1:
            target_yaw = math.pi         # Left (-X)
        else:
            continue

        diff = normalize_angle(target_yaw - cur_yaw)

        if abs(diff) < 0.1:
            actions.append(0)  # Move Forward
        elif 0.1 <= diff <= math.pi - 0.1:
            actions.append(2)  # Turn Left (+90 deg)
            actions.append(0)  # Move Forward
            cur_yaw = normalize_angle(cur_yaw + math.pi / 2.0)
        elif -math.pi + 0.1 <= diff <= -0.1:
            actions.append(3)  # Turn Right (-90 deg)
            actions.append(0)  # Move Forward
            cur_yaw = normalize_angle(cur_yaw - math.pi / 2.0)
        else:  # Around 180 deg turn
            actions.append(2)  # Turn Left
            actions.append(2)  # Turn Left
            actions.append(0)  # Move Forward
            cur_yaw = normalize_angle(cur_yaw + math.pi)

    return actions

def get_obstacle_grid_from_env(env):
    """Extracts binary obstacle grid (20x20) from environment obstacle bodies."""
    grid = np.zeros((env.grid_size, env.grid_size), dtype=np.int32)
    for obs_id in env.obstacle_ids:
        aabb_min, aabb_max = pb.getAABB(obs_id, physicsClientId=env.client_id)
        min_r, min_c = env._world_to_grid(aabb_max[0], aabb_min[1])
        max_r, max_c = env._world_to_grid(aabb_min[0], aabb_max[1])
        r0, r1 = min(min_r, max_r), max(min_r, max_r)
        c0, c1 = min(min_c, max_c), max(min_c, max_c)
        grid[r0:r1+1, c0:c1+1] = 1
    return grid

def verify_lidar_global_transforms(env):
    """
    RIGOROUS GLOBAL TRANSFORM LIDAR TEST:
    Transforms local Lidar raycast hits into Global World Coordinates:
        X_global = R_x + d * cos(yaw + ray_angle)
        Y_global = R_y + d * sin(yaw + ray_angle)
    And compares these Lidar-derived global coordinates against PyBullet ground-truth geometry.
    """
    print("\n  --- RUNNING RIGOROUS GLOBAL TRANSFORM LIDAR VERIFICATION ---")

    robot_pos, robot_orn = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
    rx, ry, rz = robot_pos
    yaw = pb.getEulerFromQuaternion(robot_orn)[2]

    num_rays = 360
    max_range = 4.0
    lidar_results, ray_froms, ray_tos = env._get_lidar_scan(num_rays=num_rays, max_range=max_range)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    obstacle_hits_verified = 0
    resource_hits_verified = 0

    res_pos, _ = pb.getBasePositionAndOrientation(env.resource_id, physicsClientId=env.client_id)
    res_center = np.array(res_pos[:2])

    grid_map = get_obstacle_grid_from_env(env)

    for i, hit in enumerate(lidar_results):
        hit_uid, hit_link, hit_frac, hit_pos, hit_norm = hit
        if hit_uid >= 0 and hit_frac < 1.0:
            dist = hit_frac * max_range
            abs_angle = yaw + angles[i]

            # FORWARD KINEMATIC TRANSFORM TO GLOBAL FRAME
            global_x_lidar = rx + dist * math.cos(abs_angle)
            global_y_lidar = ry + dist * math.sin(abs_angle)

            # PyBullet reported hit position in global frame
            pb_global_x, pb_global_y = hit_pos[0], hit_pos[1]

            # 1. Verify mathematical transform against PyBullet hit position
            trans_err = np.linalg.norm(np.array([global_x_lidar, global_y_lidar]) - np.array([pb_global_x, pb_global_y]))
            assert trans_err < 0.01, f"Transform Error! Lidar calc ({global_x_lidar:.3f}, {global_y_lidar:.3f}) vs PyBullet ({pb_global_x:.3f}, {pb_global_y:.3f})"

            # 2. Verify Lidar hit cell in Global Grid Map
            grid_r, grid_c = env._world_to_grid(global_x_lidar, global_y_lidar)

            if hit_uid in env.obstacle_ids or hit_uid in env.wall_ids:
                obstacle_hits_verified += 1
                # Check that Lidar hit location falls inside or adjacent to an obstacle grid cell
                is_valid_obs_cell = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        check_r, check_c = grid_r + dr, grid_c + dc
                        if 0 <= check_r < env.grid_size and 0 <= check_c < env.grid_size:
                            if grid_map[check_r, check_c] == 1:
                                is_valid_obs_cell = True
                                break
                    if is_valid_obs_cell:
                        break
                # Boundaries/Walls check
                if abs(global_x_lidar) >= 1.95 or abs(global_y_lidar) >= 1.95:
                    is_valid_obs_cell = True

                assert is_valid_obs_cell, f"Lidar reported obstacle hit at Global ({global_x_lidar:.2f}, {global_y_lidar:.2f}) [Cell ({grid_r}, {grid_c})], but no obstacle exists there!"

            elif hit_uid == env.resource_id:
                resource_hits_verified += 1
                # Lidar hit position on cylinder boundary should be 0.06m from cylinder center
                dist_to_center = np.linalg.norm(np.array([global_x_lidar, global_y_lidar]) - res_center)
                print(f"      • Resource Lidar Hit Global Coord: ({global_x_lidar:.3f}, {global_y_lidar:.3f}) | Distance to Center = {dist_to_center:.4f}m (Radius = 0.060m)")
                assert abs(dist_to_center - 0.06) < 0.02, f"Resource Lidar hit position inaccurate! Expected radius 0.06m, got {dist_to_center:.4f}m"

    print(f"  [PASSED] Global Transform Kinematics Verified across {obstacle_hits_verified} Obstacle Hits and {resource_hits_verified} Resource Hits!")

def run_end_to_end_mechanics_test(render_gui=False, seed=None):
    print("=" * 80)
    print("  HIVEMIND RIGOROUS LIDAR GLOBAL TRANSFORM & ADJACENT CELL TEST  ")
    print("=" * 80)

    render_mode = "human" if render_gui else None
    env = HiveMindSingleAgentEnv(render_mode=render_mode, difficulty_level=4)

    if seed is None:
        seed = int(time.time() * 1000) % 100000
    print(f"Generated Random Seed: {seed}")

    obs, info = env.reset(seed=seed)

    r_pos, r_orn = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
    res_pos, _ = pb.getBasePositionAndOrientation(env.resource_id, physicsClientId=env.client_id)
    dep_pos, _ = pb.getBasePositionAndOrientation(env.depot_id, physicsClientId=env.client_id)

    start_grid = env._world_to_grid(r_pos[0], r_pos[1])
    res_grid = env._world_to_grid(res_pos[0], res_pos[1])
    dep_grid = env._world_to_grid(dep_pos[0], dep_pos[1])

    print(f"\n[1/6] LEVEL 4 RANDOMIZED MAP GENERATED:")
    print(f"  - Robot Start Cell   : {start_grid} | World ({r_pos[0]:.2f}, {r_pos[1]:.2f})")
    print(f"  - Resource Cell      : {res_grid} | World ({res_pos[0]:.2f}, {res_pos[1]:.2f}) [FORBIDDEN - DO NOT ENTER]")
    print(f"  - Depot Cell         : {dep_grid} | World ({dep_pos[0]:.2f}, {dep_pos[1]:.2f}) [FORBIDDEN - DO NOT ENTER]")
    print(f"  - Number of Obstacles: {len(env.obstacle_ids)}")

    # Extract 20x20 grid map
    grid_map = get_obstacle_grid_from_env(env)

    # Run Rigorous Lidar Global Transform Verification
    verify_lidar_global_transforms(env)

    # --------------------------------------------------------------------------
    # STEP 1: A* PATHFINDING TO AN ADJACENT CELL OF RESOURCE (STRICT NO ENTRY TO RESOURCE CELL)
    # --------------------------------------------------------------------------
    print("\n[2/6] PLANNING A* PATH TO RESOURCE ADJACENT CELL (STRICT FORBIDDEN CELL PROTECTION)...")
    adj_res_cells = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ar, ac = res_grid[0] + dr, res_grid[1] + dc
        if 0 <= ar < env.grid_size and 0 <= ac < env.grid_size and grid_map[ar, ac] == 0:
            adj_res_cells.append((ar, ac))

    path_to_res = None
    target_res_adj = None
    forbidden = {res_grid, dep_grid}

    for adj in adj_res_cells:
        path = astar_grid_path(grid_map, start_grid, adj, forbidden_cells=forbidden)
        if path:
            path_to_res = path
            target_res_adj = adj
            break

    assert path_to_res is not None, f"Failed to find A* path from {start_grid} to resource adjacent cells!"
    print(f"  -> A* Path Found ({len(path_to_res)} cells): {path_to_res}")
    assert res_grid not in path_to_res, "CRITICAL ERROR: A* path illegally entered Resource cell!"
    assert dep_grid not in path_to_res, "CRITICAL ERROR: A* path illegally entered Depot cell!"

    actions_to_res = path_to_actions(env, path_to_res)
    print(f"  -> Executing Action Sequence ({len(actions_to_res)} steps): {actions_to_res}")

    # The env forces action 4 as soon as the robot comes within 0.25 m of the resource,
    # so an A* path that routes through a cell adjacent to it will pick up mid-path. That
    # is expected behaviour, not a failure - stop walking the plan once it happens.
    picked_up_en_route = False
    for step_num, act in enumerate(actions_to_res):
        obs, reward, term, trunc, info = env.step(act)
        curr_p, _ = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        curr_g = env._world_to_grid(curr_p[0], curr_p[1])
        assert curr_g != res_grid, f"CRITICAL ENTRY ERROR: Robot entered Resource cell {res_grid} at step {step_num}!"
        assert curr_g != dep_grid, f"CRITICAL ENTRY ERROR: Robot entered Depot cell {dep_grid} at step {step_num}!"
        assert not term, "Unexpected collision/termination while navigating!"
        if obs["is_carrying"] == 1:
            picked_up_en_route = True
            print(f"  -> Env takeover triggered pickup at step {step_num} "
                  f"(requested action {info['requested_action']} -> executed "
                  f"{info['executed_action']}), reward {reward:+.2f}")
            break

    print(f"  -> Robot arrived at resource adjacent cell: {target_res_adj} (Distance to resource = {np.linalg.norm(np.array(curr_p[:2]) - np.array(res_pos[:2])):.3f}m)")

    # --------------------------------------------------------------------------
    # STEP 2: RESOURCE PICKUP (ACTION 4 FROM ADJACENT CELL)
    # --------------------------------------------------------------------------
    if picked_up_en_route:
        print("\n[3/6] PICKUP ALREADY COMPLETED BY ENV TAKEOVER - skipping explicit action 4.")
    else:
        print("\n[3/6] EXECUTING ACTION 4 (PICKUP FROM ADJACENT CELL)...")
        obs, reward, term, trunc, info = env.step(4)
        print(f"  -> Pickup Reward         : {reward:+.2f}")
    print(f"  -> is_carrying Status    : {obs['is_carrying']} (Expected: 1)")
    print(f"  -> Arm Cardinal Angle    : {math.degrees(env.current_arm_yaw):.1f}deg")
    print(f"  -> Claw Finger Clamped   : {env.current_finger_pos:.3f}m")
    print(f"  -> Lidar Mast Elevation  : {env.current_lidar_height:.3f}m")

    assert obs["is_carrying"] == 1, "Pickup failed: is_carrying should be 1!"
    assert env.current_finger_pos == -0.01, "Gripper claw did not close!"
    assert env.current_lidar_height == 0.07, "Lidar mast did not elevate!"

    # --------------------------------------------------------------------------
    # STEP 3: A* PATHFINDING TO AN ADJACENT CELL OF DEPOT (STRICT NO ENTRY TO DEPOT CELL)
    # --------------------------------------------------------------------------
    print("\n[4/6] PLANNING A* PATH TO DEPOT ADJACENT CELL (STRICT FORBIDDEN CELL PROTECTION)...")
    adj_dep_cells = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ar, ac = dep_grid[0] + dr, dep_grid[1] + dc
        if 0 <= ar < env.grid_size and 0 <= ac < env.grid_size and grid_map[ar, ac] == 0:
            adj_dep_cells.append((ar, ac))

    # Plan from where the robot actually is. If the takeover fired mid-path it stopped
    # short of target_res_adj, and planning from a cell it never reached would produce a
    # sequence that walks it through walls.
    curr_p, _ = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
    depot_start = env._world_to_grid(curr_p[0], curr_p[1])
    print(f"  -> Planning from the robot's actual cell: {depot_start}")

    path_to_depot = None
    target_dep_adj = None
    for adj in adj_dep_cells:
        path = astar_grid_path(grid_map, depot_start, adj, forbidden_cells=forbidden)
        if path:
            path_to_depot = path
            target_dep_adj = adj
            break

    assert path_to_depot is not None, f"Failed to find A* path from {depot_start} to depot adjacent cells!"
    print(f"  -> A* Path to Depot Adjacent Cell Found ({len(path_to_depot)} cells): {path_to_depot}")
    assert dep_grid not in path_to_depot, "CRITICAL ERROR: A* path illegally entered Depot cell!"

    actions_to_depot = path_to_actions(env, path_to_depot)
    print(f"  -> Executing Action Sequence to Depot ({len(actions_to_depot)} steps): {actions_to_depot}")

    # Same takeover caveat as the resource leg: coming within 0.25 m of the depot forces
    # action 5, which delivers and terminates the episode mid-plan.
    dropped_en_route = False
    for step_num, act in enumerate(actions_to_depot):
        obs, reward, term, trunc, info = env.step(act)
        curr_p, _ = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        curr_g = env._world_to_grid(curr_p[0], curr_p[1])
        assert curr_g != dep_grid, f"CRITICAL ENTRY ERROR: Robot entered Depot cell {dep_grid} at step {step_num}!"
        if term and obs["is_carrying"] == 0:
            dropped_en_route = True
            print(f"  -> Env takeover triggered dropoff at step {step_num} "
                  f"(requested action {info['requested_action']} -> executed "
                  f"{info['executed_action']}), reward {reward:+.2f}")
            break
        assert not term, "Unexpected collision/termination while navigating to depot!"
        assert obs["is_carrying"] == 1, "Lost resource while carrying!"

    print(f"  -> Robot finished the depot leg toward adjacent cell {target_dep_adj} "
          f"(Distance to depot = {np.linalg.norm(np.array(curr_p[:2]) - np.array(dep_pos[:2])):.3f}m)")

    # --------------------------------------------------------------------------
    # STEP 4: DEPOT DROPOFF (ACTION 5 FROM ADJACENT CELL INTO DEPOT)
    # --------------------------------------------------------------------------
    if dropped_en_route:
        print("\n[5/6] DROP OFF ALREADY COMPLETED BY ENV TAKEOVER - skipping explicit action 5.")
    else:
        print("\n[5/6] EXECUTING ACTION 5 (DROP OFF INTO DEPOT FROM ADJACENT CELL)...")
        obs, reward, term, trunc, info = env.step(5)
    print(f"  -> Dropoff Reward        : {reward:+.2f}")
    print(f"  -> is_carrying Status    : {obs['is_carrying']} (Expected: 0)")
    print(f"  -> Claw Finger Released  : {env.current_finger_pos:.3f}m")
    print(f"  -> Lidar Mast Lowered    : {env.current_lidar_height:.3f}m")
    print(f"  -> Episode Terminated    : {term} (Expected: True)")

    assert obs["is_carrying"] == 0, "Dropoff failed: is_carrying should be 0!"
    assert env.current_finger_pos == 0.015, "Gripper claw did not open!"
    assert env.current_lidar_height == 0.0, "Lidar mast did not lower!"
    assert term == True, "Episode did not terminate on successful dropoff!"

    final_res_pos, _ = pb.getBasePositionAndOrientation(env.resource_id, physicsClientId=env.client_id)
    final_res_dist_to_depot = np.linalg.norm(np.array(final_res_pos[:2]) - np.array(dep_pos[:2]))
    print(f"  -> Final Resource position after dropoff: ({final_res_pos[0]:.2f}, {final_res_pos[1]:.2f}) | Distance to Depot Center = {final_res_dist_to_depot:.3f}m")
    assert final_res_dist_to_depot < 0.05, "Resource was not dropped precisely into depot center!"

    print("\n" + "=" * 80)
    print("  [PASSED] ALL RIGOROUS LIDAR GLOBAL TRANSFORM & ADJACENT CELL TESTS SUCCESSFUL!  ")
    print("=" * 80)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Level 4 A* Adjacent Cell & Lidar Global Transform Test")
    parser.add_argument("--gui", action="store_true", help="Run PyBullet in GUI 3D visual mode (60 FPS)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for map generation")
    args = parser.parse_args()

    run_end_to_end_mechanics_test(render_gui=args.gui, seed=args.seed)
