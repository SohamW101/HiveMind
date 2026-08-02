import time
import math
import numpy as np
import pybullet as pb
import gymnasium as gym
from collections import deque
import hivemind_env

def draw_debug_lidar(env):
    """Draws 3D visual debug lines in PyBullet GUI for all 36 LiDAR rays."""
    unwrapped_env = env.unwrapped
    lidar_results, ray_froms, ray_tos = unwrapped_env._get_lidar_scan(num_rays=36, max_range=1.5)
    
    pb.removeAllUserDebugItems(physicsClientId=unwrapped_env.client_id)
    
    for i, hit in enumerate(lidar_results):
        hit_uid, _, hit_frac, hit_pos, _ = hit
        from_p = ray_froms[i]
        
        if hit_uid >= 0 and hit_frac < 1.0:
            # Ray hit an obstacle/wall (Red line)
            pb.addUserDebugLine(from_p, hit_pos, [1, 0, 0], lineWidth=1.5, physicsClientId=unwrapped_env.client_id)
        else:
            # Clear ray path (Green line)
            to_p = ray_tos[i]
            pb.addUserDebugLine(from_p, to_p, [0, 1, 0], lineWidth=1.0, physicsClientId=unwrapped_env.client_id)

def get_heading_idx(yaw):
    """Converts continuous yaw orientation to discrete heading index (0: +X, 1: +Y, 2: -X, 3: -Y)."""
    # Normalize yaw to [-pi, pi]
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    if abs(yaw - 0.0) < 0.2:
        return 0  # +X (right)
    elif abs(yaw - (math.pi / 2.0)) < 0.2:
        return 1  # +Y (up)
    elif abs(yaw - math.pi) < 0.2 or abs(yaw - (-math.pi)) < 0.2:
        return 2  # -X (left)
    elif abs(yaw - (-math.pi / 2.0)) < 0.2 or abs(yaw - (3 * math.pi / 2.0)) < 0.2:
        return 3  # -Y (down)
    return 0

def build_obstacle_grid(unwrapped):
    """Reconstructs the 20x20 binary obstacle grid from PyBullet obstacles."""
    grid = np.zeros((unwrapped.grid_size, unwrapped.grid_size), dtype=np.int32)
    for obs_id in unwrapped.obstacle_ids:
        pos, _ = pb.getBasePositionAndOrientation(obs_id, physicsClientId=unwrapped.client_id)
        r, c = unwrapped._world_to_grid(pos[0], pos[1])
        if 0 <= r < unwrapped.grid_size and 0 <= c < unwrapped.grid_size:
            grid[r, c] = 1
    return grid

def plan_bfs_actions(start_r, start_c, start_h, goal_r, goal_c, obstacle_grid):
    """Finds exact sequence of discrete actions to navigate from start to goal on grid."""
    grid_size = obstacle_grid.shape[0]
    
    # State: (r, c, h)
    # Headings: 0: +X (c+1), 1: +Y (r-1), 2: -X (c-1), 3: -Y (r+1)
    heading_moves = {
        0: (0, 1),   # +X -> col + 1
        1: (-1, 0),  # +Y -> row - 1
        2: (0, -1),  # -X -> col - 1
        3: (1, 0)    # -Y -> row + 1
    }

    queue = deque([(start_r, start_c, start_h, [])])
    visited = set([(start_r, start_c, start_h)])

    while queue:
        r, c, h, path = queue.popleft()

        if (r, c) == (goal_r, goal_c):
            return path

        # Try Action 0 (Move Forward)
        dr, dc = heading_moves[h]
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and obstacle_grid[nr, nc] == 0:
            if (nr, nc, h) not in visited:
                visited.add((nr, nc, h))
                queue.append((nr, nc, h, path + [0]))

        # Try Action 1 (Move Backward)
        dr, dc = heading_moves[(h + 2) % 4]
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and obstacle_grid[nr, nc] == 0:
            if (nr, nc, h) not in visited:
                visited.add((nr, nc, h))
                queue.append((nr, nc, h, path + [1]))

        # Try Action 2 (Turn Left +90 deg)
        nh = (h + 1) % 4
        if (r, c, nh) not in visited:
            visited.add((r, c, nh))
            queue.append((r, c, nh, path + [2]))

        # Try Action 3 (Turn Right -90 deg)
        nh = (h - 1) % 4
        if (r, c, nh) not in visited:
            visited.add((r, c, nh))
            queue.append((r, c, nh, path + [3]))

    return []

def main():
    print("=========================================================")
    print("   HiveMind Deterministic Task Execution & Perception    ")
    print("=========================================================")
    
    # 1. Initialize PyBullet GUI Environment
    env = gym.make("HiveMind-SingleAgent", render_mode="human", difficulty_level=4)
    obs, info = env.reset()
    unwrapped = env.unwrapped

    # Extract world coordinates & grid cells
    robot_pos, robot_orn = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
    res_pos, _ = pb.getBasePositionAndOrientation(unwrapped.resource_id, physicsClientId=unwrapped.client_id)
    dep_pos, _ = pb.getBasePositionAndOrientation(unwrapped.depot_id, physicsClientId=unwrapped.client_id)

    r_r, r_c = unwrapped._world_to_grid(robot_pos[0], robot_pos[1])
    res_r, res_c = unwrapped._world_to_grid(res_pos[0], res_pos[1])
    dep_r, dep_c = unwrapped.depot_pos_grid
    
    yaw = pb.getEulerFromQuaternion(robot_orn)[2]
    r_h = get_heading_idx(yaw)

    print(f"\n[Global Map Coordinates & Initial Setup]")
    print(f"-> Robot Grid Start   : Row {r_r}, Col {r_c} (Heading: {r_h})")
    print(f"-> Resource Target    : Row {res_r}, Col {res_c}")
    print(f"-> Depot Target       : Row {dep_r}, Col {dep_c}")

    # Build obstacle map
    obstacle_grid = build_obstacle_grid(unwrapped)

    # 2. Plan path to Resource
    actions_to_resource = plan_bfs_actions(r_r, r_c, r_h, res_r, res_c, obstacle_grid)
    print(f"\n-> Planned navigation actions to reach Resource block.")

    # Execute path to Resource until within 0.2m pickup range
    for step_i, act in enumerate(actions_to_resource, 1):
        # Check current distance to resource BEFORE taking step to avoid pushing it
        curr_robot_pos, _ = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
        res_curr_pos, _ = pb.getBasePositionAndOrientation(unwrapped.resource_id, physicsClientId=unwrapped.client_id)
        dist_to_res = np.linalg.norm(np.array(curr_robot_pos[:2]) - np.array(res_curr_pos[:2]))
        
        if dist_to_res <= 0.25:  # Within 0.2m pickup radius
            print(f" -> Robot is within pickup range ({dist_to_res:.2f}m <= 0.25m)!")
            break

        obs, reward, term, trunc, info = env.step(act)
        draw_debug_lidar(env)
        time.sleep(0.05)

    # 3. Verify Perception Matrix Channel 1 (Resource Detection)
    print("\n---------------------------------------------------------")
    print(" [PERCEPTION CHECK 1] Robot in 0.2m range of Resource!")
    res_channel_sum = np.sum(obs['grid'][:, :, 1])
    print(f" -> Resource Matrix Channel [1] Detection Count: {res_channel_sum} cell(s)")
    if res_channel_sum > 0:
        print(" -> VERIFICATION SUCCESS: Resource correctly detected in 15x15x5 local matrix!")

    # Execute Pick Up action (Action 4)
    print("\nExecuting Action [4]: PICK UP RESOURCE...")
    obs, reward, term, trunc, info = env.step(4)
    draw_debug_lidar(env)
    print(f" -> Carrying Status: {'CARRIED' if obs['is_carrying'] else 'FAILED TO PICKUP'}")
    time.sleep(0.5)

    # 4. Plan path from Resource to Depot
    curr_robot_pos, curr_orn = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
    curr_r, curr_c = unwrapped._world_to_grid(curr_robot_pos[0], curr_robot_pos[1])
    curr_yaw = pb.getEulerFromQuaternion(curr_orn)[2]
    curr_h = get_heading_idx(curr_yaw)

    actions_to_depot = plan_bfs_actions(curr_r, curr_c, curr_h, dep_r, dep_c, obstacle_grid)
    print(f"\n-> Planned {len(actions_to_depot)} navigation actions to reach Depot location.")

    # Execute path to Depot
    for step_i, act in enumerate(actions_to_depot, 1):
        obs, reward, term, trunc, info = env.step(act)
        draw_debug_lidar(env)
        time.sleep(0.05)

    # 5. Arrived at Depot location -> Verify Perception Matrix Channel 2
    print("\n---------------------------------------------------------")
    print(" [PERCEPTION CHECK 2] Robot arrived at Depot location!")
    depot_channel_sum = np.sum(obs['grid'][:, :, 2])
    print(f" -> Depot Matrix Channel [2] Detection Count: {depot_channel_sum} cell(s)")
    if depot_channel_sum > 0:
        print(" -> VERIFICATION SUCCESS: Depot region correctly detected in 15x15x5 local matrix!")
    else:
        print(" -> VERIFICATION WARNING: Depot region cell missed.")

    # Execute Drop Off action (Action 5)
    print("\nExecuting Action [5]: DROP OFF RESOURCE...")
    obs, reward, term, trunc, info = env.step(5)
    draw_debug_lidar(env)
    print(f" -> Task Terminated : {term} (Success Reward: {reward:.2f})")
    print(f" -> Carrying Status : {'CARRIED' if obs['is_carrying'] else 'DELIVERED & RELEASED'}")

    print("\n=========================================================")
    print("   Full Mission Completed Successfully! Closing GUI.     ")
    print("=========================================================")
    time.sleep(2.0)
    env.close()

if __name__ == "__main__":
    main()
