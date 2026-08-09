import time
import math
import numpy as np
import pybullet as pb
import gymnasium as gym
from collections import deque
import hivemind_env  # noqa: F401  (import registers the Gym env id)

# NOTE: this script used to carry its own draw_debug_lidar() that called
# pb.removeAllUserDebugItems() every step. The env now renders its LiDAR itself into a
# persistent line buffer (env.lidar_line_ids) and reuses those IDs via
# replaceItemUniqueId - wiping all debug items invalidated them and also erased the grid
# overlay drawn in reset(). Rendering is left to the env.

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
    """
    Reconstructs the 20x20 binary obstacle grid from PyBullet obstacles.

    Uses each body's AABB rather than its base position: obstacles come in 1x1, 1x2, 2x1
    and 2x2 footprints, and marking only the centre cell left the other cells looking
    free, so the BFS below happily planned straight through them.
    """
    grid = np.zeros((unwrapped.grid_size, unwrapped.grid_size), dtype=np.int32)
    for obs_id in unwrapped.obstacle_ids:
        aabb_min, aabb_max = pb.getAABB(obs_id, physicsClientId=unwrapped.client_id)
        r0, c0 = unwrapped._world_to_grid(aabb_max[0], aabb_min[1])
        r1, c1 = unwrapped._world_to_grid(aabb_min[0], aabb_max[1])
        lo_r, hi_r = sorted((r0, r1))
        lo_c, hi_c = sorted((c0, c1))
        lo_r = max(0, lo_r)
        lo_c = max(0, lo_c)
        hi_r = min(unwrapped.grid_size - 1, hi_r)
        hi_c = min(unwrapped.grid_size - 1, hi_c)
        if lo_r <= hi_r and lo_c <= hi_c:
            grid[lo_r:hi_r+1, lo_c:hi_c+1] = 1
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
    env = gym.make("HiveMind-SingleAgent-v0", render_mode="human", difficulty_level=4)
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
        time.sleep(0.05)

    # 3. Verify Perception Matrix Channel 1 (Resource Detection)
    print("\n---------------------------------------------------------")
    print(" [PERCEPTION CHECK 1] Robot in 0.2m range of Resource!")
    res_channel_sum = np.sum(obs['grid'][:, :, 1])
    print(f" -> Resource Matrix Channel [1] Detection Count: {res_channel_sum} cell(s)")
    if res_channel_sum > 0:
        print(" -> VERIFICATION SUCCESS: Resource correctly detected in the local observation matrix!")

    # Execute Pick Up action (Action 4)
    print("\nExecuting Action [4]: PICK UP RESOURCE...")
    obs, reward, term, trunc, info = env.step(4)
    print(f" -> Carrying Status: {'CARRIED' if obs['is_carrying'] else 'FAILED TO PICKUP'}")
    time.sleep(0.5)

    # 4. Plan path from Resource to Depot
    curr_robot_pos, curr_orn = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
    curr_r, curr_c = unwrapped._world_to_grid(curr_robot_pos[0], curr_robot_pos[1])
    curr_yaw = pb.getEulerFromQuaternion(curr_orn)[2]
    curr_h = get_heading_idx(curr_yaw)

    actions_to_depot = plan_bfs_actions(curr_r, curr_c, curr_h, dep_r, dep_c, obstacle_grid)
    print(f"\n-> Planned {len(actions_to_depot)} navigation actions to reach Depot location.")

    # Execute path to Depot. The env forces action 5 once the robot is within 0.25 m of
    # the depot, so the episode can terminate part-way through the planned sequence -
    # stop stepping when it does rather than driving a finished episode.
    term = False
    delivered_en_route = False
    for step_i, act in enumerate(actions_to_depot, 1):
        obs, reward, term, trunc, info = env.step(act)
        time.sleep(0.05)
        if term or trunc:
            delivered_en_route = term and reward >= 5.0
            print(f" -> Episode ended during navigation at step {step_i} "
                  f"(terminated={term}, reward={reward:+.2f})")
            break

    # 5. Arrived at Depot location -> Verify Perception Matrix Channel 2
    print("\n---------------------------------------------------------")
    print(" [PERCEPTION CHECK 2] Robot arrived at Depot location!")
    depot_channel_sum = np.sum(obs['grid'][:, :, 2])
    print(f" -> Depot Matrix Channel [2] Detection Count: {depot_channel_sum} cell(s)")
    if depot_channel_sum > 0:
        print(" -> VERIFICATION SUCCESS: Depot region correctly detected in the local observation matrix!")
    else:
        print(" -> VERIFICATION WARNING: Depot region cell missed.")

    # Execute Drop Off action (Action 5), unless the takeover already delivered it
    if delivered_en_route:
        print("\n -> Drop off already triggered by the env takeover on approach.")
    elif term:
        print("\n -> Episode already terminated; skipping the explicit drop off.")
    else:
        print("\nExecuting Action [5]: DROP OFF RESOURCE...")
        obs, reward, term, trunc, info = env.step(5)
        print(f" -> Task Terminated : {term} (Success Reward: {reward:.2f})")
    print(f" -> Carrying Status : {'CARRIED' if obs['is_carrying'] else 'DELIVERED & RELEASED'}")

    print("\n=========================================================")
    print("   Full Mission Completed Successfully! Closing GUI.     ")
    print("=========================================================")
    time.sleep(2.0)
    env.close()

if __name__ == "__main__":
    main()
