import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import random
from collections import deque
import math
import time

def _bfs_path_exists(grid, start, goal):
    """Checks if a valid path exists between start and goal on a binary grid."""
    rows, cols = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return False
    queue = deque([start])
    visited = set([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

class HiveMindSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, difficulty_level=1):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty_level = difficulty_level
        self.dt = 1.0 / 240.0
        self.is_carrying = False
        self.grid_size = 20
        self.cell_size = 0.2
        
        # Actions: 0: Forward, 1: Backward, 2: Turn Left, 3: Turn Right, 4: Pick Up, 5: Drop Off, 6: Stay
        self.action_space = spaces.Discrete(7)
        
        # Observation: Local grid around agent 15x15x5 
        # Channels: 0: obstacles, 1: resources, 2: depot, 3: boundaries, 4: agent's heading
        self.obs_size = 15
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 5), dtype=np.float32),
            "is_carrying": spaces.Discrete(2)
        })

        if self.render_mode == "human":
            self.client_id = pb.connect(pb.GUI)
        else:
            self.client_id = pb.connect(pb.DIRECT)
            
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.robot_id = None
        self.resource_id = None
        self.depot_id = None
        self.obstacle_ids = []
        self.depot_pos_grid = (0, 0)
        self.max_steps = 500
        self.current_step = 0

    def _grid_to_world(self, r, c):
        x = (c - self.grid_size/2.0) * self.cell_size + (self.cell_size/2.0)
        y = (self.grid_size/2.0 - r) * self.cell_size - (self.cell_size/2.0)
        return x, y
        
    def _world_to_grid(self, x, y):
        c = int((x / self.cell_size) + self.grid_size/2.0)
        r = int(self.grid_size/2.0 - (y / self.cell_size))
        return r, c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.is_carrying = False
        self.current_step = 0
        
        pb.resetSimulation(physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        pb.loadURDF("plane.urdf", physicsClientId=self.client_id)

        r_pos, res_pos, dep_pos, obstacles = self._generate_valid_map()
        self.depot_pos_grid = dep_pos

        rx, ry = self._grid_to_world(r_pos[0], r_pos[1])
        resx, resy = self._grid_to_world(res_pos[0], res_pos[1])
        dx, dy = self._grid_to_world(dep_pos[0], dep_pos[1])

        # Robot
        robot_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.4, self.cell_size*0.4, 0.05], physicsClientId=self.client_id)
        robot_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.4, self.cell_size*0.4, 0.05], rgbaColor=[0, 0, 1, 1], physicsClientId=self.client_id)
        self.robot_id = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=robot_col, baseVisualShapeIndex=robot_vis, basePosition=[rx, ry, 0.05], physicsClientId=self.client_id)

        # Resource
        res_col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=self.cell_size*0.3, height=0.1, physicsClientId=self.client_id)
        res_vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=self.cell_size*0.3, length=0.1, rgbaColor=[0, 1, 0, 1], physicsClientId=self.client_id)
        self.resource_id = pb.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=res_col, baseVisualShapeIndex=res_vis, basePosition=[resx, resy, 0.05], physicsClientId=self.client_id)

        # Depot (Visual only)
        depot_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.5, self.cell_size*0.5, 0.01], rgbaColor=[1, 0, 0, 0.5], physicsClientId=self.client_id)
        self.depot_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=depot_vis, basePosition=[dx, dy, 0.01], physicsClientId=self.client_id)

        # Obstacles
        self.obstacle_ids = []
        for (obs_r, obs_c, size_r, size_c) in obstacles:
            cx, cy = self._grid_to_world(obs_r + size_r/2.0 - 0.5, obs_c + size_c/2.0 - 0.5)
            hx, hy = (size_c * self.cell_size) / 2.0, (size_r * self.cell_size) / 2.0
            
            obs_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[hx, hy, 0.1], physicsClientId=self.client_id)
            obs_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[hx, hy, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client_id)
            obs_id = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=obs_col, baseVisualShapeIndex=obs_vis, basePosition=[cx, cy, 0.1], physicsClientId=self.client_id)
            self.obstacle_ids.append(obs_id)

        # Static Boundary Walls around the 20x20 grid (4.0m x 4.0m)
        self.wall_ids = []
        wall_half_extents = [(2.2, 0.1, 0.2), (2.2, 0.1, 0.2), (0.1, 2.0, 0.2), (0.1, 2.0, 0.2)]
        wall_positions = [(0, 2.1, 0.1), (0, -2.1, 0.1), (-2.1, 0, 0.1), (2.1, 0, 0.1)]
        for he, pos in zip(wall_half_extents, wall_positions):
            w_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.client_id)
            w_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.client_id)
            w_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=w_col, baseVisualShapeIndex=w_vis, basePosition=pos, physicsClientId=self.client_id)
            self.wall_ids.append(w_id)
            
        # Draw grid lines for visualization
        if self.render_mode == "human":
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            for i in range(self.grid_size + 1):
                x = (i - self.grid_size/2.0) * self.cell_size
                pb.addUserDebugLine([x, -self.grid_size/2.0 * self.cell_size, 0.01], 
                                    [x, self.grid_size/2.0 * self.cell_size, 0.01], 
                                    [0,0,0], physicsClientId=self.client_id)
                y = (i - self.grid_size/2.0) * self.cell_size
                pb.addUserDebugLine([-self.grid_size/2.0 * self.cell_size, y, 0.01], 
                                    [self.grid_size/2.0 * self.cell_size, y, 0.01], 
                                    [0,0,0], physicsClientId=self.client_id)

        for _ in range(10):
            pb.stepSimulation(physicsClientId=self.client_id)

        return self._get_obs(), self._get_info()

    def _generate_valid_map(self):
        while True:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            
            if self.difficulty_level == 1:
                r_pos, res_pos, dep_pos = (10, 10), (15, 15), (5, 5)
                num_obstacles = 0
            else:
                r_pos = (random.randint(2, 17), random.randint(2, 17))
                res_pos = (random.randint(2, 17), random.randint(2, 17))
                dep_pos = (random.randint(2, 17), random.randint(2, 17))
                while r_pos == res_pos or r_pos == dep_pos or res_pos == dep_pos:
                    res_pos = (random.randint(2, 17), random.randint(2, 17))
                    dep_pos = (random.randint(2, 17), random.randint(2, 17))
                num_obstacles = 0 if self.difficulty_level == 2 else random.randint(3, 8)
                
            obstacles = []
            for _ in range(num_obstacles):
                size_r, size_c = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
                obs_r = random.randint(0, self.grid_size - size_r)
                obs_c = random.randint(0, self.grid_size - size_c)
                
                overlap = False
                for pos in [r_pos, res_pos, dep_pos]:
                    if obs_r <= pos[0] < obs_r + size_r and obs_c <= pos[1] < obs_c + size_c:
                        overlap = True
                        break
                if not overlap:
                    grid[obs_r:obs_r+size_r, obs_c:obs_c+size_c] = 1
                    obstacles.append((obs_r, obs_c, size_r, size_c))
                    
            if _bfs_path_exists(grid, r_pos, res_pos) and _bfs_path_exists(grid, res_pos, dep_pos):
                return r_pos, res_pos, dep_pos, obstacles

    def _step_substep(self):
        pb.stepSimulation(physicsClientId=self.client_id)
        if self.render_mode == "human":
            time.sleep(0.015)  # 60 FPS smooth GUI visual delay per physics sub-step

    def step(self, action):
        self.current_step += 1
        reward = -0.01  # Time penalty
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rx, ry, _ = robot_pos
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]

        prev_carrying = self.is_carrying
        res_pos_world, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
        dep_pos_world, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)
        target_pos_world = res_pos_world[:2] if not prev_carrying else dep_pos_world[:2]
        dist_prev = np.linalg.norm(np.array([rx, ry]) - np.array(target_pos_world))

        num_substeps = 30  # 30 frames for smooth visual transition

        if action == 0:  # Move Forward 0.2m (1 cell)
            target_x = rx + self.cell_size * math.cos(yaw)
            target_y = ry + self.cell_size * math.sin(yaw)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                ix = rx + (target_x - rx) * alpha
                iy = ry + (target_y - ry) * alpha
                pb.resetBasePositionAndOrientation(self.robot_id, [ix, iy, 0.05], robot_orn, physicsClientId=self.client_id)
                self._step_substep()
                if self.is_carrying:
                    pb.resetBasePositionAndOrientation(self.resource_id, [ix, iy, 0.15], robot_orn, physicsClientId=self.client_id)

        elif action == 1:  # Move Backward 0.2m (1 cell)
            target_x = rx - self.cell_size * math.cos(yaw)
            target_y = ry - self.cell_size * math.sin(yaw)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                ix = rx + (target_x - rx) * alpha
                iy = ry + (target_y - ry) * alpha
                pb.resetBasePositionAndOrientation(self.robot_id, [ix, iy, 0.05], robot_orn, physicsClientId=self.client_id)
                self._step_substep()
                if self.is_carrying:
                    pb.resetBasePositionAndOrientation(self.resource_id, [ix, iy, 0.15], robot_orn, physicsClientId=self.client_id)

        elif action == 2:  # Turn Left 90 degrees (+pi/2)
            target_yaw = yaw + (math.pi / 2.0)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                iyaw = yaw + (target_yaw - yaw) * alpha
                iorn = pb.getQuaternionFromEuler([0, 0, iyaw], physicsClientId=self.client_id)
                pb.resetBasePositionAndOrientation(self.robot_id, [rx, ry, 0.05], iorn, physicsClientId=self.client_id)
                self._step_substep()
                if self.is_carrying:
                    pb.resetBasePositionAndOrientation(self.resource_id, [rx, ry, 0.15], iorn, physicsClientId=self.client_id)

        elif action == 3:  # Turn Right 90 degrees (-pi/2)
            target_yaw = yaw - (math.pi / 2.0)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                iyaw = yaw + (target_yaw - yaw) * alpha
                iorn = pb.getQuaternionFromEuler([0, 0, iyaw], physicsClientId=self.client_id)
                pb.resetBasePositionAndOrientation(self.robot_id, [rx, ry, 0.05], iorn, physicsClientId=self.client_id)
                self._step_substep()
                if self.is_carrying:
                    pb.resetBasePositionAndOrientation(self.resource_id, [rx, ry, 0.15], iorn, physicsClientId=self.client_id)

        else:  # Actions 4, 5, 6 (Pickup, Drop, Stay)
            for _ in range(num_substeps):
                self._step_substep()
                if self.is_carrying:
                    r_pos, r_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
                    pb.resetBasePositionAndOrientation(self.resource_id, [r_pos[0], r_pos[1], 0.15], r_orn, physicsClientId=self.client_id)

        # Action 4: Pick up
        if action == 4 and not self.is_carrying:
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            res_pos, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
            dist = np.linalg.norm(np.array(robot_pos) - np.array(res_pos))
            if dist <= self.cell_size * 1.25:  # 0.25m pickup range
                self.is_carrying = True
                reward += 2.0
                
        # Action 5: Drop off
        if action == 5 and self.is_carrying:
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            rr, rc = self._world_to_grid(robot_pos[0], robot_pos[1])
            dr, dc = self.depot_pos_grid
            
            # Check if exactly on the 1x1 depot cell
            if rr == dr and rc == dc:
                self.is_carrying = False
                reward += 10.0
                terminated = True
                
                # Drop resource down
                r_pos, r_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
                pb.resetBasePositionAndOrientation(self.resource_id, [r_pos[0], r_pos[1], 0.05], r_orn, physicsClientId=self.client_id)

        # Collision detection (Obstacles & Wall boundaries)
        contacts = pb.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client_id)
        for contact in contacts:
            bodyB = contact[2]
            if bodyB in self.obstacle_ids or bodyB in self.wall_ids:
                reward -= 2.0
                terminated = True
                break

        # Potential-Based Reward Shaping (PBRS) - The Attractive Force
        if not terminated and prev_carrying == self.is_carrying:
            curr_robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            dist_curr = np.linalg.norm(np.array(curr_robot_pos[:2]) - np.array(target_pos_world))
            pbrs_reward = (dist_prev - dist_curr) * 1.0
            reward += pbrs_reward

        # APF Repulsive Force (LiDAR proximity)
        lidar_results, _, _ = self._get_lidar_scan(num_rays=180, max_range=2.0)
        repulsive_penalty = 0.0
        for hit in lidar_results:
            hit_uid = hit[0]
            hit_frac = hit[2]
            if hit_uid >= 0 and hit_frac < 1.0:
                hit_dist = hit_frac * 2.0  # max_range is 2.0
                # Apply repulsive force if obstacle is closer than 0.4m
                if 0.01 < hit_dist < 0.4:
                    repulsive_penalty -= 0.02 * (1.0 / hit_dist)
        reward += repulsive_penalty

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_lidar_scan(self, num_rays=180, max_range=2.0):
        """Simulate 360-degree 2D LiDAR using PyBullet rayTestBatch."""
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rx, ry, rz = robot_pos
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]

        ray_froms = []
        ray_tos = []
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        for angle in angles:
            abs_angle = yaw + angle
            ray_froms.append([rx, ry, rz + 0.05])
            ray_tos.append([
                rx + max_range * math.cos(abs_angle),
                ry + max_range * math.sin(abs_angle),
                rz + 0.05
            ])

        results = pb.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.client_id)
        return results, ray_froms, ray_tos

    def _get_obs(self):
        grid = np.zeros((self.obs_size, self.obs_size, 5), dtype=np.float32)
        
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rx, ry, _ = robot_pos
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]

        half_obs = self.obs_size // 2

        # Get LiDAR hits for obstacles / walls
        lidar_results, _, _ = self._get_lidar_scan(num_rays=180, max_range=2.0)
        for hit in lidar_results:
            hit_uid, _, hit_frac, hit_pos, _ = hit
            if hit_uid >= 0 and hit_frac < 1.0:
                hx, hy = hit_pos[0], hit_pos[1]
                # Convert world hit position to egocentric coordinates (forward=dx, left=dy)
                rel_x = hx - rx
                rel_y = hy - ry
                dx = rel_x * math.cos(yaw) + rel_y * math.sin(yaw)
                dy = -rel_x * math.sin(yaw) + rel_y * math.cos(yaw)

                local_r = int(round(half_obs - (dx / self.cell_size)))
                local_c = int(round(half_obs - (dy / self.cell_size)))

                if 0 <= local_r < self.obs_size and 0 <= local_c < self.obs_size:
                    grid[local_r, local_c, 0] = 1.0  # Obstacle channel

        # Populate Egocentric Grid for Resource, Depot, Boundaries, and Self-Agent
        res_pos, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
        dep_pos, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)

        for r_idx in range(self.obs_size):
            for c_idx in range(self.obs_size):
                dx = (half_obs - r_idx) * self.cell_size
                dy = (half_obs - c_idx) * self.cell_size

                # Transform local ego offset to world coordinates
                wx = rx + dx * math.cos(yaw) - dy * math.sin(yaw)
                wy = ry + dx * math.sin(yaw) + dy * math.cos(yaw)

                # Channel 3: Boundary / Void (arena bounds are 4m x 4m -> [-2.0, 2.0])
                if abs(wx) >= 2.0 or abs(wy) >= 2.0:
                    grid[r_idx, c_idx, 3] = 1.0

                # Channel 1: Resource
                if not self.is_carrying:
                    if math.hypot(wx - res_pos[0], wy - res_pos[1]) <= self.cell_size * 0.5:
                        grid[r_idx, c_idx, 1] = 1.0

                # Channel 2: Depot
                if math.hypot(wx - dep_pos[0], wy - dep_pos[1]) <= self.cell_size * 0.7:
                    grid[r_idx, c_idx, 2] = 1.0

        # Channel 4: Self-Agent center & heading indicator
        grid[half_obs, half_obs, 4] = 1.0  # Center cell (robot location)
        if half_obs - 1 >= 0:
            grid[half_obs - 1, half_obs, 4] = 0.5

        return {
            "grid": grid,
            "is_carrying": int(self.is_carrying)
        }

    def _get_info(self):
        if self.robot_id is not None:
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        else:
            robot_pos = (0.0, 0.0, 0.0)
        lidar_results, _, _ = self._get_lidar_scan(num_rays=180, max_range=2.0)
        hit_distances = [hit[2] * 2.0 for hit in lidar_results]
        return {
            "difficulty": self.difficulty_level,
            "robot_pos": robot_pos,
            "lidar_distances": hit_distances
        }

    def render(self):
        pass

    def close(self):
        if hasattr(self, 'client_id'):
            pb.disconnect(physicsClientId=self.client_id)
