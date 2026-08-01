import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import random
from collections import deque
import math

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
        wall_half_extents = [(2.2, 0.1, 0.2), (2.2, 0.1, 0.2), (0.1, 2.0, 0.2), (0.1, 2.0, 0.2)]
        wall_positions = [(0, 2.1, 0.1), (0, -2.1, 0.1), (-2.1, 0, 0.1), (2.1, 0, 0.1)]
        for he, pos in zip(wall_half_extents, wall_positions):
            w_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.client_id)
            w_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.client_id)
            pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=w_col, baseVisualShapeIndex=w_vis, basePosition=pos, physicsClientId=self.client_id)
            
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

    def step(self, action):
        self.current_step += 1
        reward = -0.01  # Time penalty
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]
        
        lin_vel = 0.0
        ang_vel = 0.0
        
        if action == 0:
            lin_vel = 0.5
        elif action == 1:
            lin_vel = -0.5
        elif action == 2:
            ang_vel = 2.0
        elif action == 3:
            ang_vel = -2.0
            
        vx = lin_vel * math.cos(yaw)
        vy = lin_vel * math.sin(yaw)
        
        pb.resetBaseVelocity(self.robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, ang_vel], physicsClientId=self.client_id)
        
        # Step simulation for a few frames to make movement noticeable per step
        for _ in range(20):
            pb.stepSimulation(physicsClientId=self.client_id)
            # If carrying, teleport resource to robot
            if self.is_carrying:
                r_pos, r_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
                # carry above robot
                pb.resetBasePositionAndOrientation(self.resource_id, [r_pos[0], r_pos[1], r_pos[2]+0.1], r_orn, physicsClientId=self.client_id)
        
        # Action 4: Pick up
        if action == 4 and not self.is_carrying:
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            res_pos, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
            dist = np.linalg.norm(np.array(robot_pos) - np.array(res_pos))
            if dist <= self.cell_size * 1.0: # Highly constrained pickup range
                self.is_carrying = True
                reward += 1.0
                
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

        # Stop velocity after step
        pb.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0], physicsClientId=self.client_id)
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        grid = np.zeros((self.obs_size, self.obs_size, 5), dtype=np.float32)
        
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rr, rc = self._world_to_grid(robot_pos[0], robot_pos[1])
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]
        
        # We will keep this simple for now: fill the local grid around the robot
        # Channel 0: obstacles, Channel 1: resource, Channel 2: depot, Channel 3: boundaries, Channel 4: heading
        half_obs = self.obs_size // 2
        
        for r_offset in range(-half_obs, half_obs + 1):
            for c_offset in range(-half_obs, half_obs + 1):
                r = rr + r_offset
                c = rc + c_offset
                local_r = r_offset + half_obs
                local_c = c_offset + half_obs
                
                # Boundary
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                    grid[local_r, local_c, 3] = 1.0
                    continue
                    
                # We can refine obstacles/resources based on world positions if needed.
                # For now, just a placeholder. 
                # (In a real scenario, you query AABBs from PyBullet)
                
        return {
            "grid": grid,
            "is_carrying": int(self.is_carrying)
        }

    def _get_info(self):
        if self.robot_id is not None:
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        else:
            robot_pos = (0.0, 0.0, 0.0)
        return {
            "difficulty": self.difficulty_level,
            "robot_pos": robot_pos
        }

    def render(self):
        pass

    def close(self):
        if hasattr(self, 'client_id'):
            pb.disconnect(physicsClientId=self.client_id)
