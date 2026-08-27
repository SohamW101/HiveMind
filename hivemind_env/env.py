import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import random
import math
import time

class HiveMindMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, difficulty_level=1, obs_size=21, show_lidar=None):
        super().__init__()
        self.render_mode = render_mode
        self.num_agents = 4
        self.difficulty_level = difficulty_level
        self.show_lidar = (render_mode == "human") if show_lidar is None else show_lidar
        self.dt = 1.0 / 240.0
        self.grid_size = 13
        self.cell_size = 1.0
        
        # Actions: 0: Forward, 1: Backward, 2: Turn Left, 3: Turn Right, 4: Pick Up, 5: Drop Off, 6: Stay
        self.action_space = spaces.MultiDiscrete([7] * self.num_agents)
        self.obs_size = obs_size

        if self.render_mode == "human":
            self.client_id = pb.connect(pb.GUI)
        else:
            self.client_id = pb.connect(pb.DIRECT)
            
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.robot_ids = []
        self.resource_ids = []
        self.depot_id = None
        self.obstacle_ids = []
        self.depot_pos_grid = (0, 0)
        self.max_steps = 2000
        self.current_step = 0

    def _grid_to_world(self, r, c):
        x = (c - self.grid_size/2.0) * self.cell_size + (self.cell_size/2.0)
        y = (self.grid_size/2.0 - r) * self.cell_size - (self.cell_size/2.0)
        return x, y
        
    def _world_to_grid(self, x, y):
        c = int(round((x / self.cell_size) + self.grid_size/2.0 - 0.5))
        r = int(round(self.grid_size/2.0 - (y / self.cell_size) - 0.5))
        return r, c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.is_carrying = [False] * self.num_agents
        self.carried_resource_ids = [None] * self.num_agents
        self.current_step = 0
        
        pb.resetSimulation(physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        pb.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Depot position (Corner: r=0, c=0)
        self.depot_pos_grid = (0, 0)
        dx, dy = self._grid_to_world(0, 0)
        depot_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.5, self.cell_size*0.5, 0.01], rgbaColor=[1, 0, 0, 0.5], physicsClientId=self.client_id)
        self.depot_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=depot_vis, basePosition=[dx, dy, 0.01], physicsClientId=self.client_id)

        # Spawn 4 bots near the depot
        spawn_cells = [(0, 1), (1, 0), (0, 2), (2, 0)]
        
        self.robot_ids = []
        self.agent_state = []

        import os
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "diff_drive_bot.urdf")
        
        for i in range(self.num_agents):
            rx, ry = self._grid_to_world(spawn_cells[i][0], spawn_cells[i][1])
            rid = pb.loadURDF(urdf_path, basePosition=[rx, ry, 0.1], physicsClientId=self.client_id)
            self.robot_ids.append(rid)
            
            state = {
                'left_wheel_indices': [],
                'right_wheel_indices': [],
                'arm_yaw_joint_idx': None,
                'left_finger_joint_idx': None,
                'right_finger_joint_idx': None,
                'lidar_joint_idx': None,
                'current_arm_yaw': 0.0,
                'current_finger_pos': 0.03,  # Scaled by 2x from original 0.015
                'current_lidar_height': 0.0
            }
            
            for j in range(pb.getNumJoints(rid, physicsClientId=self.client_id)):
                info = pb.getJointInfo(rid, j, physicsClientId=self.client_id)
                jname = info[1].decode("utf-8")
                if "left_wheel" in jname:
                    state['left_wheel_indices'].append(j)
                elif "right_wheel" in jname:
                    state['right_wheel_indices'].append(j)
                elif jname == "arm_yaw_joint":
                    state['arm_yaw_joint_idx'] = j
                elif jname == "left_finger_joint":
                    state['left_finger_joint_idx'] = j
                elif jname == "right_finger_joint":
                    state['right_finger_joint_idx'] = j
                elif jname == "lidar_joint":
                    state['lidar_joint_idx'] = j
                    
            self.agent_state.append(state)
            self._set_arm_and_lidar_joints(i, state['current_arm_yaw'], state['current_finger_pos'], state['current_lidar_height'])

        # 13x13 Warehouse Generation
        self.obstacle_ids = []
        self.resource_ids = []
        
        shelf_urdf_path = os.path.join(os.path.dirname(__file__), "assets", "shelf.urdf")
        shelf_rows = [1, 3, 5, 7, 9, 11]
        for r in shelf_rows:
            partitions = [(1, 3), (5, 3), (9, 3)]
            for (start_c, length) in partitions:
                cx, cy = self._grid_to_world(r, start_c + length/2.0 - 0.5)
                obs_id = pb.loadURDF(shelf_urdf_path, basePosition=[cx, cy, 0.0], useFixedBase=True, physicsClientId=self.client_id)
                self.obstacle_ids.append(obs_id)
            
            for c in [4, 8]:
                resx, resy = self._grid_to_world(r, c)
                res_col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=0.15, height=0.2, physicsClientId=self.client_id)
                res_vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.15, length=0.2, rgbaColor=[0, 1, 0, 1], physicsClientId=self.client_id)
                res_id = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=res_col, baseVisualShapeIndex=res_vis, basePosition=[resx, resy, 0.1], physicsClientId=self.client_id)
                self.resource_ids.append(res_id)

        # Static Boundary Walls around the 13x13 grid
        self.wall_ids = []
        b_size = self.grid_size * self.cell_size / 2.0
        wall_half_extents = [(b_size + 0.1, 0.1, 0.5), (b_size + 0.1, 0.1, 0.5), (0.1, b_size, 0.5), (0.1, b_size, 0.5)]
        wall_positions = [(0, b_size + 0.1, 0.5), (0, -b_size - 0.1, 0.5), (-b_size - 0.1, 0, 0.5), (b_size + 0.1, 0, 0.5)]
        for he, pos in zip(wall_half_extents, wall_positions):
            w_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.client_id)
            w_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.client_id)
            w_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=w_col, baseVisualShapeIndex=w_vis, basePosition=pos, physicsClientId=self.client_id)
            self.wall_ids.append(w_id)
            
        if self.render_mode == "human":
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)

        for _ in range(20):
            pb.stepSimulation(physicsClientId=self.client_id)

        return self._get_obs(), self._get_info()

    def _set_arm_and_lidar_joints(self, agent_idx, arm_yaw, finger_pos, lidar_height):
        rid = self.robot_ids[agent_idx]
        st = self.agent_state[agent_idx]
        if st['arm_yaw_joint_idx'] is not None:
            pb.resetJointState(rid, st['arm_yaw_joint_idx'], arm_yaw, physicsClientId=self.client_id)
        if st['left_finger_joint_idx'] is not None:
            pb.resetJointState(rid, st['left_finger_joint_idx'], finger_pos, physicsClientId=self.client_id)
        if st['right_finger_joint_idx'] is not None:
            pb.resetJointState(rid, st['right_finger_joint_idx'], -finger_pos, physicsClientId=self.client_id)
        if st['lidar_joint_idx'] is not None:
            pb.resetJointState(rid, st['lidar_joint_idx'], lidar_height, physicsClientId=self.client_id)

    def _get_cardinal_direction_angle(self, target_world_pos, robot_world_pos, robot_yaw):
        dx = target_world_pos[0] - robot_world_pos[0]
        dy = target_world_pos[1] - robot_world_pos[1]
        target_angle = math.atan2(dy, dx)
        rel_angle = target_angle - robot_yaw
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
        cardinal_step = round(rel_angle / (math.pi / 2.0))
        cardinal_angle = cardinal_step * (math.pi / 2.0)
        if cardinal_angle == math.pi or cardinal_angle == -math.pi:
            cardinal_angle = math.pi
        return cardinal_angle

    def step(self, actions):
        self.current_step += 1
        num_substeps = 30

        # Pre-compute trajectories
        starts = []
        targets = []
        
        for i in range(self.num_agents):
            rid = self.robot_ids[i]
            pos, orn = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
            yaw = pb.getEulerFromQuaternion(orn)[2]
            
            # Snap yaw to exact cardinal direction to prevent drift
            yaw = round(yaw / (math.pi / 2.0)) * (math.pi / 2.0)
            
            # Snap position to exact grid cell center to prevent drift
            r, c = self._world_to_grid(pos[0], pos[1])
            gx, gy = self._grid_to_world(r, c)
            pos = (gx, gy, pos[2])
            
            action = actions[i]
            
            st = self.agent_state[i]
            start_state = {
                'pos': pos, 'yaw': yaw,
                'arm_yaw': st['current_arm_yaw'],
                'finger': st['current_finger_pos'],
                'lidar': st['current_lidar_height'],
                'wheel_delta': 0.0
            }
            target_state = start_state.copy()
            
            if action == 0:  # Forward
                target_state['pos'] = (pos[0] + self.cell_size * math.cos(yaw), pos[1] + self.cell_size * math.sin(yaw), pos[2])
                target_state['wheel_delta'] = 0.119
            elif action == 1:  # Backward
                target_state['pos'] = (pos[0] - self.cell_size * math.cos(yaw), pos[1] - self.cell_size * math.sin(yaw), pos[2])
                target_state['wheel_delta'] = -0.119
            elif action == 2:  # Turn Left
                target_state['yaw'] = yaw + (math.pi / 2.0)
                target_state['wheel_delta'] = 0.05
            elif action == 3:  # Turn Right
                target_state['yaw'] = yaw - (math.pi / 2.0)
                target_state['wheel_delta'] = -0.05
            elif action == 4 and not self.is_carrying[i]:  # Pick Up
                nearest_res = None
                min_dist = float('inf')
                for res_id in self.resource_ids:
                    res_pos, _ = pb.getBasePositionAndOrientation(res_id, physicsClientId=self.client_id)
                    dist = math.hypot(res_pos[0] - pos[0], res_pos[1] - pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_res = res_id
                
                if nearest_res is not None and min_dist <= self.cell_size * 1.5:
                    res_pos, _ = pb.getBasePositionAndOrientation(nearest_res, physicsClientId=self.client_id)
                    target_state['arm_yaw'] = self._get_cardinal_direction_angle(res_pos, pos, yaw)
                    target_state['finger'] = -0.01
                    target_state['lidar'] = 0.1
                    self.is_carrying[i] = True
                    self.carried_resource_ids[i] = nearest_res
                    self.resource_ids.remove(nearest_res)
                    target_state['res_start_pos'] = res_pos
                    target_state['pickup_target'] = nearest_res
            elif action == 5 and self.is_carrying[i]:  # Drop Off
                dep_pos, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)
                dist = math.hypot(pos[0] - dep_pos[0], pos[1] - dep_pos[1])
                if dist <= self.cell_size * 1.5:
                    target_state['arm_yaw'] = self._get_cardinal_direction_angle(dep_pos, pos, yaw)
                    target_state['finger'] = 0.03
                    target_state['lidar'] = 0.0
                    target_state['dropoff'] = True
                    target_state['drop_target'] = dep_pos

            starts.append(start_state)
            targets.append(target_state)

        # Simultaneous Execution
        for step_idx in range(1, num_substeps + 1):
            alpha = step_idx / float(num_substeps)
            for i in range(self.num_agents):
                st = self.agent_state[i]
                rid = self.robot_ids[i]
                s = starts[i]
                t = targets[i]
                
                # Interpolate Pos & Yaw
                ix = s['pos'][0] + (t['pos'][0] - s['pos'][0]) * alpha
                iy = s['pos'][1] + (t['pos'][1] - s['pos'][1]) * alpha
                iyaw = s['yaw'] + (t['yaw'] - s['yaw']) * alpha
                iorn = pb.getQuaternionFromEuler([0, 0, iyaw], physicsClientId=self.client_id)
                pb.resetBasePositionAndOrientation(rid, [ix, iy, s['pos'][2]], iorn, physicsClientId=self.client_id)
                
                # Interpolate Joints
                st['current_arm_yaw'] = s['arm_yaw'] + (t['arm_yaw'] - s['arm_yaw']) * alpha
                st['current_finger_pos'] = s['finger'] + (t['finger'] - s['finger']) * alpha
                st['current_lidar_height'] = s['lidar'] + (t['lidar'] - s['lidar']) * alpha
                self._set_arm_and_lidar_joints(i, st['current_arm_yaw'], st['current_finger_pos'], st['current_lidar_height'])
                
                # Wheels
                if actions[i] in [0, 1]:
                    wd = t['wheel_delta']
                    for idx in st['left_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                    for idx in st['right_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                elif actions[i] == 2: # Turn Left
                    wd = t['wheel_delta']
                    for idx in st['left_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos - wd, physicsClientId=self.client_id)
                    for idx in st['right_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                elif actions[i] == 3: # Turn Right
                    wd = t['wheel_delta']
                    for idx in st['left_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                    for idx in st['right_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos - wd, physicsClientId=self.client_id)

                # Resource interpolation
                arm_world_angle = iyaw + st['current_arm_yaw']
                carried_rx = ix + 0.3 * math.cos(arm_world_angle)
                carried_ry = iy + 0.3 * math.sin(arm_world_angle)
                
                if actions[i] == 4 and 'pickup_target' in t: # Picking up
                    res_id = t['pickup_target']
                    start_res_pos = t['res_start_pos']
                    cur_res_x = start_res_pos[0] + alpha * (carried_rx - start_res_pos[0])
                    cur_res_y = start_res_pos[1] + alpha * (carried_ry - start_res_pos[1])
                    pb.resetBasePositionAndOrientation(res_id, [cur_res_x, cur_res_y, 0.1], iorn, physicsClientId=self.client_id)
                elif actions[i] == 5 and 'dropoff' in t: # Dropping off
                    res_id = self.carried_resource_ids[i]
                    if res_id:
                        dep_pos = t['drop_target']
                        cur_res_x = carried_rx + alpha * (dep_pos[0] - carried_rx)
                        cur_res_y = carried_ry + alpha * (dep_pos[1] - carried_ry)
                        pb.resetBasePositionAndOrientation(res_id, [cur_res_x, cur_res_y, 0.1], iorn, physicsClientId=self.client_id)
                elif self.is_carrying[i] and self.carried_resource_ids[i]: # Carrying
                    res_id = self.carried_resource_ids[i]
                    pb.resetBasePositionAndOrientation(res_id, [carried_rx, carried_ry, 0.1], iorn, physicsClientId=self.client_id)

            pb.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(0.01)

        # Post-substep handling
        for i in range(self.num_agents):
            if actions[i] == 4 and 'pickup_target' in targets[i]:
                self.agent_state[i]['current_arm_yaw'] = 0.0
                self._set_arm_and_lidar_joints(i, 0.0, -0.01, 0.1)
                
            elif actions[i] == 5 and 'dropoff' in targets[i]:
                res_id = self.carried_resource_ids[i]
                if res_id:
                    pb.removeBody(res_id, physicsClientId=self.client_id)
                self.is_carrying[i] = False
                self.carried_resource_ids[i] = None
                self.agent_state[i]['current_arm_yaw'] = targets[i]['arm_yaw']

        return self._get_obs(), [0]*self.num_agents, False, False, self._get_info()

    def _get_obs(self):
        return []

    def _get_info(self):
        poses = []
        for rid in self.robot_ids:
            p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
            poses.append(p)
        return {
            "robot_pos": poses,
            "remaining_resources": len(self.resource_ids)
        }

    def render(self):
        pass

    def close(self):
        if hasattr(self, 'client_id'):
            pb.disconnect(physicsClientId=self.client_id)
