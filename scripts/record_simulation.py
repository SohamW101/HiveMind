import os
import time
import math
import numpy as np
import pybullet as pb
import gymnasium as gym
from PIL import Image
import hivemind_env
from test_run import plan_bfs_actions, build_obstacle_grid, get_heading_idx

def record_simulation():
    print("=========================================================")
    print(" Recording 4-Wheel Rectangular Robot Task Simulation   ")
    print("=========================================================")

    # Initialize headless environment
    env = gym.make("HiveMind-SingleAgent", render_mode=None, difficulty_level=3)
    obs, info = env.reset(seed=101)
    unwrapped = env.unwrapped

    # Artifacts output directory
    output_dir = "/home/taksh/HiveMind/artifacts"
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    
    def capture_frame(pitch=-35, yaw=45, dist=3.2, target=[0, 0, 0]):
        width, height = 640, 480
        view_mat = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=dist,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=unwrapped.client_id
        )
        proj_mat = pb.computeProjectionMatrixFOV(
            fov=55,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=20.0,
            physicsClientId=unwrapped.client_id
        )
        _, _, rgb, _, _ = pb.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=unwrapped.client_id
        )
        img = Image.fromarray(rgb[:, :, :3])
        return img

    # Initial frame
    frames.append(capture_frame())

    # Get positions
    robot_pos, robot_orn = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
    res_pos, _ = pb.getBasePositionAndOrientation(unwrapped.resource_id, physicsClientId=unwrapped.client_id)
    dep_pos, _ = pb.getBasePositionAndOrientation(unwrapped.depot_id, physicsClientId=unwrapped.client_id)

    r_r, r_c = unwrapped._world_to_grid(robot_pos[0], robot_pos[1])
    res_r, res_c = unwrapped._world_to_grid(res_pos[0], res_pos[1])
    dep_r, dep_c = unwrapped.depot_pos_grid
    
    r_h = get_heading_idx(pb.getEulerFromQuaternion(robot_orn)[2])
    obstacle_grid = build_obstacle_grid(unwrapped)

    print(f"-> Robot Start : ({r_r}, {r_c}) | Resource: ({res_r}, {res_c}) | Depot: ({dep_r}, {dep_c})")

    # 1. Path to Resource
    actions_to_res = plan_bfs_actions(r_r, r_c, r_h, res_r, res_c, obstacle_grid)
    for act in actions_to_res:
        curr_r_pos, _ = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
        res_curr_pos, _ = pb.getBasePositionAndOrientation(unwrapped.resource_id, physicsClientId=unwrapped.client_id)
        if np.linalg.norm(np.array(curr_r_pos[:2]) - np.array(res_curr_pos[:2])) <= 0.25:
            break
        obs, reward, term, trunc, info = env.step(act)
        frames.append(capture_frame(target=[curr_r_pos[0], curr_r_pos[1], 0.1]))

    # Capture Pickup snapshot
    frames.append(capture_frame())
    print("-> Pickup Resource (Action 4)...")
    obs, reward, term, trunc, info = env.step(4)
    for _ in range(3):
        frames.append(capture_frame())

    # Save pickup frame as PNG
    pickup_img_path = os.path.join(output_dir, "simulation_pickup.png")
    frames[-1].save(pickup_img_path)

    # 2. Path to Depot
    curr_r_pos, curr_orn = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
    curr_r, curr_c = unwrapped._world_to_grid(curr_r_pos[0], curr_r_pos[1])
    curr_h = get_heading_idx(pb.getEulerFromQuaternion(curr_orn)[2])
    
    actions_to_depot = plan_bfs_actions(curr_r, curr_c, curr_h, dep_r, dep_c, obstacle_grid)
    for act in actions_to_depot:
        obs, reward, term, trunc, info = env.step(act)
        curr_p, _ = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
        frames.append(capture_frame(target=[curr_p[0], curr_p[1], 0.1]))

    # 3. Dropoff
    print("-> Dropoff Resource (Action 5)...")
    obs, reward, term, trunc, info = env.step(5)
    for _ in range(5):
        frames.append(capture_frame())

    # Save dropoff frame as PNG
    dropoff_img_path = os.path.join(output_dir, "simulation_delivery.png")
    frames[-1].save(dropoff_img_path)

    # Save Animated GIF and WebP
    gif_path = os.path.join(output_dir, "robot_simulation.gif")
    webp_path = os.path.join(output_dir, "robot_simulation.webp")

    # Downsample frame rate for sleek animation size
    anim_frames = frames[::2]
    anim_frames[0].save(gif_path, save_all=True, append_images=anim_frames[1:], duration=120, loop=0)
    anim_frames[0].save(webp_path, save_all=True, append_images=anim_frames[1:], duration=120, loop=0)

    print(f"-> Saved Animated GIF : {gif_path}")
    print(f"-> Saved Animated WebP: {webp_path}")
    print(f"-> Saved Pickup Image : {pickup_img_path}")
    print(f"-> Saved Delivery Image: {dropoff_img_path}")
    env.close()

if __name__ == "__main__":
    record_simulation()
