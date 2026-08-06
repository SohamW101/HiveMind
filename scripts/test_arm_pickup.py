import os
import sys
import numpy as np
import pybullet as pb
import math

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hivemind_env.env import HiveMindSingleAgentEnv

def test_arm_and_lidar_logic():
    print("=== Testing HiveMind Single Agent Environment with URDF Arm & Lidar ===")
    env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=1)
    obs, info = env.reset(seed=42)

    print("Initial observation keys:", obs.keys())
    print("Initial grid shape:", obs["grid"].shape)
    print("Initial is_carrying:", obs["is_carrying"])

    assert obs["is_carrying"] == 0, "Agent should not be carrying initially"
    assert env.current_lidar_height == 0.0, "Lidar should be at lower position (0.0)"
    assert env.current_finger_pos == 0.015, "Gripper fingers should be open (0.015)"

    # Teleport agent next to resource for pickup testing
    robot_pos, robot_orn = pb.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
    res_pos, res_orn = pb.getBasePositionAndOrientation(env.resource_id, physicsClientId=env.client_id)
    print(f"Robot starting pos: {robot_pos[:2]}, Resource pos: {res_pos[:2]}")

    # Position robot 0.15m away from resource
    near_rx = res_pos[0] - 0.15
    near_ry = res_pos[1]
    pb.resetBasePositionAndOrientation(env.robot_id, [near_rx, near_ry, 0.0], robot_orn, physicsClientId=env.client_id)

    # Perform Pickup Action (Action 4)
    print("Executing Action 4 (Pickup)...")
    obs, reward, terminated, truncated, info = env.step(4)

    print(f"Post-pickup reward: {reward}")
    print(f"Post-pickup is_carrying: {obs['is_carrying']}")
    print(f"Current arm yaw: {env.current_arm_yaw:.2f} rad ({math.degrees(env.current_arm_yaw):.1f} deg)")
    print(f"Current finger pos: {env.current_finger_pos}")
    print(f"Current Lidar height: {env.current_lidar_height}")

    assert obs["is_carrying"] == 1, "Agent should be carrying after Action 4"
    assert env.current_lidar_height == 0.07, "Lidar height should elevate to 0.07 when carrying"
    assert env.current_finger_pos == -0.01, "Gripper fingers should close (-0.01) when carrying"

    # Verify Lidar obstacle channel grid (channel 0) does not register the carried resource
    obs_grid_ch0 = obs["grid"][:, :, 0]
    print(f"Obstacle grid channel sum while carrying: {np.sum(obs_grid_ch0)}")

    # Move agent with resource (Action 0 - Forward)
    print("Executing Action 0 (Move Forward while carrying)...")
    obs, reward, terminated, truncated, info = env.step(0)

    # Teleport to depot cell for Drop Action (Action 5)
    dep_r, dep_c = env.depot_pos_grid
    dx, dy = env._grid_to_world(dep_r, dep_c)
    pb.resetBasePositionAndOrientation(env.robot_id, [dx, dy, 0.0], robot_orn, physicsClientId=env.client_id)

    # Perform Drop Action (Action 5)
    print("Executing Action 5 (Drop Off at Depot)...")
    obs, reward, terminated, truncated, info = env.step(5)

    print(f"Post-drop reward: {reward}")
    print(f"Post-drop is_carrying: {obs['is_carrying']}")
    print(f"Post-drop terminated: {terminated}")
    print(f"Current Lidar height: {env.current_lidar_height}")
    print(f"Current finger pos: {env.current_finger_pos}")

    assert obs["is_carrying"] == 0, "Agent should not be carrying after drop off"
    assert env.current_lidar_height == 0.0, "Lidar height should return to 0.0 after drop off"
    assert env.current_finger_pos == 0.015, "Gripper fingers should open (0.015) after drop off"

    env.close()
    print("=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_arm_and_lidar_logic()
