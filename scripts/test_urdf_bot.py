import time
import os
import pybullet as pb
import gymnasium as gym
import hivemind_env

def test_urdf_robot():
    print("=========================================================")
    print("  Testing 4-Wheel Rectangular Robot URDF Model Loading  ")
    print("=========================================================")
    
    env = gym.make("HiveMind-SingleAgent", render_mode="human", difficulty_level=1)
    obs, info = env.reset()
    unwrapped = env.unwrapped
    
    robot_id = unwrapped.robot_id
    num_joints = pb.getNumJoints(robot_id, physicsClientId=unwrapped.client_id)
    
    print(f"-> Robot Body ID          : {robot_id}")
    print(f"-> URDF Wheel Detection   : {'SUCCESS' if unwrapped.has_urdf_wheels else 'FALLBACK BOX'}")
    print(f"-> Total Joints in URDF   : {num_joints}")
    print(f"-> Left Wheel Indices     : {unwrapped.left_wheel_indices}")
    print(f"-> Right Wheel Indices    : {unwrapped.right_wheel_indices}")
    
    for i in range(num_joints):
        j_info = pb.getJointInfo(robot_id, i, physicsClientId=unwrapped.client_id)
        j_name = j_info[1].decode("utf-8")
        j_type = j_info[2]
        print(f"   Joint [{i}]: {j_name} (type: {j_type})")
        
    print("\nExecuting test actions to verify 4-wheel movement...")
    # Action 0: Move Forward
    print("-> Action 0: Move Forward")
    for _ in range(5):
        env.step(0)
        
    # Action 2: Turn Left
    print("-> Action 2: Turn Left")
    for _ in range(5):
        env.step(2)
        
    # Action 3: Turn Right
    print("-> Action 3: Turn Right")
    for _ in range(5):
        env.step(3)

    # Action 1: Move Backward
    print("-> Action 1: Move Backward")
    for _ in range(5):
        env.step(1)

    print("\n4-Wheel URDF test completed successfully! Closing environment...")
    time.sleep(1.0)
    env.close()

if __name__ == "__main__":
    test_urdf_robot()
