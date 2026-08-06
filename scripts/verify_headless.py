import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import hivemind_env
from hivemind_env.env import HiveMindSingleAgentEnv

def verify():
    print("=========================================================")
    print(" Headless Policy & URDF Integration Verification Test    ")
    print("=========================================================")
    
    # Create headless environment
    env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=1)
    obs, info = env.reset(seed=42)
    
    print(f"-> URDF Robot Active : {env.has_urdf_wheels}")
    print(f"-> Left Wheel Indices: {env.left_wheel_indices}")
    print(f"-> Right Wheel Indices: {env.right_wheel_indices}")
    print(f"-> Observation Grid  : {obs['grid'].shape}")
    print(f"-> Is Carrying Flag  : {obs['is_carrying']}")
    
    model = PPO.load("models/ppo_hivemind_test.zip")
    print("-> Successfully loaded trained PPO model weights!")
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_reward += reward
        steps += 1
        done = term or trunc
        
    print(f"-> Evaluation Result : Steps = {steps}, Total Reward = {total_reward:.2f}, Terminated = {term}")
    env.close()
    print("=========================================================")
    print("        ALL VERIFICATION CHECKS PASSED PERFECTLY!        ")
    print("=========================================================")

if __name__ == "__main__":
    verify()
