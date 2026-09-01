import os
import glob
import time
import pybullet as pb
from stable_baselines3 import PPO

from hivemind_env.env import HiveMindMultiAgentEnv
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS

def main():
    model_path = "models/ppo_shared_20260830_165941_final.zip"
    print(f"Loading model weights from: {model_path}")
    
    # Load model with the correct custom objects for inference
    model = PPO.load(model_path, custom_objects=INFERENCE_CUSTOM_OBJECTS, device="cpu")
    
    print("Initializing environment...")
    env = HiveMindMultiAgentEnv(render_mode="human")
    
    try:
        obs, info = env.reset()
        
        # Position the camera for a top-down view similar to play_multi.py
        pb.resetDebugVisualizerCamera(
            cameraDistance=16.0, 
            cameraYaw=0,
            cameraPitch=-89.9, 
            cameraTargetPosition=[0, 0, 0]
        )
        
        done = False
        steps = 0
        print("Starting test run...")
        
        while not done:
            # obs is a list/array of 4 observations (one for each robot)
            # By passing it directly, PPO interprets it as a batch of 4 and returns 4 actions.
            actions, _ = model.predict(obs, deterministic=True)
            
            # Step the environment with the predicted joint actions
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            steps += 1
            # Add a small delay so it's not too fast to watch
            time.sleep(0.03) 
            
        print(f"\nEpisode finished after {steps} steps.")
        print(f"Success: {info['is_success']}")
        print(f"All delivered: {info['all_delivered']}")
        print(f"Delivered count: {info['delivered']}")
            
    except KeyboardInterrupt:
        print("\nTest run stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
