import os
import time
from hivemind_env.env import HiveMindSingleAgentEnv
from stable_baselines3 import PPO

def main():
    print("=======================================================")
    print("  HiveMind Visual Demo Player")
    print("=======================================================")
    
    # We will use the V1 final model for the demo
    model_path = "models/ppo_hivemind_v1_final.zip"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find model at {model_path}")
        return
        
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")
    
    # Create the environment with GUI enabled (render_mode="human")
    # We will show Level 4 (the hardest level with obstacles)
    # Note: v1 model was trained with obs_size=15
    env = HiveMindSingleAgentEnv(render_mode="human", difficulty_level=3, obs_size=15)
    
    num_episodes = 3
    print(f"\nStarting {num_episodes} episodes of Level 4...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        reward_sum = 0
        
        print(f"\nEpisode {ep+1}/{num_episodes} started...")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            reward_sum += reward
            step += 1
            
            # Slow it down just slightly so you can watch it easily
            time.sleep(0.01)
            
        print(f"Episode {ep+1} finished! Steps: {step}, Reward: {reward_sum:.2f}")
        time.sleep(1) # Pause for a second before the next map loads

    env.close()
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
