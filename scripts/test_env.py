import gymnasium as gym
import hivemind_env

def main():
    print("Initializing HiveMind Single-Agent Environment...")
    env = gym.make("HiveMind-SingleAgent-v0", render_mode="human", difficulty_level=1)
    
    obs, info = env.reset()
    print("Environment reset successfully.")
    print(f"Observation keys: {obs.keys()}")
    
    # Run 10 random steps to verify everything runs without crashing
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | Action: {action} | Reward: {reward} | Terminated: {terminated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
