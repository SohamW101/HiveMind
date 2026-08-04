import argparse
import gymnasium as gym
from stable_baselines3 import PPO

# Import exactly what train.py used to ensure compatibility with the trained weights
from hivemind_env.env import HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor

def run_demo(model_path, difficulty, episodes):
    print(f"Loading model from: {model_path}")
    print(f"Setting environment difficulty to: {difficulty}")
    
    try:
        # Load the model. stable-baselines3 will look for CustomCombinedExtractor in the namespace.
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}.")
        print("Please check the path. According to train.py, it should be at 'models/ppo_hivemind_test.zip'.")
        return
        
    # Create the environment using the exact class used in train.py, but with human rendering
    env = HiveMindSingleAgentEnv(render_mode="human", difficulty_level=difficulty)
    
    success_count = 0
    total_rewards = []
    
    for ep in range(episodes):
        # We don't pass a seed to ensure a truly random, procedurally generated layout each time
        obs, _ = env.reset(seed=None)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # deterministic=True is standard for evaluating/testing reinforcement learning models
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        total_rewards.append(total_reward)
        
        # In train.py's curriculum callback, reward >= 5.0 was counted as a success
        success = total_reward >= 5.0
        if success:
            success_count += 1
            
        status = "SUCCESS" if success else "FAILED/COLLISION"
        print(f"Episode {ep+1:02d} | Status: {status:<16} | Reward: {total_reward:7.2f} | Steps: {steps}")

    env.close()
    
    print("\n" + "="*40)
    print("DEMO VERIFICATION REPORT")
    print("="*40)
    print(f"Difficulty Level: {difficulty}")
    print(f"Total Episodes  : {episodes}")
    print(f"Success Rate    : {(success_count/episodes)*100:.1f}%")
    print(f"Average Reward  : {sum(total_rewards)/episodes:.2f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Verify the PPO model trained via train.py")
    parser.add_argument("--model", type=str, default="models/ppo_hivemind_test.zip", 
                        help="Path to the trained model .zip file")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Complexity/Difficulty level to test (1 to 4)")
    parser.add_argument("--episodes", type=int, default=5, 
                        help="Number of test episodes")
    args = parser.parse_args()
    
    run_demo(args.model, args.difficulty, args.episodes)
