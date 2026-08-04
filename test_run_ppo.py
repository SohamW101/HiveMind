import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
import time
import random
from train_ppo import make_env

def plot_learning_curve(npz_path, save_path="loss_curve.png"):
    """
    Plots the learning curve (eval rewards) from the SB3 evaluations.npz file.
    SB3 does not save a single 'loss curve' by default outside of tensorboard, 
    so the evaluation reward curve is used as the standard proxy for model improvement.
    """
    print(f"Generating learning curve from {npz_path}...")
    try:
        data = np.load(npz_path)
        timesteps = data['timesteps']
        results = data['results']
        
        # results shape is (n_evaluations, n_eval_episodes)
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, mean_rewards, label='Mean Eval Reward', color='b')
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='b', alpha=0.2)
        plt.title('PPO Learning Curve (Evaluation Rewards)')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"-> Learning curve successfully saved to {save_path}")
    except FileNotFoundError:
        print(f"Error: {npz_path} not found. Could not generate plot.")
    except Exception as e:
        print(f"Error plotting learning curve: {e}")

def run_environment(model_path, difficulty=3, episodes=3):
    print(f"\nLoading environment (difficulty={difficulty})...")
    
    # We instantiate the environment manually instead of using make_env
    # to avoid make_env's fixed initial seed resetting the global random state.
    env = gym.make("HiveMind-SingleAgent", render_mode="human", difficulty_level=difficulty)
    from train_ppo import OutcomeInfoWrapper
    env = OutcomeInfoWrapper(env)
    
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for ep in range(episodes):
        # We supply a completely random seed based on current time
        # to ensure random obstacles, depot, resource each episode and across executions.
        ep_seed = int(time.time() * 1000) % 1000000 + ep * 1000
        obs, _ = env.reset(seed=ep_seed)
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Starting Episode {ep+1} (Seed: {ep_seed}) ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = term or trunc
            time.sleep(0.05)  # Slight delay to make the visualization easier to watch

        # Determine outcome
        key = "SUCCESS" if info.get("is_success") else "COLLISION" if info.get("is_collision") else "TIMEOUT"
        print(f"Episode {ep+1}/{episodes}: {key} - Reward: {total_reward:.2f}, Steps: {steps}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO Model and Plot Loss/Learning Curve")
    parser.add_argument("--model", type=str, default="runs/ppo/best_model.zip", help="Path to best_model.zip")
    parser.add_argument("--eval-data", type=str, default="runs/ppo/eval/evaluations.npz", help="Path to evaluations.npz")
    parser.add_argument("--difficulty", type=int, default=3, help="Environment difficulty level")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering environment (useful for testing only the plot)")
    
    args = parser.parse_args()

    # Generate the loss/learning curve
    plot_learning_curve(args.eval_data)
    
    # Run the environment
    if not args.no_render:
        run_environment(args.model, args.difficulty, args.episodes)
    else:
        print("\nSkipping environment rendering due to --no-render flag.")
