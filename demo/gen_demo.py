import os
import sys
import json
import random
import numpy as np
from stable_baselines3 import PPO

# Add parent directory to sys.path to import hivemind_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hivemind_env.env import HiveMindSingleAgentEnv

def generate_demos(model_path="models/ppo_hivemind_v1_final.zip", num_episodes=100, difficulty_level=4, top_k=5):
    print("=========================================================")
    print("  HIVEMIND HEADLESS DEMO GENERATOR (100 EPISODES)  ")
    print("=========================================================")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find trained model at {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    # Create headless environment for fast evaluation
    env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=difficulty_level, obs_size=15)

    episodes_data = []
    successes = 0
    total_rewards = []
    total_steps_list = []

    print(f"\nRunning {num_episodes} headless simulation episodes (Level {difficulty_level})...")

    for ep_idx in range(num_episodes):
        # Generate a unique seed for deterministic map generation
        ep_seed = random.randint(10000, 999999)
        obs, info = env.reset(seed=ep_seed)

        done = False
        step = 0
        reward_sum = 0.0
        actions_taken = []
        pickup_success = False
        dropoff_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            actions_taken.append(action_int)

            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            reward_sum += reward
            step += 1

            if env.is_carrying:
                pickup_success = True
            if terminated and reward > 5.0:
                dropoff_success = True

        is_success = pickup_success and dropoff_success
        if is_success:
            successes += 1

        total_rewards.append(reward_sum)
        total_steps_list.append(step)

        episodes_data.append({
            "episode_index": ep_idx + 1,
            "seed": ep_seed,
            "difficulty_level": difficulty_level,
            "total_reward": round(reward_sum, 2),
            "total_steps": step,
            "success": is_success,
            "pickup": pickup_success,
            "dropoff": dropoff_success,
            "actions": actions_taken
        })

        if (ep_idx + 1) % 10 == 0 or ep_idx == num_episodes - 1:
            print(f"  Processed {ep_idx + 1}/{num_episodes} episodes | Current Success Rate: {successes}/{ep_idx + 1} ({successes/(ep_idx+1)*100:.1f}%)")

    env.close()

    # Calculate overall metrics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(total_steps_list)
    success_rate = (successes / num_episodes) * 100.0

    print("\n=========================================================")
    print("  HEADLESS EVALUATION METRICS SUMMARY  ")
    print("=========================================================")
    print(f"  Total Episodes Tested : {num_episodes}")
    print(f"  Success Rate          : {successes}/{num_episodes} ({success_rate:.1f}%)")
    print(f"  Average Reward        : {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Average Steps Taken   : {avg_steps:.1f}")
    print(f"  Reward Range          : [{min(total_rewards):.2f}, {max(total_rewards):.2f}]")
    print("=========================================================")

    # Sort episodes by total_reward descending (successful ones first)
    successful_episodes = [ep for ep in episodes_data if ep["success"]]
    # If not enough successful episodes, fall back to highest reward overall
    pool = successful_episodes if len(successful_episodes) >= top_k else episodes_data
    sorted_episodes = sorted(pool, key=lambda x: (x["success"], x["total_reward"], -x["total_steps"]), reverse=True)

    best_episodes = sorted_episodes[:top_k]

    # Assign ranks
    for rank, ep in enumerate(best_episodes, start=1):
        ep["rank"] = rank

    output_dir = os.path.dirname(__file__)
    save_path = os.path.join(output_dir, "best_episodes.json")

    with open(save_path, "w") as f:
        json.dump(best_episodes, f, indent=2)

    print(f"\n[SUCCESS] Top {top_k} best episodes saved to: {save_path}")
    print("\nTop Episodes Overview:")
    for ep in best_episodes:
        print(f"  Rank #{ep['rank']} | Seed: {ep['seed']} | Reward: {ep['total_reward']} | Steps: {ep['total_steps']} | Success: {ep['success']}")

if __name__ == "__main__":
    generate_demos()
