import os
import sys
import json
import time

# Add parent directory to sys.path to import hivemind_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hivemind_env.env import HiveMindSingleAgentEnv

def main():
    print("=======================================================")
    print("  HiveMind Visual Demo Player")
    print("=======================================================")
    
    model_path = "models/ppo_hivemind_v1_final.zip"
    print(f"Loading model: {model_path}")

    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "best_episodes.json")

    if not os.path.exists(json_path):
        print(f"ERROR: Could not find saved episodes file at: {json_path}")
        print("Please run python demo/gen_demo.py first to generate demo episodes!")
        return

    with open(json_path, "r") as f:
        saved_episodes = json.load(f)

    num_episodes = len(saved_episodes)
    difficulty = saved_episodes[0].get("difficulty_level", 4) if num_episodes > 0 else 4

    # Create PyBullet GUI environment with LiDAR visualization enabled
    env = HiveMindSingleAgentEnv(render_mode="human", difficulty_level=difficulty, obs_size=15, show_lidar=True)

    print(f"\nStarting {num_episodes} episodes of Level {difficulty}...")

    for ep_idx, ep_data in enumerate(saved_episodes):
        seed = ep_data["seed"]
        actions = ep_data["actions"]

        print(f"\nEpisode {ep_idx + 1}/{num_episodes} started...")

        # Deterministic reset using the saved seed
        obs, info = env.reset(seed=seed)
        done = False
        step = 0
        reward_sum = 0.0

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            step += 1
            done = terminated or truncated

            # Visual delay for smooth 60 FPS GUI playback
            time.sleep(0.01)

            if done:
                break

        print(f"Episode {ep_idx + 1} finished! Steps: {step}, Reward: {reward_sum:.2f}")
        time.sleep(1.0)  # Pause for 1 second before loading next episode

    env.close()
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
