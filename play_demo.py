import os
import time

import pybullet as pb

from hivemind_env.env import (
    OBS_SIZE_V1,
    HiveMindSingleAgentEnv,
    PhysicsDisconnectedError,
)
from hivemind_env.training import load_policy

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
    model, _ = load_policy(model_path, device="cpu")
    
    # Create the environment with GUI enabled (render_mode="human").
    # Level 3 is "random spawns + obstacles". Levels 3 and 4 currently generate from the
    # same distribution, so this is the hardest map the env produces either way.
    # obs_size must be 15 - that is what the v1 model was trained on.
    difficulty = 3
    env = HiveMindSingleAgentEnv(
        render_mode="human", difficulty_level=difficulty, obs_size=OBS_SIZE_V1
    )

    num_episodes = 3
    print(f"\nStarting {num_episodes} episodes of Level {difficulty}...")
    
    completed = 0
    try:
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

            completed += 1
            print(f"Episode {ep+1} finished! Steps: {step}, Reward: {reward_sum:.2f}")
            time.sleep(1)  # Pause for a second before the next map loads

    except (PhysicsDisconnectedError, pb.error) as exc:
        # Closing the viewer mid-episode kills the physics server, and every subsequent
        # PyBullet call raises. That is a normal way to stop the demo, not a crash -
        # report it as such instead of unwinding a traceback from inside the substep loop.
        print(f"\nViewer closed during episode {completed + 1} - stopping early.")
        print(f"  ({exc})")
        print(f"Completed {completed}/{num_episodes} episode(s) before the window went away.")
        return

    env.close()
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
