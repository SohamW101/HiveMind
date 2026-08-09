import argparse
import os

import numpy as np

# Import exactly what train.py used to ensure compatibility with the trained weights
from hivemind_env.env import OBS_SIZE_V1, OBS_SIZE_V2, HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor  # noqa: F401  (SB3 unpickles it by name)
from hivemind_env.training import load_policy


def run_demo(model_path, difficulty, episodes, obs_size):
    print(f"Loading model from: {model_path}")
    print(f"Setting environment difficulty to: {difficulty}")
    print(f"Observation window: {obs_size}x{obs_size}x5")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    try:
        # SB3 raises ValueError (not FileNotFoundError) for an unreadable archive.
        # load_policy() picks PPO vs RecurrentPPO from the filename and neutralises the
        # cross-Python-version pickled schedules.
        model, is_recurrent = load_policy(model_path, device="cpu")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: could not load {model_path}: {exc}")
        return

    env = HiveMindSingleAgentEnv(
        render_mode="human", difficulty_level=difficulty, obs_size=obs_size
    )

    success_count = 0
    total_rewards = []

    for ep in range(episodes):
        # No seed, so each episode gets a freshly generated layout
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # RecurrentPPO threads LSTM state through predict(); PPO ignores these.
        lstm_states = None
        episode_start = True

        while not done:
            # deterministic=True is standard for evaluating/testing RL models
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=np.array([episode_start]),
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            episode_start = False

            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(total_reward)

        # In the curriculum callback, reward >= 5.0 on the final step counts as a
        # delivery. Episode totals are dominated by distance shaping, so classify on
        # the terminal event instead.
        success = terminated and reward >= 5.0
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
    parser = argparse.ArgumentParser(description="Test and Verify a model trained via train.py")
    parser.add_argument("--model", type=str, default="models/ppo_hivemind_v1_final.zip",
                        help="Path to the trained model .zip file")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Complexity/Difficulty level to test (1 to 4)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    parser.add_argument("--obs-size", type=int, default=OBS_SIZE_V1,
                        choices=[OBS_SIZE_V1, OBS_SIZE_V2],
                        help="Observation window. Must match what the model was trained on.")
    args = parser.parse_args()

    run_demo(args.model, args.difficulty, args.episodes, args.obs_size)
