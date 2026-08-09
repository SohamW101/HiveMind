import time
from hivemind_env.env import OBS_SIZE_V1, HiveMindSingleAgentEnv

def test():
    print("Testing environment...")
    env = HiveMindSingleAgentEnv(
        render_mode="human", difficulty_level=2, obs_size=OBS_SIZE_V1
    )
    obs, info = env.reset()
    print("Environment reset successful.")

    episodes = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(int(action))
        time.sleep(0.05)

        # Previously this loop ignored the flags and kept driving a finished episode.
        if terminated or truncated:
            episodes += 1
            print(f"  Episode {episodes} ended at step {i+1} "
                  f"(terminated={terminated}, truncated={truncated}) - resetting.")
            obs, info = env.reset()

    env.close()
    print(f"Test finished successfully. Episodes completed: {episodes}")

if __name__ == "__main__":
    test()
