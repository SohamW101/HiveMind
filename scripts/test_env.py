import gymnasium as gym
import hivemind_env  # noqa: F401  (import registers the Gym env id)

def main():
    print("Initializing HiveMind Single-Agent Environment...")
    # render_mode=None is the headless mode. "DIRECT" is a PyBullet connection mode, not
    # a Gymnasium render mode - it is not in the env's metadata["render_modes"] and only
    # happened to work because the env treats anything that is not "human" as headless.
    env = gym.make("HiveMind-SingleAgent-v0", render_mode=None, difficulty_level=1)

    obs, info = env.reset()
    print("Environment reset successfully.")
    print(f"Observation keys: {obs.keys()}")
    print(f"Grid shape: {obs['grid'].shape}")

    # Run 10 random steps to verify everything runs without crashing
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | Action: {action} | Reward: {reward:.3f} | Terminated: {terminated}")

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
