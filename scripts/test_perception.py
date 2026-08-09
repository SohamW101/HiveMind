import gymnasium as gym
import numpy as np
import hivemind_env  # noqa: F401  (import registers the Gym env id)
from hivemind_env.env import LIDAR_NUM_RAYS

def test_perception():
    print("--- Perception System Test (Step 3) ---")
    env = gym.make("HiveMind-SingleAgent-v0", render_mode=None, difficulty_level=2)
    obs, info = env.reset(seed=42)

    # Assert against what the env is actually configured for rather than hardcoded
    # numbers: obs_size is now a per-run choice (15 for v1 models, 21 for v2) and the
    # info-dict scan uses LIDAR_NUM_RAYS, which is 180, not the 36 this used to expect.
    obs_size = env.unwrapped.obs_size
    centre = obs_size // 2

    grid = obs["grid"]
    is_carrying = obs["is_carrying"]
    lidar_distances = info["lidar_distances"]

    print(f"Observation 'grid' shape: {grid.shape} (Expected: ({obs_size}, {obs_size}, 5))")
    print(f"Observation 'is_carrying': {is_carrying}")
    print(f"LiDAR rays count: {len(lidar_distances)} (Expected: {LIDAR_NUM_RAYS})")
    print(f"LiDAR distance range: min={min(lidar_distances):.3f}m, max={max(lidar_distances):.3f}m")

    assert grid.shape == (obs_size, obs_size, 5), f"Invalid grid shape {grid.shape}"
    assert len(lidar_distances) == LIDAR_NUM_RAYS, f"Invalid ray count {len(lidar_distances)}"
    assert grid[centre, centre, 4] == 1.0, "Self-agent center channel missing"

    # Step through 5 actions and print perception state
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {step+1}: Action={action}, "
              f"Robot Pos={[round(p,2) for p in info['robot_pos']]}, "
              f"Obstacle cells detected={np.sum(obs['grid'][:,:,0])}")
        if term or trunc:
            obs, info = env.reset()

    env.close()
    print("--- Perception System Test Passed Successfully! ---")

if __name__ == "__main__":
    test_perception()
