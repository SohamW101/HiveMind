import gymnasium as gym
import numpy as np
import hivemind_env

def test_perception():
    print("--- Perception System Test (Step 3) ---")
    env = gym.make("HiveMind-SingleAgent", render_mode=None, difficulty_level=2)
    obs, info = env.reset(seed=42)

    grid = obs["grid"]
    is_carrying = obs["is_carrying"]
    lidar_distances = info["lidar_distances"]

    print(f"Observation 'grid' shape: {grid.shape} (Expected: (15, 15, 5))")
    print(f"Observation 'is_carrying': {is_carrying}")
    print(f"LiDAR rays count: {len(lidar_distances)} (Expected: 36)")
    print(f"LiDAR distance range: min={min(lidar_distances):.3f}m, max={max(lidar_distances):.3f}m")

    assert grid.shape == (15, 15, 5), f"Invalid grid shape {grid.shape}"
    assert len(lidar_distances) == 36, f"Invalid ray count {len(lidar_distances)}"
    assert grid[7, 7, 4] == 1.0, "Self-agent center channel missing"

    # Step through 5 actions and print perception state
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {step+1}: Action={action}, Robot Pos={[round(p,2) for p in info['robot_pos']]}, Obstacle cells detected={np.sum(obs['grid'][:,:,0])}")

    env.close()
    print("--- Perception System Test Passed Successfully! ---")

if __name__ == "__main__":
    test_perception()
