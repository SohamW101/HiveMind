import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HiveMindSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, difficulty_level=1):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty_level = difficulty_level
        self.dt = 0.1
        self.is_carrying = False

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(15, 15, 5), dtype=np.float32),
            "is_carrying": spaces.Discrete(2)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.is_carrying = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        grid = np.zeros((15, 15, 5), dtype=np.float32)
        return {
            "grid": grid,
            "is_carrying": int(self.is_carrying)
        }

    def _get_info(self):
        return {
            "difficulty": self.difficulty_level,
            "robot_pos": (0.0, 0.0, 0.0)
        }

    def render(self):
        pass

    def close(self):
        pass
