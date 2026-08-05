import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from hivemind_env.env import HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor
import numpy as np
import os
from collections import deque

class CurriculumCallback(BaseCallback):
    """
    Callback for updating difficulty based on success rate.
    """
    def __init__(self, verbose=0, check_freq=1000, target_success_rate=0.9, window_size=100):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.delivery_history = deque(maxlen=window_size)
        
    def _on_step(self) -> bool:
        # Check if environment terminated with a drop-off reward
        # In our env, a successful drop-off gives +10 reward.
        for done, reward in zip(self.locals['dones'], self.locals['rewards']):
            if done:
                if reward >= 5.0:  # heuristic: success gives +10, subtract small penalties
                    self.delivery_history.append(1)
                else:
                    self.delivery_history.append(0)
                    
        if self.n_calls % self.check_freq == 0 and len(self.delivery_history) == self.window_size:
            success_rate = sum(self.delivery_history) / self.window_size
            if success_rate >= self.target_success_rate:
                current_difficulties = self.training_env.get_attr('difficulty_level')
                if current_difficulties and current_difficulties[0] < 4:
                    new_diff = current_difficulties[0] + 1
                    print(f"\n[Curriculum] Success rate {success_rate*100:.1f}% >= {self.target_success_rate*100:.1f}%. Upgrading difficulty to {new_diff}!\n")
                    self.training_env.set_attr('difficulty_level', new_diff)
                    self.delivery_history.clear()
        return True

def make_env(difficulty_level=1):
    def _init():
        # Using render_mode=None for faster training
        env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=difficulty_level)
        return env
    return _init

if __name__ == "__main__":
    # Dynamically scale to use up to 16 CPU cores on the server
    num_cpu = min(16, os.cpu_count() or 4)
    print(f"Server configuration detected: Spawning {num_cpu} parallel environments.")
    
    # Needs to be inside __main__ for SubprocVecEnv on Windows
    env = SubprocVecEnv([make_env(difficulty_level=1) for _ in range(num_cpu)])
    
    # Configure custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    print("Initializing PPO model with Custom CNN Extractor...")
    # PyTorch 2.x drops support for GTX 10-series (sm_61). Fallback to CPU on older GPUs.
    device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"
    print(f"Using device: {device}")
    
    # Initialize PPO
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=1, tensorboard_log="./tensorboard_logs/", device=device)
    
    # Setup callbacks
    curriculum_callback = CurriculumCallback(check_freq=500)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_cpu,
        save_path="./models/checkpoints/",
        name_prefix="ppo_hivemind"
    )
    
    print("Starting overnight training (5,000,000 steps)...")
    # Train for a massive overnight run
    model.learn(total_timesteps=5_000_000, callback=[curriculum_callback, checkpoint_callback])
    
    print("Training complete! Saving final model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_hivemind_final")
    env.close()
    print("Done!")
