import os
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import RecurrentPPO
from hivemind_env.env import HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor
import numpy as np

class CurriculumCallback(BaseCallback):
    """
    Custom callback for Curriculum Learning.
    Evaluates agent success rate and dynamically scales difficulty from Level 1 to 4.
    """
    def __init__(self, check_freq: int = 1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history = []
        self.current_level = 1

    def _on_step(self) -> bool:
        # Check if the episode just ended
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                # We define success as ending the episode near the depot with a positive reward
                success = 1.0 if self.locals["rewards"][0] > 5.0 else 0.0
                self.success_history.append(success)
                if len(self.success_history) > 100:
                    self.success_history.pop(0)

        # Check curriculum progression periodically
        if self.n_calls % self.check_freq == 0 and len(self.success_history) == 100:
            success_rate = np.mean(self.success_history)
            if success_rate >= 0.85 and self.current_level < 4:
                self.current_level += 1
                if self.verbose > 0:
                    print(f"\n--- CURRICULUM UPGRADE: Success Rate {success_rate:.2f} >= 85%. Advancing to Level {self.current_level} ---\n")
                
                # Update difficulty level in all environments
                self.training_env.env_method("set_difficulty", self.current_level)
                self.success_history.clear()
        return True

def make_env(difficulty_level=1):
    def _init():
        # Using render_mode=None for faster training
        env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=difficulty_level)
        # We must add a set_difficulty method if it doesn't exist, or just modify the env directly
        def set_difficulty(level):
            env.unwrapped.difficulty_level = level
        env.set_difficulty = set_difficulty
        return env
    return _init

if __name__ == "__main__":
    # Dynamically scale to use up to 16 CPU cores on the server
    num_cpu = min(16, os.cpu_count() or 4)
    print(f"Server configuration detected: Spawning {num_cpu} parallel environments.")
    
    # Create the parallel environments
    env = SubprocVecEnv([make_env(difficulty_level=1) for _ in range(num_cpu)])
    
    # CRITICAL FIX: Wrap in VecMonitor so TensorBoard logs the ep_rew_mean and ep_len_mean!
    env = VecMonitor(env)
    
    # Configure custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    print("Initializing RecurrentPPO (LSTM) model with Custom CNN Extractor...")
    # PyTorch 2.x drops support for GTX 10-series (sm_61). Fallback to CPU on older GPUs.
    device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"
    print(f"Using device: {device}")
    
    # Initialize RecurrentPPO (LSTMs) for True Spatial Memory
    model = RecurrentPPO(
        "MultiInputLstmPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log="./tensorboard_logs/", 
        device=device,
        # Optimal Hyperparameters for LSTMs & Sparse/APF Level 4 exploration
        n_steps=4096 // num_cpu,  # Stabilizes training horizon
        ent_coef=0.05,            # Aggressive exploration
        gamma=0.995               # High discount for long-term target focus
    )
    
    # Setup callbacks
    curriculum_callback = CurriculumCallback(check_freq=500)
    checkpoint_callback = CheckpointCallback(
        save_freq=250000 // num_cpu,
        save_path="./models/checkpoints/",
        name_prefix="recurrent_ppo_hivemind"
    )
    
    print("Starting optimal Robotics APF training (10,000,000 steps)...")
    # Train for a massive run to conquer Level 4
    model.learn(total_timesteps=10_000_000, callback=[curriculum_callback, checkpoint_callback])
    
    print("Training complete! Saving final model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/recurrent_ppo_hivemind_final")
    env.close()
    print("Done!")
