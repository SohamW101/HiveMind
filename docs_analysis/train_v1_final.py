import gymnasium as gym
import torch
import os
import sys
import numpy as np
from collections import deque
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from hivemind_env.env import HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────────────────────
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate decay from initial_value → 0.0 over the entire training run.
    progress_remaining goes from 1.0 (start) to 0.0 (end).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Callback
# ─────────────────────────────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Upgrades environment difficulty when the rolling success rate
    exceeds `target_success_rate` over the last `window_size` completed episodes.

    Also:
    - Logs `curriculum/difficulty_level` to TensorBoard at every check so you
      can see the exact timestep each difficulty transition happened.
    - Resets the learning rate to `initial_lr` whenever the difficulty increases,
      giving the agent a fresh learning burst to adapt to the new environment.
    """
    def __init__(
        self,
        initial_lr: float,
        check_freq: int = 1000,
        target_success_rate: float = 0.70,
        window_size: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.check_freq = check_freq
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.delivery_history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # Record outcome for every environment that finished this step
        for done, reward in zip(self.locals["dones"], self.locals["rewards"]):
            if done:
                # Successful delivery: dropoff gives +10 minus small penalties → net > 5.0
                self.delivery_history.append(1 if reward >= 5.0 else 0)

        # Check curriculum condition every `check_freq` steps
        if self.n_calls % self.check_freq == 0:
            current_level = self.training_env.get_attr("difficulty_level")[0]

            # Always log current difficulty so TensorBoard shows the exact transition step
            self.logger.record("curriculum/difficulty_level", current_level)

            if len(self.delivery_history) == self.window_size:
                success_rate = sum(self.delivery_history) / self.window_size
                self.logger.record("curriculum/success_rate", success_rate)

                if success_rate >= self.target_success_rate and current_level < 4:
                    new_level = current_level + 1
                    print(
                        f"\n[Curriculum] Step {self.num_timesteps:,} | "
                        f"Success rate {success_rate*100:.1f}% >= {self.target_success_rate*100:.0f}% | "
                        f"Upgrading difficulty: Level {current_level} → Level {new_level}\n"
                    )
                    self.training_env.set_attr("difficulty_level", new_level)
                    self.delivery_history.clear()

                    # Reset learning rate to initial value so the agent adapts quickly
                    # to the new environment configuration
                    self.model.lr_schedule = linear_schedule(self.initial_lr)
                    print(f"[Curriculum] Learning rate reset to {self.initial_lr:.0e}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment Factory
# ─────────────────────────────────────────────────────────────────────────────
def make_env(difficulty_level: int = 1):
    def _init():
        return HiveMindSingleAgentEnv(render_mode=None, difficulty_level=difficulty_level)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Device Detection (safe for local GTX 1050 sm_61 AND server A5000 sm_86)
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        if cap[0] >= 7:
            print(f"[Device] GPU: {name} (sm_{cap[0]}{cap[1]}) → Using CUDA")
            torch.backends.cudnn.benchmark = True  # +10-15% speed on fixed-size inputs
            return "cuda"
        else:
            print(
                f"[Device] GPU: {name} (sm_{cap[0]}{cap[1]}) is below sm_70. "
                f"PyTorch 2.x requires sm_70+. Falling back to CPU."
            )
            return "cpu"
    print("[Device] No CUDA GPU detected → Using CPU")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Hyperparameters ──────────────────────────────────────────────────────
    TOTAL_TIMESTEPS   = 10_000_000
    INITIAL_LR        = 3e-4          # Decays linearly to 0.0 over training
    N_STEPS           = 2048          # Steps collected per env per update
    BATCH_SIZE        = 512           # Minibatch size for gradient updates
    N_EPOCHS          = 10            # PPO update epochs per rollout
    ENT_COEF          = 0.01          # Exploration coefficient (standard PPO)
    GAMMA             = 0.99          # Discount factor
    GAE_LAMBDA        = 0.95          # GAE smoothing
    CLIP_RANGE        = 0.2           # PPO clip range
    CURRICULUM_THRESH = 0.70          # Success rate to graduate to next level
    CURRICULUM_WINDOW = 100           # Rolling window size for success rate
    CHECK_FREQ        = 1000          # How often curriculum checks (in steps)
    CHECKPOINT_FREQ   = 500_000       # Save checkpoint every N steps
    LOG_DIR           = "./tensorboard_logs_ppo_v1/"
    CHECKPOINT_DIR    = "./models/checkpoints_ppo_v1/"
    FINAL_MODEL_PATH  = "./models/ppo_hivemind_v1_final"

    # ── CPU count ────────────────────────────────────────────────────────────
    num_cpu = min(16, os.cpu_count() or 4)
    print(f"[Config] Spawning {num_cpu} parallel environments.")

    # ── SubprocVecEnv start method (fork is faster on Linux servers) ─────────
    if sys.platform != "win32":
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)

    # ── Build and wrap environment ───────────────────────────────────────────
    env = SubprocVecEnv([make_env(difficulty_level=1) for _ in range(num_cpu)])
    env = VecMonitor(env)  # Required for rollout/ep_rew_mean and rollout/ep_len_mean

    # ── Feature extractor ────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # ── Device ───────────────────────────────────────────────────────────────
    device = get_device()

    # ── PPO Model ────────────────────────────────────────────────────────────
    print("[Config] Initializing PPO with Standard Feed-Forward policy...")
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(INITIAL_LR),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    curriculum_callback = CurriculumCallback(
        initial_lr=INITIAL_LR,
        check_freq=CHECK_FREQ,
        target_success_rate=CURRICULUM_THRESH,
        window_size=CURRICULUM_WINDOW,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_v1",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    print(
        f"\n[Training] Starting 10,000,000 step run.\n"
        f"  Logs    : {LOG_DIR}\n"
        f"  Checkpts: {CHECKPOINT_DIR}\n"
        f"  Final   : {FINAL_MODEL_PATH}.zip\n"
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
        tb_log_name="PPO_v1",       # TensorBoard run folder = tensorboard_logs_ppo_v1/PPO_v1_1/
        reset_num_timesteps=True,
    )

    print("[Training] Complete! Saving final model...")
    model.save(FINAL_MODEL_PATH)
    env.close()
    print(f"[Done] Model saved to {FINAL_MODEL_PATH}.zip")
