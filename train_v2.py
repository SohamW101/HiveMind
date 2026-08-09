"""
RecurrentPPO (LSTM) training run at the v1 15x15 window.

Kept for comparison against the feed-forward baseline. It shares the curriculum callback
with the PPO runs - the local copy this file used to carry credited env 0's reward to
every finished episode, so the success rate driving promotions was noise from one worker.
"""
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from hivemind_env.env import OBS_SIZE_V1
from hivemind_env.models import CustomCombinedExtractor
from hivemind_env.training import (
    CurriculumCallback,
    get_device,
    make_env,
    num_parallel_envs,
)


if __name__ == "__main__":

    TOTAL_TIMESTEPS   = 10_000_000
    OBS_SIZE          = OBS_SIZE_V1   # matches models/checkpoints/recurrent_ppo_*.zip
    FEATURES_DIM      = 256
    INITIAL_LR        = 3e-4
    ENT_COEF          = 0.05          # Aggressive exploration
    GAMMA             = 0.995         # High discount for long-term target focus
    CURRICULUM_THRESH = 0.85
    CURRICULUM_WINDOW = 100
    CHECK_FREQ        = 500
    LOG_DIR           = "./tensorboard_logs/"
    CHECKPOINT_DIR    = "./models/checkpoints/"
    FINAL_MODEL_PATH  = "models/recurrent_ppo_hivemind_final"

    num_cpu = num_parallel_envs()
    print(f"[Config] Spawning {num_cpu} parallel environments at obs_size={OBS_SIZE}.")

    env = SubprocVecEnv([make_env(1, OBS_SIZE) for _ in range(num_cpu)])
    env = VecMonitor(env)  # Required for rollout/ep_rew_mean and rollout/ep_len_mean

    device = get_device()

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )

    print("Initializing RecurrentPPO (LSTM) model with Custom CNN Extractor...")
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
        learning_rate=INITIAL_LR,
        n_steps=4096 // num_cpu,  # Stabilizes training horizon
        ent_coef=ENT_COEF,
        gamma=GAMMA,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    curriculum_callback = CurriculumCallback(
        initial_lr=INITIAL_LR,
        check_freq=CHECK_FREQ,
        target_success_rate=CURRICULUM_THRESH,
        window_size=CURRICULUM_WINDOW,
        # RecurrentPPO is constructed with a constant LR, so there is no decay schedule
        # to restart. Rescaling one would silently drive the LR to zero.
        reset_lr_on_promotion=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000 // num_cpu,
        save_path=CHECKPOINT_DIR,
        name_prefix="recurrent_ppo_hivemind",
    )

    print(f"Starting RecurrentPPO training ({TOTAL_TIMESTEPS:,} steps)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
    )

    print("Training complete! Saving final model...")
    model.save(FINAL_MODEL_PATH)
    env.close()
    print(f"[Done] Model saved to {FINAL_MODEL_PATH}.zip")
