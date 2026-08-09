"""
PPO v1 training run: 15x15 observation window, 256-dim extractor, 10M steps.

This is the configuration that produced models/ppo_hivemind_v1_final.zip. The shared
scaffolding (schedule, curriculum callback, env factory, device probe) lives in
hivemind_env/training.py so v1 and v2 cannot drift apart again.
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from hivemind_env.env import OBS_SIZE_V1
from hivemind_env.models import CustomCombinedExtractor
from hivemind_env.training import (
    CurriculumCallback,
    get_device,
    linear_schedule,
    make_env,
    num_parallel_envs,
)
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == "__main__":

    # -- Hyperparameters ------------------------------------------------------
    TOTAL_TIMESTEPS   = 10_000_000
    OBS_SIZE          = OBS_SIZE_V1   # 15x15 - must match the model you intend to reuse
    FEATURES_DIM      = 256
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

    num_cpu = num_parallel_envs()
    print(f"[Config] Spawning {num_cpu} parallel environments at obs_size={OBS_SIZE}.")

    env = SubprocVecEnv([make_env(1, OBS_SIZE) for _ in range(num_cpu)])
    env = VecMonitor(env)  # Required for rollout/ep_rew_mean and rollout/ep_len_mean

    device = get_device()

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )

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

    print(
        f"\n[Training] Starting {TOTAL_TIMESTEPS:,} step run.\n"
        f"  Obs size: {OBS_SIZE}x{OBS_SIZE}x5\n"
        f"  Logs    : {LOG_DIR}\n"
        f"  Checkpts: {CHECKPOINT_DIR}\n"
        f"  Final   : {FINAL_MODEL_PATH}.zip\n"
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
        tb_log_name="PPO_v1",       # -> tensorboard_logs_ppo_v1/PPO_v1_1/
        reset_num_timesteps=True,
    )

    print("[Training] Complete! Saving final model...")
    model.save(FINAL_MODEL_PATH)
    env.close()
    print(f"[Done] Model saved to {FINAL_MODEL_PATH}.zip")
