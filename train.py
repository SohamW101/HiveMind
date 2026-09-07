"""
Train the shared HiveMind MARL policy on branch multi-agent-v2.

Uses the modular semantic feature extractor (HiveMindExtractor with 1D CNN for LiDAR
and dedicated encoders for self, teammates, cartons, and comms), parameter-shared
Stable-Baselines3 PPO, and a robust 1 -> 2 -> 4 -> 8 -> 12 carton curriculum.

Quick pipeline test:
    python train.py --smoke --worlds 2

Full training run (starts at 1 carton and walks up on rolling success):
    python -u train.py --run-name v2_main --num-cartons 1 --curriculum --timesteps 5000000 --worlds 8 --checkpoint-every 25000
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from hivemind_env.env import (
    DEFAULT_OBS_DIM,
    MSG_DROPOUT_DEFAULT,
    MSG_TOKENS,
    NUM_AGENTS,
    SHAPING_SCALE_DEFAULT,
    max_steps_for,
)
from hivemind_env.models import HiveMindExtractor, get_policy_kwargs
from hivemind_env.subproc_vec_env import HiveMindSubprocVecEnv
from hivemind_env.training import (
    CurriculumCallback,
    EpisodeStatsCallback,
    MessageStatsCallback,
    get_device,
    linear_schedule,
    num_parallel_envs,
)
from hivemind_env.vec_env import HiveMindSharedPolicyVecEnv


def tensorboard_available() -> bool:
    """Probes whether tensorboard is importable."""
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def build_env(
    worlds: int,
    difficulty: int = 1,
    seed: int | None = None,
    backend: str = "subproc",
    substeps: int | None = None,
    max_steps: int | None = None,
    num_cartons: int | None = None,
    shaping: bool = True,
    gamma: float = 0.99,
    shaping_scale: float | None = None,
    comms: bool = False,
    msg_dropout: float = MSG_DROPOUT_DEFAULT,
):
    """
    Builds the vectorized multi-agent environment wrapped in VecMonitor.
    Presents N 4-robot warehouses as 4N parameter-sharing single-agent slots.
    """
    cls = HiveMindSubprocVecEnv if backend == "subproc" else HiveMindSharedPolicyVecEnv
    kwargs = dict(
        substeps=substeps,
        max_steps=max_steps,
        shaping=shaping,
        gamma=gamma,
        comms=comms,
        msg_dropout=msg_dropout,
    )
    if num_cartons is not None:
        kwargs["num_cartons"] = num_cartons
    if shaping_scale is not None:
        kwargs["shaping_scale"] = shaping_scale

    vec = cls(
        num_worlds=worlds,
        difficulty_level=difficulty,
        obs_dim=DEFAULT_OBS_DIM,
        seed=seed,
        **kwargs,
    )
    return VecMonitor(vec)


def main():
    parser = argparse.ArgumentParser(description="Train the HiveMind MARL policy")
    parser.add_argument("--run-name", type=str, default=None, help="Name for run, logs, and checkpoints")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total robot steps")
    parser.add_argument("--worlds", type=int, default=None, help="Parallel warehouse worlds (each has 4 robots)")
    parser.add_argument("--backend", choices=["subproc", "inprocess"], default="subproc", help="subproc or inprocess")
    parser.add_argument("--substeps", type=int, default=5, help="Physics substeps per env step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout length per slot")
    parser.add_argument("--batch-size", type=int, default=1024, help="PPO minibatch size")
    parser.add_argument("--difficulty", type=int, default=1, help="Initial difficulty level")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--curriculum", action="store_true", help="Promote 1 -> 2 -> 3 -> 4 -> 8 -> 12 cartons on success")
    parser.add_argument("--curriculum-threshold", type=float, default=0.70, help="Rolling success rate threshold for promotion")
    parser.add_argument("--curriculum-fraction", type=float, default=0.75, help="Rolling delivered fraction threshold for promotion")
    parser.add_argument("--num-cartons", type=int, default=1, help="Starting cartons (default: 1)")
    parser.add_argument("--max-steps", type=int, default=None, help="Episode step limit (auto per carton count if omitted)")
    parser.add_argument("--shaping-scale", type=float, default=SHAPING_SCALE_DEFAULT, help="Potential shaping scale (default: 60.0)")
    parser.add_argument("--no-shaping", action="store_true", help="Turn off potential reward shaping")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--comms", action="store_true", help="Enable 16-token discrete broadcast channel")
    parser.add_argument("--msg-dropout", type=float, default=MSG_DROPOUT_DEFAULT, help="Per-link communication dropout fraction")
    parser.add_argument("--checkpoint-every", type=int, default=25_000, help="Checkpoint save interval in robot-steps")
    parser.add_argument("--min-steps-before-demote", type=int, default=400_000, help="Grace period steps at a level before demotion check")
    parser.add_argument("--init-from", type=str, default=None, help="Path to checkpoint zip to warm-start policy weights")
    parser.add_argument("--smoke", action="store_true", help="Run quick 4,096 step pipeline check")

    args = parser.parse_args()

    if args.smoke:
        args.timesteps = 4096
        args.worlds = args.worlds or 2
        args.batch_size = 64
        args.n_steps = 64
        args.checkpoint_every = 2048
        if args.run_name is None:
            args.run_name = "smoke_test"

    # Default worlds based on host CPU physical cores
    worlds = args.worlds or max(2, min(8, num_parallel_envs(cap=8)))
    slots = worlds * NUM_AGENTS

    # Auto-scale batch size to buffer capacity
    buffer_size = args.n_steps * slots
    if args.batch_size > buffer_size:
        print(f"  [auto] batch_size {args.batch_size} > buffer {buffer_size}; clamped to {buffer_size}")
        args.batch_size = buffer_size

    run_name = args.run_name or f"ppo_v2_{datetime.now():%Y%m%d_%H%M%S}"
    device = get_device()
    has_tb = tensorboard_available()

    print("=" * 80)
    print("  HiveMind v2 — Shared-Policy PPO with Modular Semantic Feature Extractor")
    print("=" * 80)
    print(f"  run name     : {run_name}")
    print(f"  worlds       : {worlds} parallel ({slots} policy slots, {NUM_AGENTS} bots/world)")
    print(f"  timesteps    : {args.timesteps:,} robot-steps (~{args.timesteps // slots:,} world-steps)")
    print(f"  starting box : {args.num_cartons} carton(s) (difficulty {args.difficulty})")
    print(f"  curriculum   : {'ENABLED (1 -> 2 -> 3 -> 4 -> 8 -> 12)' if args.curriculum else 'disabled'}")
    print(f"  shaping      : {'OFF (sparse reward)' if args.no_shaping else f'ON (scale {args.shaping_scale})'}")
    print(f"  comms        : {'ON (16 tokens)' if args.comms else 'off (silent)'}")
    print(f"  extractor    : HiveMindExtractor (1D CNN + Modular Self/Teammates/Cartons/Comms)")
    print(f"  device       : {device}")
    print(f"  init from    : {args.init_from or 'none (scratch)'}")
    print(f"  tensorboard  : {'logging to tensorboard_logs/' if has_tb else 'disabled (tensorboard not installed)'}")
    print(f"  checkpoints  : every {args.checkpoint_every:,} steps to models/checkpoints/{run_name}/")
    print("=" * 80, flush=True)

    # Build vectorized env
    env = build_env(
        worlds=worlds,
        difficulty=args.difficulty,
        seed=args.seed,
        backend=args.backend,
        substeps=args.substeps,
        max_steps=args.max_steps,
        num_cartons=args.num_cartons,
        shaping=not args.no_shaping,
        gamma=args.gamma,
        shaping_scale=args.shaping_scale,
        comms=args.comms,
        msg_dropout=args.msg_dropout,
    )

    # Setup callbacks
    os.makedirs(f"models/checkpoints/{run_name}", exist_ok=True)
    callbacks = [
        EpisodeStatsCallback(window_size=100),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // slots),
            save_path=f"models/checkpoints/{run_name}",
            name_prefix="ckpt",
            verbose=1,
        ),
    ]

    if args.curriculum:
        # Note: reset_lr_on_promotion=False to prevent the catastrophic sawtooth LR bug
        callbacks.append(
            CurriculumCallback(
                initial_lr=args.lr,
                check_freq=1000,
                target_success_rate=args.curriculum_threshold,
                target_fraction=args.curriculum_fraction,
                window_size=500,
                reset_lr_on_promotion=False,
                min_steps_before_demote=args.min_steps_before_demote,
                verbose=1,
            )
        )

    if args.comms:
        callbacks.append(MessageStatsCallback())

    # Build PPO model with HiveMindExtractor
    policy_kwargs = get_policy_kwargs(features_dim=256)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(args.lr),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        tensorboard_log="tensorboard_logs" if has_tb else None,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=args.seed,
    )

    if args.init_from:
        print(f"\n[Warm-Start] Loading policy weights from: {args.init_from}")
        src = PPO.load(args.init_from, device=device)
        model.policy.load_state_dict(src.policy.state_dict())
        print("[Warm-Start] Policy weights successfully loaded!\n")

    print(f"\nStarting training on {device}...", flush=True)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=run_name,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model checkpoint...")

    # Save final model
    os.makedirs("models", exist_ok=True)
    final_path = f"models/{run_name}_final.zip"
    model.save(final_path)
    print(f"\nTraining finished! Saved final model weights to: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
