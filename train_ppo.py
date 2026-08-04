"""Train a PPO agent on the HiveMind single-agent pick-and-deliver task.

Examples
--------
Quick smoke test (a few thousand steps, single process):
    python train_ppo.py --total-timesteps 20000 --n-envs 2 --no-subproc

Full run on difficulty 1 (fixed layout, no obstacles):
    python train_ppo.py --difficulty 1 --total-timesteps 1000000 --n-envs 8

Harder run with random obstacles, warm-starting from a previous model:
    python train_ppo.py --difficulty 3 --total-timesteps 3000000 \
        --load-from runs/d1/best_model.zip --run-name d3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed

from hivemind_rl.callbacks import RolloutStatsCallback
from hivemind_rl.env_utils import make_vec_envs
from hivemind_rl.features import HiveMindExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for HiveMind single-agent env")

    # --- experiment ---
    p.add_argument("--run-name", type=str, default="ppo_hivemind")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--load-from", type=str, default=None,
                   help="Path to a .zip model to continue training from.")

    # --- environment ---
    p.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3],
                   help="1: fixed layout, no obstacles | 2: random spawns | 3: + obstacles")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-substeps", type=int, default=4,
                   help="PyBullet sub-steps per env step. Lower = faster training.")
    p.add_argument("--no-subproc", action="store_true",
                   help="Use DummyVecEnv instead of SubprocVecEnv (easier debugging).")

    # --- PPO hyper-parameters ---
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-steps", type=int, default=512, help="Rollout length PER environment.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)

    # --- evaluation / checkpointing ---
    p.add_argument("--eval-freq", type=int, default=25_000,
                   help="Total env steps between evaluations.")
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--save-freq", type=int, default=100_000,
                   help="Total env steps between periodic checkpoints.")
    p.add_argument("--tensorboard", action="store_true",
                   help="Also write TensorBoard logs to <run>/tb.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # ---------------- environments ----------------
    train_env = make_vec_envs(
        n_envs=args.n_envs,
        difficulty_level=args.difficulty,
        seed=args.seed,
        num_substeps=args.num_substeps,
        max_steps=args.max_steps,
        use_subproc=not args.no_subproc,
    )
    # Separate eval env with a different seed so we measure generalisation,
    # not memorisation of the training layouts.
    eval_env = make_vec_envs(
        n_envs=1,
        difficulty_level=args.difficulty,
        seed=args.seed + 10_000,
        num_substeps=args.num_substeps,
        max_steps=args.max_steps,
        use_subproc=False,
    )

    # ---------------- model ----------------
    policy_kwargs = dict(
        features_extractor_class=HiveMindExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256, flag_output_dim=16),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=th.nn.ReLU,
    )

    common = dict(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tb") if args.tensorboard else None,
    )

    if args.load_from:
        print(f"Resuming from {args.load_from}")
        model = PPO.load(args.load_from, env=train_env, device=args.device)
        # Re-apply the CLI learning rate / clip range to the loaded model
        # (other hyper-parameters keep the values stored in the checkpoint).
        model.learning_rate = args.lr
        model.lr_schedule = get_schedule_fn(args.lr)
        model.clip_range = get_schedule_fn(args.clip_range)
    else:
        model = PPO("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, **common)

    print(model.policy)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Trainable parameters: {n_params:,}")

    # ---------------- callbacks ----------------
    # EvalCallback / CheckpointCallback count *per-env* steps, so divide.
    per_env = max(1, args.n_envs)
    callbacks = [
        RolloutStatsCallback(window=100, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=max(1, args.eval_freq // per_env),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(1, args.save_freq // per_env),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ]

    # ---------------- train ----------------
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.load_from is None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted - saving current model...")
    finally:
        model.save(str(run_dir / "final_model"))
        train_env.close()
        eval_env.close()
        print(f"\nSaved: {run_dir/'final_model.zip'}")
        print(f"Best:  {run_dir/'best_model.zip'}")


if __name__ == "__main__":
    # Required for SubprocVecEnv with the 'spawn' start method on Windows/macOS.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
