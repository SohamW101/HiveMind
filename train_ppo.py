"""PPO training for the HiveMind single-agent environment.

Self-contained: feature extractor, wrappers, callbacks, training, evaluation and
GUI replay all live in this file. Works with the existing `hivemind_env` package
as-is - no changes to env.py required.

Install once:
    pip install stable-baselines3 torch tqdm rich

Usage
-----
    python train_ppo.py --total-timesteps 20000 --n-envs 2      # smoke test (~1 min)
    python train_ppo.py --difficulty 1 --total-timesteps 1000000 --n-envs 8
    python train_ppo.py --mode eval  --model runs/d1/best_model.zip --episodes 50
    python train_ppo.py --mode watch --model runs/d1/best_model.zip --episodes 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

import hivemind_env  # noqa: F401  registers "HiveMind-SingleAgent"

ENV_ID = "HiveMind-SingleAgent"


# --------------------------------------------------------------------------------------
# Wrapper: tag episodes as success / collision so SB3 can log a success rate.
# Terminal transitions are unambiguous: delivery gives +10, collision gives -2,
# and an episode can only terminate for one of those two reasons.
# --------------------------------------------------------------------------------------
class OutcomeInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            success = reward > 5.0
            info["is_success"] = bool(success)
            info["is_collision"] = bool(not success)
        elif truncated:
            info["is_success"] = False
            info["is_collision"] = False
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------------------
# Feature extractor: CNN over the 15x15x5 egocentric grid + MLP over the carrying flag.
# SB3's default would flatten the grid into 1125 numbers and throw away its spatial
# structure; convolutions keep "obstacle two cells ahead" meaningful anywhere in the map.
# --------------------------------------------------------------------------------------
class HiveMindExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_dim: int = 256, flag_dim: int = 16):
        super().__init__(observation_space, features_dim=cnn_dim + flag_dim)
        h, w, c = observation_space.spaces["grid"].shape  # (15, 15, 5)
        self.in_channels = c

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 15 -> 8
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 8 -> 4
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n_flat, cnn_dim), nn.ReLU())

        n_flag = int(observation_space.spaces["is_carrying"].n)  # SB3 one-hots this
        self.flag_head = nn.Sequential(nn.Linear(n_flag, flag_dim), nn.ReLU())

    def forward(self, obs: dict) -> th.Tensor:
        grid = obs["grid"]
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if grid.shape[-1] == self.in_channels:      # (B,H,W,C) -> (B,C,H,W)
            grid = grid.permute(0, 3, 1, 2).contiguous()
        flag = obs["is_carrying"].float()
        # Flatten to (N, -1) to handle extra singleton dimensions from SB3 rollout buffer
        flag = flag.view(flag.shape[0], -1)
        return th.cat([self.cnn_head(self.cnn(grid.float())), self.flag_head(flag)], dim=1)


# --------------------------------------------------------------------------------------
# Logging callback
# --------------------------------------------------------------------------------------
class StatsCallback(BaseCallback):
    def __init__(self, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.window = window

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer or [])[-self.window:]
        if not buf:
            return
        succ = float(np.mean([ep.get("is_success", 0.0) for ep in buf]))
        coll = float(np.mean([ep.get("is_collision", 0.0) for ep in buf]))
        self.logger.record("rollout/success_rate", succ)
        self.logger.record("rollout/collision_rate", coll)
        if self.verbose:
            print(f"[{self.num_timesteps:>9,}] success={succ:.1%}  collision={coll:.1%}  "
                  f"reward={np.mean([e['r'] for e in buf]):.2f}  len={np.mean([e['l'] for e in buf]):.0f}")


# --------------------------------------------------------------------------------------
# Env factories
# --------------------------------------------------------------------------------------
def make_env(difficulty: int, seed: int = 0, rank: int = 0, render_mode=None, monitor=True):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode, difficulty_level=difficulty)
        env = OutcomeInfoWrapper(env)
        if monitor:
            env = Monitor(env, info_keywords=("is_success", "is_collision"))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


def make_vec(n_envs: int, difficulty: int, seed: int, subproc: bool):
    thunks = [make_env(difficulty, seed, i, monitor=False) for i in range(n_envs)]
    venv = SubprocVecEnv(thunks, start_method="spawn") if (subproc and n_envs > 1) else DummyVecEnv(thunks)
    return VecMonitor(venv, info_keywords=("is_success", "is_collision"))


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------
def train(a: argparse.Namespace) -> None:
    run_dir = Path("runs") / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=2))

    train_env = make_vec(a.n_envs, a.difficulty, a.seed, not a.no_subproc)
    eval_env = make_vec(1, a.difficulty, a.seed + 10_000, subproc=False)

    if a.load_from:
        print(f"Resuming from {a.load_from}")
        model = PPO.load(a.load_from, env=train_env, device=a.device)
    else:
        model = PPO(
            "MultiInputPolicy", train_env,
            policy_kwargs=dict(
                features_extractor_class=HiveMindExtractor,
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
            n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=10,
            learning_rate=a.lr, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=a.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
            verbose=1, seed=a.seed, device=a.device,
            tensorboard_log=str(run_dir / "tb") if a.tensorboard else None,
        )

    # Set up logger to save progress.csv (which contains the loss curves)
    new_logger = configure(str(run_dir), ["stdout", "csv"])
    model.set_logger(new_logger)

    callbacks = [
        StatsCallback(),
        EvalCallback(eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir / "eval"),
                     eval_freq=max(1, 25_000 // a.n_envs), n_eval_episodes=20, deterministic=True),
        CheckpointCallback(save_freq=max(1, 100_000 // a.n_envs),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ppo"),
    ]

    try:
        model.learn(total_timesteps=a.total_timesteps, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=a.load_from is None)
    except KeyboardInterrupt:
        print("\nInterrupted - saving...")
    finally:
        model.save(str(run_dir / "final_model"))
        train_env.close()
        eval_env.close()
        print(f"\nSaved {run_dir/'final_model.zip'} (best: {run_dir/'best_model.zip'})")


def evaluate(a: argparse.Namespace) -> None:
    env = make_env(a.difficulty, seed=a.seed)()
    model = PPO.load(a.model, device=a.device)
    rewards, lengths, outcome = [], [], {"success": 0, "collision": 0, "timeout": 0}

    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total, steps, done = total + r, steps + 1, term or trunc
        key = "success" if info.get("is_success") else "collision" if info.get("is_collision") else "timeout"
        outcome[key] += 1
        rewards.append(total)
        lengths.append(steps)
        print(f"ep {ep+1:>3}/{a.episodes}  reward={total:7.2f}  steps={steps:>4}  {key}")

    env.close()
    n = a.episodes
    print("\n" + "=" * 42)
    print(f"  success rate : {outcome['success']/n:.1%}")
    print(f"collision rate : {outcome['collision']/n:.1%}")
    print(f"  timeout rate : {outcome['timeout']/n:.1%}")
    print(f"   mean reward : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"   mean length : {np.mean(lengths):.1f}")
    print("=" * 42)


def watch(a: argparse.Namespace) -> None:
    env = make_env(a.difficulty, seed=a.seed, render_mode="human", monitor=False)()
    model = PPO.load(a.model, device=a.device)
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total, steps, done = total + r, steps + 1, term or trunc
        key = "SUCCESS" if info.get("is_success") else "COLLISION" if info.get("is_collision") else "timeout"
        print(f"Episode {ep+1}: {key}  reward={total:.2f}  steps={steps}")
    env.close()


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO for HiveMind single-agent env")
    p.add_argument("--mode", choices=["train", "eval", "watch"], default="train")
    p.add_argument("--run-name", default="ppo")
    p.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # train
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length PER env")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--no-subproc", action="store_true", help="single process; easier debugging")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--load-from", default=None, help="warm-start from a saved .zip")
    # eval / watch
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    if args.mode == "train":
        train(args)
    else:
        if not args.model:
            raise SystemExit("--model is required for --mode eval/watch")
        (evaluate if args.mode == "eval" else watch)(args)
