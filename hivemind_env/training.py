"""
Shared training scaffolding for the HiveMind PPO runs.

`train.py` (v1), `train_ppo_v2.py` (v2) and `train_v2.py` (RecurrentPPO) all used to
carry their own byte-identical copies of the schedule, the curriculum callback, the env
factory and the device probe. Keeping one copy means a fix lands everywhere at once -
the learning-rate reset bug below existed in two files and the success-attribution bug
in a third.
"""
import os
from collections import deque
from typing import Callable

import torch

from hivemind_env.env import DEFAULT_OBS_SIZE, HiveMindSingleAgentEnv

from stable_baselines3.common.callbacks import BaseCallback

# Highest difficulty the curriculum will promote to.
MAX_DIFFICULTY_LEVEL = 4

# An episode that ends on a successful dropoff scores +10 (minus the 0.01 time penalty)
# on its final step. Collisions end at -2.01 and truncation at whatever the last shaping
# term was, so this threshold cleanly separates deliveries from every other outcome.
SUCCESS_REWARD_THRESHOLD = 5.0


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate decay from initial_value -> 0.0 over the entire training run.
    progress_remaining goes from 1.0 (start) to 0.0 (end).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def restart_schedule(initial_value: float, progress_at_restart: float) -> Callable[[float], float]:
    """
    Linear decay that returns `initial_value` at the moment of the restart and still
    reaches 0.0 at the end of the run.

    Reinstalling `linear_schedule(initial_value)` mid-run does nothing at all: SB3 always
    evaluates the schedule at the globally decreasing `_current_progress_remaining`, so
    the "reset" schedule is the same function that was already installed. Rescaling by
    the progress remaining at the restart is what actually lifts the LR back up.
    """
    if progress_at_restart <= 0.0:
        return lambda progress_remaining: 0.0

    def func(progress_remaining: float) -> float:
        return initial_value * max(0.0, progress_remaining / progress_at_restart)
    return func


class CurriculumCallback(BaseCallback):
    """
    Upgrades environment difficulty when the rolling success rate exceeds
    `target_success_rate` over the last `window_size` completed episodes.

    Also:
    - Logs `curriculum/difficulty_level` at every check so TensorBoard shows the exact
      timestep each difficulty transition happened.
    - Restarts the learning rate schedule whenever the difficulty increases, giving the
      agent a fresh learning burst to adapt to the new environment.
    """
    def __init__(
        self,
        initial_lr: float,
        check_freq: int = 1000,
        target_success_rate: float = 0.70,
        window_size: int = 100,
        reset_lr_on_promotion: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.check_freq = check_freq
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.reset_lr_on_promotion = reset_lr_on_promotion
        self.delivery_history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # Record outcome for every environment that finished this step. Pair each `done`
        # with *its own* env's reward - indexing rewards[0] inside this loop credited
        # env 0's outcome to all 16 workers.
        for done, reward in zip(self.locals["dones"], self.locals["rewards"]):
            if done:
                self.delivery_history.append(1 if reward >= SUCCESS_REWARD_THRESHOLD else 0)

        if self.n_calls % self.check_freq != 0:
            return True

        current_level = self.training_env.get_attr("difficulty_level")[0]
        self.logger.record("curriculum/difficulty_level", current_level)

        if len(self.delivery_history) < self.window_size:
            return True

        success_rate = sum(self.delivery_history) / self.window_size
        self.logger.record("curriculum/success_rate", success_rate)

        if success_rate >= self.target_success_rate and current_level < MAX_DIFFICULTY_LEVEL:
            new_level = current_level + 1
            print(
                f"\n[Curriculum] Step {self.num_timesteps:,} | "
                f"Success rate {success_rate*100:.1f}% >= {self.target_success_rate*100:.0f}% | "
                f"Upgrading difficulty: Level {current_level} -> Level {new_level}\n"
            )
            self.training_env.set_attr("difficulty_level", new_level)
            self.delivery_history.clear()

            if self.reset_lr_on_promotion:
                progress = self.model._current_progress_remaining
                self.model.lr_schedule = restart_schedule(self.initial_lr, progress)
                print(f"[Curriculum] Learning rate schedule restarted at {self.initial_lr:.0e}")

        return True


# SB3 pickles `learning_rate`, `lr_schedule` and `clip_range` as closures. Rebuilding a
# code object pickled under a different Python minor version is unsafe: on 3.12 it
# surfaced as "UserWarning: Could not deserialize object lr_schedule ... code expected at
# most 16 arguments, got 18", and on 3.14 it segfaults the interpreter outright.
#
# None of the three matters at inference time - they only drive optimisation - so replace
# them with harmless stand-ins when loading a model to evaluate or demo.
INFERENCE_CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _progress_remaining: 0.0,
    "clip_range": lambda _progress_remaining: 0.2,
}


def load_policy(model_path: str, device: str = "cpu", recurrent: bool | None = None):
    """
    Loads a saved policy for evaluation, immune to cross-Python-version pickle breakage.

    `recurrent` defaults to sniffing the filename, so a RecurrentPPO checkpoint is not
    handed to PPO.load (which cannot build an LSTM policy).
    Returns (model, is_recurrent).
    """
    if recurrent is None:
        recurrent = "recurrent" in os.path.basename(model_path).lower()

    if recurrent:
        from sb3_contrib import RecurrentPPO
        algo = RecurrentPPO
    else:
        from stable_baselines3 import PPO
        algo = PPO

    model = algo.load(model_path, device=device, custom_objects=INFERENCE_CUSTOM_OBJECTS)
    return model, recurrent


def make_env(difficulty_level: int = 1, obs_size: int = DEFAULT_OBS_SIZE):
    """
    Env factory for SubprocVecEnv.

    `obs_size` is explicit on purpose: it determines the CNN's flatten width, so a model
    trained at one size cannot be loaded into an env built at another.
    """
    def _init():
        return HiveMindSingleAgentEnv(
            render_mode=None, difficulty_level=difficulty_level, obs_size=obs_size
        )
    return _init


def get_device() -> str:
    """Safe for a local GTX 1050 (sm_61) and a server A5000 (sm_86) alike."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        if cap[0] >= 7:
            print(f"[Device] GPU: {name} (sm_{cap[0]}{cap[1]}) -> Using CUDA (cuDNN disabled for stability)")
            torch.backends.cudnn.enabled = False
            return "cuda"
        print(
            f"[Device] GPU: {name} (sm_{cap[0]}{cap[1]}) is below sm_70. "
            f"PyTorch 2.x requires sm_70+. Falling back to CPU."
        )
        return "cpu"
    print("[Device] No CUDA GPU detected -> Using CPU")
    return "cpu"


def num_parallel_envs(cap: int = 16) -> int:
    return min(cap, os.cpu_count() or 4)
