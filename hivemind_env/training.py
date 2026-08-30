"""
Shared training scaffolding for the HiveMind PPO runs.

Ported from the `single-agent-rl` branch (roadmap step 2) and adapted for the 4-robot
warehouse. The single-agent version existed because `train.py` (v1), `train_ppo_v2.py`
(v2) and `train_v2.py` (RecurrentPPO) all carried byte-identical copies of the schedule,
the curriculum callback, the env factory and the device probe. Keeping one copy means a
fix lands everywhere at once - the learning-rate reset bug below existed in two files and
the success-attribution bug in a third.

WHAT CHANGED IN THE PORT (each site is marked "PORT NOTE"):
  - env class and factory now build HiveMindMultiAgentEnv
  - the curriculum ladder is carton count (4 -> 8 -> 12), not the old obstacle levels
  - the success test reads a per-episode signal from `info`, not the last step's reward
  - obs-dim constants are imported from env.py, which pins them (roadmap step 3)
  - make_env takes an explicit per-worker seed

STATUS: roadmap steps 3 and 4 have landed. The env reports a real (4, OBS_DIM_V2)
observation, pays the reward structure from MAWC_Technical_Specification.pdf section 3,
and ends episodes on completion or the step limit. What is still missing before a run
means anything: the greedy baseline (step 5) and models.py / train.py (step 6).
"""
import os
from collections import deque
from typing import Callable

import torch

from hivemind_env.env import DEFAULT_OBS_DIM, OBS_DIM_V3, HiveMindMultiAgentEnv

from stable_baselines3.common.callbacks import BaseCallback

# PORT NOTE: the single-agent env exported OBS_SIZE_V1 / OBS_SIZE_V2 / DEFAULT_OBS_SIZE
# and this module imported them. When this file was first ported, the multi-agent env
# exported no constants at all, so placeholders were declared here.
#
# Roadmap step 3 has since landed and env.py is now the single source of truth: the
# observation is pinned at OBS_DIM_V3 = 177 floats per robot (129 world features + 48
# reserved message slots), with the full component table in env.py's header. Import it,
# never restate it - a second copy of the number is exactly how the two drift apart.
NUM_AGENTS = 4

# Curriculum ladder. The single-agent branch promoted through four obstacle-density
# levels; here the difficulty knob is how many of the 12 cartons must be delivered
# (the project roadmap, step 8: "the 4 -> 8 -> 12 carton curriculum").
#
# `HiveMindMultiAgentEnv` accepts `difficulty_level` and stores it, but nothing reads it
# yet, so promotion is currently a no-op on the world. The callback still logs and still
# promotes, which keeps the wiring testable before step 8 makes it bite.
CURRICULUM_CARTONS = {1: 4, 2: 8, 3: 12}
MAX_DIFFICULTY_LEVEL = max(CURRICULUM_CARTONS)

# PORT NOTE - this constant changed meaning.
#
# Single-agent: an episode ending on a successful dropoff scored +10 on its final step,
# collisions ended at -2.01 and truncation at whatever the last shaping term was, so 5.0
# cleanly separated deliveries from every other outcome.
#
# Multi-agent has no single terminal reward that means "the job is done". Now that step
# 4 has landed, a completing episode's final step measures ~+143 (10 delivery + 100
# completion + ~49 makespan, weighted 0.90, plus 0.10 x 2 own delivery) while any
# non-completing step sits near zero. 50.0 sits in that gap with room on both sides.
# It is a fallback only: `_episode_succeeded` prefers `info["is_success"]`, which the
# env now sets explicitly.
SUCCESS_REWARD_THRESHOLD = 50.0


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


def _episode_succeeded(info, reward) -> bool:
    """
    Did the episode that just finished deliver every carton it was asked to?

    PORT NOTE: the single-agent callback thresholded the final step's reward, which was
    sound there because the terminal dropoff bonus dominated. In the multi-agent env the
    90/10 shared/individual reward split (the project roadmap, step 4) means each of the four agents
    sees a *different* final number for the same shared outcome, so a reward threshold is
    a proxy at best. Prefer an explicit signal when step 4 provides one.

    Recognised keys, in order: `is_success` (SB3's own convention, which also makes
    `rollout/success_rate` work for free), then `all_delivered`, then
    `remaining_resources == 0` - the last of which `_get_info()` already returns today.
    """
    if isinstance(info, dict):
        if "is_success" in info:
            return bool(info["is_success"])
        if "all_delivered" in info:
            return bool(info["all_delivered"])
        if "remaining_resources" in info:
            return int(info["remaining_resources"]) == 0
    return float(reward) >= SUCCESS_REWARD_THRESHOLD


class CurriculumCallback(BaseCallback):
    """
    Upgrades environment difficulty when the rolling success rate exceeds
    `target_success_rate` over the last `window_size` completed episodes.

    Also:
    - Logs `curriculum/difficulty_level` at every check so TensorBoard shows the exact
      timestep each difficulty transition happened.
    - Restarts the learning rate schedule whenever the difficulty increases, giving the
      agent a fresh learning burst to adapt to the new environment.

    PORT NOTE: the body is unchanged apart from the success test. It already iterates
    per vector-env slot, which is exactly what the project roadmap, step 6's "wrap the env so the 4
    robots look like 4 parallel single-agent envs" produces - 4x as many slots, same
    logic. Be aware that under that wrapper `window_size` counts robot-episodes, not
    world-episodes.
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
        # with *its own* env's reward and info - indexing rewards[0] inside this loop
        # credited env 0's outcome to all 16 workers.
        dones = self.locals["dones"]
        infos = self.locals.get("infos") or [{}] * len(dones)
        for done, reward, info in zip(dones, self.locals["rewards"], infos):
            if done:
                self.delivery_history.append(1 if _episode_succeeded(info, reward) else 0)

        if self.n_calls % self.check_freq != 0:
            return True

        current_level = self.training_env.get_attr("difficulty_level")[0]
        self.logger.record("curriculum/difficulty_level", current_level)
        self.logger.record("curriculum/target_cartons", CURRICULUM_CARTONS[current_level])

        if len(self.delivery_history) < self.window_size:
            return True

        success_rate = sum(self.delivery_history) / self.window_size
        self.logger.record("curriculum/success_rate", success_rate)

        if success_rate >= self.target_success_rate and current_level < MAX_DIFFICULTY_LEVEL:
            new_level = current_level + 1
            print(
                f"\n[Curriculum] Step {self.num_timesteps:,} | "
                f"Success rate {success_rate*100:.1f}% >= {self.target_success_rate*100:.0f}% | "
                f"Upgrading difficulty: Level {current_level} -> Level {new_level} "
                f"({CURRICULUM_CARTONS[current_level]} -> {CURRICULUM_CARTONS[new_level]} cartons)\n"
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
# This branch runs on Python 3.14, so that is not hypothetical - anything saved by an
# older interpreter must be loaded through these stand-ins.
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


def make_env(difficulty_level: int = 1, obs_dim: int = DEFAULT_OBS_DIM, seed: int | None = None):
    """
    Env factory for SubprocVecEnv.

    `obs_dim` is explicit on purpose: it is baked into the policy's input layer, so a
    model trained at one width cannot be loaded into an env built at another. env.py
    rejects any value but the pinned one, so passing it is an assertion, not a knob.

    PORT NOTE: `seed` is new. The single-agent factory left seeding to SB3, but this
    env's world generation is driven by the module-level `random` / `np.random` state
    that only `reset(seed=...)` touches (the project notes, Conventions), so without a distinct
    seed per worker every SubprocVecEnv process would regenerate identical warehouses.

    NOTE: this returns the raw joint multi-agent env. the project roadmap, step 6 calls for a
    wrapper presenting the 4 robots as 4 single-agent slots sharing one policy - build
    that wrapper on top of this factory, not inside it, so the greedy baseline (step 5)
    and the evaluation harness can still see the real joint env.
    """
    def _init():
        env = HiveMindMultiAgentEnv(
            render_mode=None, difficulty_level=difficulty_level, obs_dim=obs_dim
        )
        if seed is not None:
            env.reset(seed=seed)
        return env
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
    """
    PORT NOTE: the cap is now a number of *worlds*, and each world carries 4 robots. Under
    step 6's wrapper the effective batch is 4x this, so the single-agent branch's habit of
    running 16 workers becomes 64 robot-streams here. Start lower.
    """
    return min(cap, os.cpu_count() or 4)
