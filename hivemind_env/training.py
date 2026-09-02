"""
Shared training scaffolding: learning-rate schedules, the curriculum callback, the
communication diagnostics, checkpoint loading and the device probe.

One copy, imported by train.py and the evaluation scripts, so a fix lands everywhere at
once. The single-agent branch this was ported from kept three byte-identical copies and
carried the same learning-rate bug in two of them.
"""
import os
from collections import deque
from typing import Callable

import numpy as np
import torch

from hivemind_env.env import MSG_SILENT, MSG_TOKENS, max_steps_for

from stable_baselines3.common.callbacks import BaseCallback

NUM_AGENTS = 4

# The ladder EVALUATION reports against. The callback sets `num_cartons` on every env
# directly; it used to set `difficulty_level`, which the world stored and never read, so
# promotion was a no-op that only wrote a number to TensorBoard.
CURRICULUM_CARTONS = {1: 4, 2: 8, 3: 12}

# The ladder TRAINING walks, which is finer than the one evaluation reports against.
#
# WHY IT STARTS AT ONE CARTON
#
# `success_rate` is an AND over four robots: an episode terminates only when EVERY
# carton is delivered. At 4 cartons that conjunction is so unlikely under an untrained
# policy that it never fires - measured over 15 random episodes at each level:
#
#     cartons   random completions   delivered/ep
#        1          5/15  (33%)          0.33
#        4          0/15  ( 0%)          ~0.5 of 4
#
# So through 5M + 409k + 153k + 215k steps of training, **the learner had never once
# observed the +100 completion bonus or the +50 makespan bonus** - the two largest
# terms in the entire reward table. Its value function had no data on terminal states,
# because it had never reached one. Every run plateaued with `ep_len_mean` pinned at
# the cap, which is exactly what that looks like from outside.
#
# One carton fixes the conjunction: any single robot delivering ends the episode, so
# terminal reward becomes reachable by accident and the critic gets something to fit.
# 4 -> 8 -> 12 is still the reported curriculum (roadmap step 8); 1 and 2 are a ramp
# onto it, not a change to it.
TRAINING_CURRICULUM = {1: 1, 2: 2, 3: 4, 4: 8, 5: 12}
MAX_TRAINING_LEVEL = max(TRAINING_CURRICULUM)

# A fallback only - `_episode_succeeded` prefers `info["is_success"]`. There is no single
# terminal reward here that means "the job is done", but a completing episode's final
# step measures ~+143 while a non-completing one sits near zero, so 50.0 has room on both
# sides.
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

    Prefers an explicit signal over a reward threshold: the 90/10 split gives each of the
    four agents a different final number for the same shared outcome, so a threshold is a
    proxy at best. Keys in order: `is_success` (SB3's convention, which also makes
    `rollout/success_rate` work for free), `all_delivered`, `remaining_resources == 0`.
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
    Promotes along TRAINING_CURRICULUM when the rolling success rate over the last
    `window_size` completed episodes exceeds `target_success_rate`, logging the level at
    every check so TensorBoard shows exactly when each transition happened, and
    restarting the learning-rate schedule on promotion.

    Under the shared-policy wrapper `window_size` counts robot-episodes, not
    world-episodes - four slots per world all report the same outcome.
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
        self.logger.record("curriculum/target_cartons", TRAINING_CURRICULUM[current_level])

        if len(self.delivery_history) < self.window_size:
            return True

        success_rate = sum(self.delivery_history) / self.window_size
        self.logger.record("curriculum/success_rate", success_rate)

        if success_rate >= self.target_success_rate and current_level < MAX_TRAINING_LEVEL:
            new_level = current_level + 1
            print(
                f"\n[Curriculum] Step {self.num_timesteps:,} | "
                f"Success rate {success_rate*100:.1f}% >= {self.target_success_rate*100:.0f}% | "
                f"Upgrading difficulty: Level {current_level} -> Level {new_level} "
                f"({TRAINING_CURRICULUM[current_level]} -> {TRAINING_CURRICULUM[new_level]} cartons)\n"
            )
            self.training_env.set_attr("difficulty_level", new_level)
            # This is the line that makes promotion real.
            self.training_env.set_attr("num_cartons", TRAINING_CURRICULUM[new_level])
            # And the cap has to grow with it, or the harder level is handed an episode
            # budget sized for the easier one. Both take effect at the next reset.
            self.training_env.set_attr(
                "max_steps", max_steps_for(TRAINING_CURRICULUM[new_level])
            )
            self.delivery_history.clear()

            if self.reset_lr_on_promotion:
                progress = self.model._current_progress_remaining
                self.model.lr_schedule = restart_schedule(self.initial_lr, progress)
                print(f"[Curriculum] Learning rate schedule restarted at {self.initial_lr:.0e}")

        return True


class MessageStatsCallback(BaseCallback):
    """
    Communication diagnostics on the TensorBoard curves (roadmap steps 7 and 8).

    WHY THIS IS NOT OPTIONAL DECORATION

    A communicating run and a silent one both produce an `ep_rew_mean` curve, and this
    project has already misread three of those. The specific failure to watch for here
    is a policy that emits tokens which mean nothing - the reward curve looks identical
    whether the channel carries a protocol or noise, because the movement policy can
    carry the whole task on its own. What separates them is measurable and cheap:

        comms/token_entropy_bits    H(M) over the pooled marginal, max log2(16) = 4.0
        comms/tokens_used           symbols holding at least 1% of the mass
        comms/top_token_share       mass on the single most common symbol
        comms/mi_carrying_bits      I(M ; am I carrying?), max 1.0

    READ THEM IN PAIRS, NOT SEPARATELY. High entropy alone is not evidence of a
    protocol - PPO's entropy bonus actively pushes the token head toward uniform, so a
    policy that ignores the channel entirely will sit near 4.0 bits, which is the
    maximum. That is the trap this callback exists to expose rather than create:

        H high,  MI ~ 0     the entropy bonus talking. Tokens are noise.
        H low,   MI ~ 0     collapsed to one symbol. Silence with extra steps.
        H mid,   MI > 0     symbols are conditioned on the world. A protocol.

    `mi_carrying_bits` is a floor, not a measure of the protocol's content: carrying is
    the one piece of speaker state cheap enough to pair with every token inside the
    training loop. A protocol about *which carton I am claiming* would score near zero
    here and still be real. Zero MI is therefore not proof of failure; positive MI is
    proof that something is being encoded. The full measurement - mutual information
    against several state variables, plus the intervention tests that show the
    listeners actually act on what they hear - is scripts/analyse_messages.py, and that
    is what a result should be reported from.

    Silent under comms=False: every token is MSG_SILENT, nothing is recorded, nothing
    is logged.
    """

    def __init__(self, check_freq: int = 2048, window_size: int = 20000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.tokens = deque(maxlen=window_size)
        self.carrying = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        for info in (self.locals.get("infos") or []):
            token = info.get("message_token", MSG_SILENT)
            if token != MSG_SILENT:
                self.tokens.append(int(token))
                self.carrying.append(bool(info.get("is_carrying", False)))

        if self.n_calls % self.check_freq != 0 or not self.tokens:
            return True

        tokens = np.asarray(self.tokens)
        carrying = np.asarray(self.carrying)

        counts = np.bincount(tokens, minlength=MSG_TOKENS).astype(float)
        p = counts / counts.sum()
        self.logger.record("comms/token_entropy_bits", entropy_bits(p))
        self.logger.record("comms/tokens_used", int((p >= 0.01).sum()))
        self.logger.record("comms/top_token_share", float(p.max()))

        # I(M ; C) = H(M) - H(M | C), with C the carrying flag. Computed from the same
        # window so the two curves are always directly comparable.
        h_cond = 0.0
        for value in (False, True):
            mask = carrying == value
            weight = mask.mean()
            if weight <= 0.0:
                continue
            sub = np.bincount(tokens[mask], minlength=MSG_TOKENS).astype(float)
            h_cond += weight * entropy_bits(sub / sub.sum())
        self.logger.record("comms/mi_carrying_bits",
                           max(0.0, entropy_bits(p) - h_cond))
        return True


def entropy_bits(x, counts=False) -> float:
    """
    Shannon entropy in bits. Accepts a probability vector, a count vector
    (`counts=True`) or a sequence of labels (also `counts=True` after bincount).

    The single implementation: it had drifted into three files, once per script that
    needed to report message entropy. abs() only turns -0.0 into 0.0 for a degenerate
    distribution; entropy is never negative.
    """
    p = np.asarray(x, dtype=float)
    if counts or p.sum() > 1.0 + 1e-9:
        total = p.sum()
        if total <= 0:
            return 0.0
        p = p / total
    p = p[p > 0.0]
    return abs(float(-(p * np.log2(p)).sum()))


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
    A number of WORLDS, each carrying 4 robots - so the effective batch is 4x this and
    the single-agent branch's habit of 16 workers would be 64 robot-streams. Start lower.
    """
    return min(cap, os.cpu_count() or 4)
