"""
Presents N four-robot warehouses to Stable-Baselines3 as 4N single-agent environments
that share one policy.

This is CLAUDE.md roadmap step 6's "cheap option first":

    Stable-Baselines3 does not do MAPPO out of the box. Try the cheap option first -
    wrap the env so the 4 robots look like 4 parallel single-agent envs sharing a
    policy. Only reach for a dedicated multi-agent PPO implementation if that plateaus.

WHAT THIS IS, AND WHAT IT IS NOT

It IS parameter sharing: one actor and one critic, applied to each robot's own
observation, trained on the pooled experience of every robot in every world. All four
robots in a world act simultaneously on the same physics step, so their interactions -
collisions, racing for the same carton, the shared 90% of the reward - are all real.

It is NOT a centralised critic in the MAPPO sense. The critic sees one robot's
observation, not the joint state. That is a genuine simplification and it is worth
being clear about why it is tolerable here rather than pretending otherwise: this
environment's observation is already close to global. Every robot sees all four poses,
all twelve carton statuses and positions, and the elapsed time. What it does not see is
the other robots' LiDAR returns and (later) the raw message each is about to send. So
the value function is estimating from something much closer to full state than a
typical partially-observed multi-agent setup would give it.

If step 6 plateaus below the greedy baseline, this is the first thing to replace, and
the honest way to do it is a real CTDE implementation with an asymmetric critic - not
more tuning here.

WHY THE SHARED REWARD DOES NOT DOUBLE-COUNT

Each robot's reward already contains its 90% share of the shared term, so pooling four
slots per world means the shared component appears four times in the batch. That is
correct for parameter sharing: each robot genuinely did receive that reward. It does
mean the effective batch is 4x the number of worlds, so `n_steps` counts robot-steps
rather than world-steps - see the note in train.py about sizing rollouts.

TERMINATION IS PER WORLD, NOT PER ROBOT

A world terminates when all twelve cartons are delivered and truncates at the step
limit. There is no such thing as one robot finishing early, so all four slots of a
world go `done` on the same step and reset together. SB3's auto-reset contract is
honoured per slot: the returned observation is the first of the new episode, and the
final observation of the old one is placed in `info["terminal_observation"]`.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hivemind_env.env import DEFAULT_OBS_DIM, NUM_AGENTS, HiveMindMultiAgentEnv


class HiveMindSharedPolicyVecEnv(VecEnv):
    """
    `num_worlds` warehouses -> `num_worlds * NUM_AGENTS` policy-facing slots.

    Slot ordering is world-major: slots [0..3] are world 0's robots 0..3, slots [4..7]
    are world 1's, and so on. `world_of(slot)` and `agent_of(slot)` make that explicit
    rather than leaving callers to rediscover the arithmetic.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_worlds: int = 1, difficulty_level: int = 1,
                 obs_dim: int = DEFAULT_OBS_DIM, seed: int | None = None, **env_kwargs):
        self.num_worlds = int(num_worlds)
        if self.num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")

        self.envs = [
            HiveMindMultiAgentEnv(render_mode=None, difficulty_level=difficulty_level,
                                  obs_dim=obs_dim, **env_kwargs)
            for _ in range(self.num_worlds)
        ]

        # Per-world seeds. World generation runs off module-level random state that only
        # reset(seed=...) touches, so without distinct seeds every world would build the
        # identical warehouse and the extra parallelism would buy nothing but throughput.
        self._base_seed = seed
        self._episode_counter = [0] * self.num_worlds

        single_obs = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        single_act = spaces.Discrete(int(self.envs[0].action_space.nvec[0]))
        super().__init__(self.num_worlds * NUM_AGENTS, single_obs, single_act)

        self._obs = np.zeros((self.num_envs, obs_dim), dtype=np.float32)
        self._actions: np.ndarray | None = None

    # -- slot <-> (world, agent) ------------------------------------------------
    def world_of(self, slot: int) -> int:
        return slot // NUM_AGENTS

    def agent_of(self, slot: int) -> int:
        return slot % NUM_AGENTS

    def _slots(self, world: int):
        start = world * NUM_AGENTS
        return range(start, start + NUM_AGENTS)

    def _world_seed(self, world: int):
        if self._base_seed is None:
            return None
        # Distinct per world AND per episode, so a world does not replay one warehouse
        # for the whole run.
        return self._base_seed + 1000 * world + self._episode_counter[world]

    # -- VecEnv API -------------------------------------------------------------
    def reset(self):
        for w, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self._world_seed(w))
            self._episode_counter[w] += 1
            for a, slot in enumerate(self._slots(w)):
                self._obs[slot] = obs[a]
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for w, env in enumerate(self.envs):
            slots = list(self._slots(w))
            joint = self._actions[slots]
            obs, rew, terminated, truncated, info = env.step(joint)
            done = bool(terminated or truncated)

            for a, slot in enumerate(slots):
                rewards[slot] = rew[a]
                dones[slot] = done
                # Per-slot info: the world's shared facts plus this robot's own events,
                # so a callback reading infos[slot] does not have to know the layout.
                infos[slot] = {
                    "world": w,
                    "agent": a,
                    "is_success": info["is_success"],
                    "all_delivered": info["all_delivered"],
                    "delivered": info["delivered"],
                    "remaining_resources": info["remaining_resources"],
                    "collisions": info["collisions"],
                    "picked_up": info["pickups"][a],
                    "delivered_by_me": info["deliveries"][a],
                    "invalid_action": info["invalid_actions"][a],
                }

            if done:
                # SB3 auto-reset contract: hand back the first observation of the new
                # episode and stash the last one of the old episode in the info dict.
                # Getting this wrong is silent - it just bootstraps values across an
                # episode boundary and slightly poisons the advantage estimates.
                for a, slot in enumerate(slots):
                    infos[slot]["terminal_observation"] = obs[a].copy()
                    infos[slot]["TimeLimit.truncated"] = bool(truncated and not terminated)
                new_obs, _ = env.reset(seed=self._world_seed(w))
                self._episode_counter[w] += 1
                for a, slot in enumerate(slots):
                    self._obs[slot] = new_obs[a]
            else:
                for a, slot in enumerate(slots):
                    self._obs[slot] = obs[a]

        return self._obs.copy(), rewards, dones, infos

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                # close() is not idempotent on the underlying env and a partially torn
                # down vec env must not mask the real error with a disconnect error.
                pass

    # -- attribute plumbing -----------------------------------------------------
    # SB3 addresses these per slot, but the attributes live on the world. Reading maps
    # slot -> world; writing applies to each world a selected slot belongs to, once.
    def _worlds_for(self, indices):
        if indices is None:
            return list(range(self.num_worlds))
        if isinstance(indices, int):
            indices = [indices]
        return sorted({self.world_of(i) for i in indices})

    def _slot_indices(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def get_attr(self, attr_name: str, indices=None) -> list:
        return [getattr(self.envs[self.world_of(i)], attr_name)
                for i in self._slot_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        for w in self._worlds_for(indices):
            setattr(self.envs[w], attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        return [getattr(self.envs[w], method_name)(*args, **kwargs)
                for w in self._worlds_for(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._slot_indices(indices))
