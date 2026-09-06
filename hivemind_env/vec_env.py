"""
Presents N four-robot warehouses to Stable-Baselines3 as 4N single-agent slots sharing
one policy, plus the slot bookkeeping `subproc_vec_env.py` reuses.

WHAT THIS IS, AND WHAT IT IS NOT

It is parameter sharing: one actor and one critic applied to each robot's own
observation, trained on the pooled experience of every robot in every world. All four
robots act on the same physics step, so collisions, racing for the same carton and the
shared 90% of the reward are all real.

It is NOT a centralised critic. The critic sees one robot's observation, not the joint
state. That is tolerable here only because the observation is already close to global -
every robot sees all four poses, all twelve carton statuses and positions - so what the
value function misses is the other robots' LiDAR and whatever they are about to say. If
a run plateaus below the greedy baseline, replacing this with a real CTDE critic is the
honest next move, ahead of more tuning. With comms on it is the first suspect, because
the critic cannot see the message that is about to change three other robots' actions.

Two consequences worth stating:

  - Each robot's reward already contains its 90% share of the shared term, so pooling
    four slots per world puts that component in the batch four times. That is correct
    for parameter sharing - each robot did receive it - but it means `n_steps` counts
    robot-steps, not world-steps.
  - Termination is per world. All four slots go `done` together and reset together;
    SB3's auto-reset contract is honoured per slot, with the old episode's final
    observation in `info["terminal_observation"]`.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hivemind_env.env import (
    DEFAULT_OBS_DIM,
    MOVE_ACTIONS,
    MSG_TOKENS,
    NUM_AGENTS,
    HiveMindMultiAgentEnv,
)


def single_slot_spaces(obs_dim: int, comms: bool):
    """
    One policy slot's (observation, action) spaces.

    Both backends call this rather than restating it: the action space is baked into
    saved weights exactly as the observation width is, and a policy trained against one
    backend has to load against the other.
    """
    obs = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act = (spaces.MultiDiscrete([MOVE_ACTIONS, MSG_TOKENS]) if comms
           else spaces.Discrete(MOVE_ACTIONS))
    return obs, act


def slot_infos(info, world):
    """
    A world's info dict, split into the per-robot view a callback reads.

    Deliberately narrow: the env's full info carries two LiDAR arrays per robot (576
    floats per world-step) that exist for diagnostics, and under the subprocess backend
    every field here is pickled and shipped down a pipe on every single step.
    """
    return [
        {
            "world": world,
            "agent": a,
            "is_success": info["is_success"],
            "all_delivered": info["all_delivered"],
            "delivered": info["delivered"],
            # Paired with "delivered" so EpisodeStatsCallback can report a
            # fraction, which stays comparable across curriculum levels.
            "active_cartons": info["active_cartons"],
            "remaining_resources": info["remaining_resources"],
            "collisions": info["collisions"],
            "picked_up": info["pickups"][a],
            "delivered_by_me": info["deliveries"][a],
            "invalid_action": info["invalid_actions"][a],
            # The pair MessageStatsCallback needs for I(token; carrying). The token is
            # -1 with comms off, which is what makes that callback a no-op then.
            "message_token": info["message_tokens"][a],
            "is_carrying": bool(info["is_carrying"][a]),
        }
        for a in range(NUM_AGENTS)
    ]


class SlotLayout:
    """
    Slot <-> (world, agent) arithmetic, shared by both backends.

    Slot ordering is world-major: slots 0-3 are world 0's robots, 4-7 are world 1's.
    SB3 addresses attributes per slot but they live on the world, so reads map
    slot -> world and writes apply once per world any selected slot belongs to.
    """

    num_worlds: int

    def world_of(self, slot: int) -> int:
        return slot // NUM_AGENTS

    def agent_of(self, slot: int) -> int:
        return slot % NUM_AGENTS

    def _slots(self, world: int):
        start = world * NUM_AGENTS
        return range(start, start + NUM_AGENTS)

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

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._slot_indices(indices))

    def _fan_out_masks(self, per_world):
        """
        `action_masks()` returns one (NUM_AGENTS, n) array per world, but MaskablePPO
        calls `env_method("action_masks")` and stacks the result expecting one row per
        SLOT. Without this the stack is (num_worlds, 4, n) and MaskablePPO silently
        misreads it as a batch of the wrong size.
        """
        return [row for world_masks in per_world for row in np.asarray(world_masks)]

    def _reshape_actions(self, actions):
        """(slots,) without comms, (slots, 2) with - column 1 being the token."""
        actions = np.asarray(actions, dtype=np.int64)
        return (actions.reshape(self.num_envs, 2) if self.comms
                else actions.reshape(self.num_envs))


class HiveMindSharedPolicyVecEnv(SlotLayout, VecEnv):
    """`num_worlds` warehouses, stepped sequentially in this process."""

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

        # World generation runs off the module-level random state that only
        # reset(seed=...) touches, so without distinct per-world seeds every world would
        # build the identical warehouse and the parallelism would buy only throughput.
        self._base_seed = seed
        self._episode_counter = [0] * self.num_worlds

        self.comms = bool(env_kwargs.get("comms", False))
        super().__init__(self.num_worlds * NUM_AGENTS,
                         *single_slot_spaces(obs_dim, self.comms))

        self._obs = np.zeros((self.num_envs, obs_dim), dtype=np.float32)
        self._actions: np.ndarray | None = None

    def _world_seed(self, world: int):
        if self._base_seed is None:
            return None
        # Distinct per world AND per episode, so a world does not replay one warehouse.
        return self._base_seed + 1000 * world + self._episode_counter[world]

    def reset(self):
        for w, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self._world_seed(w))
            self._episode_counter[w] += 1
            for a, slot in enumerate(self._slots(w)):
                self._obs[slot] = obs[a]
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = self._reshape_actions(actions)

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for w, env in enumerate(self.envs):
            slots = list(self._slots(w))
            obs, rew, terminated, truncated, info = env.step(self._actions[slots])
            done = bool(terminated or truncated)
            world_infos = slot_infos(info, w)

            for a, slot in enumerate(slots):
                rewards[slot] = rew[a]
                dones[slot] = done
                infos[slot] = world_infos[a]

            if done:
                # Getting the auto-reset contract wrong is silent: it bootstraps values
                # across an episode boundary and quietly poisons the advantages.
                for a, slot in enumerate(slots):
                    infos[slot]["terminal_observation"] = obs[a].copy()
                    infos[slot]["TimeLimit.truncated"] = bool(truncated and not terminated)
                obs, _ = env.reset(seed=self._world_seed(w))
                self._episode_counter[w] += 1
            for a, slot in enumerate(slots):
                self._obs[slot] = obs[a]

        return self._obs.copy(), rewards, dones, infos

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                # A partially torn down vec env must not mask the real error with a
                # disconnect error.
                pass

    def get_attr(self, attr_name: str, indices=None) -> list:
        return [getattr(self.envs[self.world_of(i)], attr_name)
                for i in self._slot_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        for w in self._worlds_for(indices):
            setattr(self.envs[w], attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        out = [getattr(self.envs[w], method_name)(*args, **kwargs)
               for w in self._worlds_for(indices)]
        return self._fan_out_masks(out) if method_name == "action_masks" else out
