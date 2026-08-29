"""
Subprocess-backed version of HiveMindSharedPolicyVecEnv - one OS process per warehouse.

WHY THIS EXISTS

Training is not GPU-bound, it is PyBullet-bound. Measured over a 1,024 robot-step
rollout on the in-process vec env:

    env stepping   13.8 s   95.0%     rigid-body physics, one core
    policy forward  0.2 s    1.3%
    PPO update      0.6 s    3.8%     the only part a GPU touches

So the lever is not a faster accelerator, it is more cores - and the in-process version
cannot use them, because it steps its worlds one after another in a single Python
process. On a 16-core machine that leaves 15 idle.

Each world here gets its own process. A world-step costs ~55-60 ms of physics wherever
it runs, so N worlds in N processes still cost ~60 ms wall-clock rather than N x 60 ms.
Expect sub-linear scaling in practice - pipe traffic, spawn cost and memory bandwidth
all take a cut - but the difference is a run measured in hours rather than overnight.

WHAT CROSSES THE PIPE

Only what training needs. The worker builds the compact per-slot info dicts itself and
sends those, rather than shipping the environment's full info back: that dict carries
two LiDAR arrays per robot (576 floats per world per step) which exist for diagnostics
and would otherwise be pickled, transferred and thrown away on every single step.

WINDOWS

Uses the "spawn" start method explicitly rather than the platform default. Spawn
re-imports this module in the child, so the worker function is defined at module level
and everything it receives is picklable. That also means anything constructing this
class must sit behind an `if __name__ == "__main__":` guard.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hivemind_env.env import DEFAULT_OBS_DIM, NUM_AGENTS, HiveMindMultiAgentEnv


def _slot_infos(info, world):
    """The per-robot view of a world's info. Mirrors the in-process wrapper exactly."""
    return [
        {
            "world": world,
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
        for a in range(NUM_AGENTS)
    ]


def _worker(remote, parent_remote, world, base_seed, env_kwargs):
    """One warehouse, driven over a pipe. Runs until told to close."""
    parent_remote.close()
    env = HiveMindMultiAgentEnv(render_mode=None, **env_kwargs)
    episode = 0

    def next_seed():
        nonlocal episode
        if base_seed is None:
            return None
        s = base_seed + 1000 * world + episode
        episode += 1
        return s

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                obs, rew, terminated, truncated, info = env.step(data)
                done = bool(terminated or truncated)
                infos = _slot_infos(info, world)
                if done:
                    # SB3's auto-reset contract, applied inside the worker so the parent
                    # never has to know an episode boundary happened.
                    for a in range(NUM_AGENTS):
                        infos[a]["terminal_observation"] = obs[a].copy()
                        infos[a]["TimeLimit.truncated"] = bool(truncated and not terminated)
                    obs, _ = env.reset(seed=next_seed())
                remote.send((obs, np.asarray(rew, dtype=np.float32), done, infos))

            elif cmd == "reset":
                obs, _ = env.reset(seed=next_seed())
                remote.send(obs)

            elif cmd == "get_attr":
                remote.send(getattr(env, data))

            elif cmd == "set_attr":
                setattr(env, data[0], data[1])
                remote.send(None)

            elif cmd == "env_method":
                name, args, kwargs = data
                remote.send(getattr(env, name)(*args, **kwargs))

            elif cmd == "close":
                remote.send(None)
                break

            else:
                raise RuntimeError(f"worker received unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()
        remote.close()


class HiveMindSubprocVecEnv(VecEnv):
    """
    Drop-in replacement for HiveMindSharedPolicyVecEnv with one process per world.

    Same slot layout - world-major, `NUM_AGENTS` slots per world - so a policy trained
    against one backend loads and runs against the other. Use the in-process version for
    debugging (a traceback in a worker is much harder to read) and this one to train.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_worlds: int = 4, difficulty_level: int = 1,
                 obs_dim: int = DEFAULT_OBS_DIM, seed: int | None = None,
                 start_method: str = "spawn", **env_kwargs):
        self.num_worlds = int(num_worlds)
        if self.num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
        self.waiting = False
        self.closed = False

        ctx = mp.get_context(start_method)
        worker_kwargs = dict(difficulty_level=difficulty_level, obs_dim=obs_dim,
                             **env_kwargs)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_worlds)]
        )
        self.processes = []
        for world, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, world, seed, worker_kwargs),
                daemon=True,   # workers must not outlive a crashed parent
            )
            proc.start()
            self.processes.append(proc)
            work_remote.close()

        single_obs = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        single_act = spaces.Discrete(7)
        super().__init__(self.num_worlds * NUM_AGENTS, single_obs, single_act)

        self._obs = np.zeros((self.num_envs, obs_dim), dtype=np.float32)

    # -- slot <-> (world, agent) ------------------------------------------------
    def world_of(self, slot: int) -> int:
        return slot // NUM_AGENTS

    def agent_of(self, slot: int) -> int:
        return slot % NUM_AGENTS

    def _slots(self, world: int):
        start = world * NUM_AGENTS
        return range(start, start + NUM_AGENTS)

    # -- VecEnv API -------------------------------------------------------------
    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        for w, remote in enumerate(self.remotes):
            obs = remote.recv()
            for a, slot in enumerate(self._slots(w)):
                self._obs[slot] = obs[a]
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        # Fire every world before reading any reply - that overlap is the whole point.
        for w, remote in enumerate(self.remotes):
            remote.send(("step", actions[list(self._slots(w))]))
        self.waiting = True

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for w, remote in enumerate(self.remotes):
            obs, rew, done, world_infos = remote.recv()
            for a, slot in enumerate(self._slots(w)):
                self._obs[slot] = obs[a]
                rewards[slot] = rew[a]
                dones[slot] = done
                infos[slot] = world_infos[a]
        self.waiting = False
        return self._obs.copy(), rewards, dones, infos

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    remote.recv()
                except EOFError:
                    pass
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.closed = True

    # -- attribute plumbing -----------------------------------------------------
    # SB3 addresses these per slot; the attributes live on the world.
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
        wanted = self._slot_indices(indices)
        cache = {}
        for w in sorted({self.world_of(i) for i in wanted}):
            self.remotes[w].send(("get_attr", attr_name))
            cache[w] = self.remotes[w].recv()
        return [cache[self.world_of(i)] for i in wanted]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        worlds = self._worlds_for(indices)
        for w in worlds:
            self.remotes[w].send(("set_attr", (attr_name, value)))
        for w in worlds:
            self.remotes[w].recv()

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        worlds = self._worlds_for(indices)
        for w in worlds:
            self.remotes[w].send(("env_method", (method_name, args, kwargs)))
        return [self.remotes[w].recv() for w in worlds]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._slot_indices(indices))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
