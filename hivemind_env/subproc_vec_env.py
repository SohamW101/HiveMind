"""
Same slot layout as `HiveMindSharedPolicyVecEnv`, one OS process per warehouse.

Training is PyBullet-bound, not GPU-bound. Measured over a 1,024 robot-step rollout on
the in-process backend: env stepping 13.8 s (95%), policy forward 0.2 s, PPO update
0.6 s. So the lever is more cores, and the in-process version cannot use them - it steps
its worlds one after another in a single Python process, leaving 15 of 16 cores idle.

A world-step costs ~55-60 ms of physics wherever it runs, so N worlds in N processes
still cost ~60 ms of wall clock rather than N x 60. Expect sub-linear scaling - pipe
traffic, spawn cost and memory bandwidth all take a cut - but it is the difference
between a run measured in hours and one measured overnight.

Uses the "spawn" start method explicitly rather than the platform default, so the worker
function is at module level and everything it receives is picklable. That also means
anything constructing this class must sit behind an `if __name__ == "__main__":` guard.

Use the in-process backend to debug: a traceback inside a worker is much harder to read.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hivemind_env.env import DEFAULT_OBS_DIM, NUM_AGENTS, HiveMindMultiAgentEnv
from hivemind_env.vec_env import SlotLayout, single_slot_spaces, slot_infos


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
                infos = slot_infos(info, world)
                if done:
                    # SB3's auto-reset contract, applied inside the worker so the parent
                    # never has to know an episode boundary happened.
                    for a in range(NUM_AGENTS):
                        infos[a]["terminal_observation"] = obs[a].copy()
                        infos[a]["TimeLimit.truncated"] = bool(truncated and not terminated)
                    obs, _ = env.reset(seed=next_seed())
                remote.send((obs, np.asarray(rew, dtype=np.float32), done, infos))

            elif cmd == "reset":
                remote.send(env.reset(seed=next_seed())[0])

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


class HiveMindSubprocVecEnv(SlotLayout, VecEnv):
    """Drop-in replacement for HiveMindSharedPolicyVecEnv with one process per world."""

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

        self.comms = bool(env_kwargs.get("comms", False))
        super().__init__(self.num_worlds * NUM_AGENTS,
                         *single_slot_spaces(obs_dim, self.comms))

        self._obs = np.zeros((self.num_envs, obs_dim), dtype=np.float32)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        for w, remote in enumerate(self.remotes):
            obs = remote.recv()
            for a, slot in enumerate(self._slots(w)):
                self._obs[slot] = obs[a]
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        actions = self._reshape_actions(actions)
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
        out = [self.remotes[w].recv() for w in worlds]
        return self._fan_out_masks(out) if method_name == "action_masks" else out

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
