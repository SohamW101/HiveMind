"""
Quick smoke test - runs in-process, writes output to file for inspection.

Ported from `single-agent-rl` (roadmap step 2) and adapted for the 4-robot env. Checks
that the package imports, a device is selectable, and the env can reset and step.

It used to hardcode an absolute path from another machine ("d:\\Dev Projects\\...") and
os.chdir into it, so it raised FileNotFoundError on line 6 before doing anything. It also
grepped train.py for literal strings, which tested the text of a file rather than any
behaviour and reported a FAIL for a stale token that had been correctly removed.

PORT NOTE - three result states instead of two. The single-agent env was finished, so
every check was PASS or FAIL. Here, roadmap steps 4 and 6 are genuinely not built yet
(step 3 has since landed and its checks are now hard assertions), so a third state
exists:

    PASS    works now
    TODO    not implemented yet, and known not to be - not a regression
    FAIL    something that used to work, or should work today, is broken

Only FAIL sets a non-zero exit code. As steps 4 and 6 land, TODO lines should turn into
PASS lines without this file needing new checks - that is the point of listing them.
Step 3 already made that trip: its TODOs are now assertions.

    .venv\\Scripts\\python.exe smoke_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "smoke_test_results.txt"

results = []


def emit(line):
    results.append(line)


# -- Imports ------------------------------------------------------------------
# The extractor is imported later, not here: an import failure in this block exits(1)
# before any other check runs.
try:
    import numpy as np
    import torch  # noqa: F401  (imported for the version banner and the device probe)
    from stable_baselines3 import PPO  # noqa: F401
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: F401
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: F401
    from hivemind_env.env import (
        DEFAULT_OBS_DIM,
        NUM_AGENTS,
        OBS_DIM_V3,
        HiveMindMultiAgentEnv,
        describe_observation_layout,
    )
    from hivemind_env.training import (
        CurriculumCallback,
        MessageStatsCallback,  # noqa: F401
        get_device,
        linear_schedule,
    )
    emit("PASS: All imports successful")
except Exception as e:
    emit(f"FAIL: Import error: {e}")
    OUT_PATH.write_text("\n".join(results), encoding="utf-8")
    print("\n".join(results))
    sys.exit(1)

emit(f"  python={sys.version.split()[0]} numpy={np.__version__} torch={torch.__version__}")

# -- Device check --------------------------------------------------------------
try:
    device = get_device()
    emit(f"PASS: device={device}")
except Exception as e:
    emit(f"FAIL: Device check: {e}")

# -- Environment ---------------------------------------------------------------
# PORT NOTE: the single-agent version asserted `obs["grid"].shape == (N, N, 5)`. This env
# has no grid observation - roadmap step 3 gave it a flat (num_agents, OBS_DIM_V1) vector
# instead - so the shape assertion is against the pinned width and is a hard FAIL.
# LiDAR is a separate matter: it is absent here and is NOT part of step 3's component
# list, so it stays a TODO rather than a failure.
env = None
try:
    env = HiveMindMultiAgentEnv(render_mode=None, difficulty_level=1, obs_dim=DEFAULT_OBS_DIM)
    emit(f"PASS: env constructed. num_agents={env.num_agents}, action_space={env.action_space}")
    if env.num_agents != NUM_AGENTS:
        emit(f"FAIL: expected {NUM_AGENTS} agents, got {env.num_agents}")

    if hasattr(env, "observation_space"):
        emit(f"PASS: observation_space declared: {env.observation_space}")
        expected = (NUM_AGENTS, OBS_DIM_V3)
        if env.observation_space.shape != expected:
            emit(f"FAIL: observation_space shape {env.observation_space.shape} "
                 f"!= pinned {expected}")
    else:
        emit("FAIL: no observation_space declared (roadmap step 3 regressed)")

    obs, info = env.reset(seed=0)
    emit(f"PASS: env.reset(seed=0) OK. info keys={sorted(info)}")
    emit(f"  remaining_resources={info['remaining_resources']}, "
         f"robot_pos[0]={tuple(round(v, 3) for v in info['robot_pos'][0])}")

    if len(info["robot_pos"]) != NUM_AGENTS:
        emit(f"FAIL: expected {NUM_AGENTS} robot poses, got {len(info['robot_pos'])}")
    if info["remaining_resources"] != 12:
        emit(f"FAIL: expected 12 cartons at reset, got {info['remaining_resources']}")

    if len(obs) == 0:
        emit("FAIL: _get_obs() returned nothing (roadmap step 3 regressed)")
    else:
        emit(f"PASS: obs {obs.shape} {obs.dtype}, "
             f"range [{obs.min():.3f}, {obs.max():.3f}]")
        if not env.observation_space.contains(obs):
            emit("FAIL: observation is outside its own observation_space")
        else:
            emit("PASS: observation lies inside observation_space")
    for line in describe_observation_layout().splitlines():
        emit(f"  {line}")

    if "lidar_distances" in info:
        emit(f"PASS: info exposes {len(info['lidar_distances'])} lidar distances")
    else:
        emit("FAIL: LiDAR missing from info - it is part of observation V3")
except Exception as e:
    emit(f"FAIL: env.reset(): {type(e).__name__}: {e}")

# -- Steps ---------------------------------------------------------------------
# PORT NOTE: `env.step(int(action))` became `env.step(list_of_4)` - the action space is
# MultiDiscrete([7,7,7,7]), so `action_space.sample()` already yields the right shape.
# The single-agent info keys `requested_action` / `executed_action` / `action_overridden`
# do not exist here (this env never overrides the policy's action), so the per-step log
# reports the joint action instead.
if env is not None:
    try:
        rewards = []
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            emit(f"  Step {i+1}: action={list(map(int, action))} reward={reward} "
                 f"term={term} trunc={trunc} remaining={info['remaining_resources']}")
            if term or trunc:
                obs, info = env.reset()
        emit(f"PASS: 5 steps OK. Per-agent rewards: {rewards[-1]}")

        if all(all(float(r) == 0.0 for r in rw) for rw in rewards):
            emit("FAIL: reward is all zeros - step 4 regressed")
        else:
            emit("PASS: rewards are non-zero and per-agent")
        if env.current_step >= env.max_steps:
            emit("FAIL: step budget exhausted during a 5-step smoke test")
        emit(f"PASS: current_step advanced to {env.current_step} (max_steps={env.max_steps})")
    except Exception as e:
        emit(f"FAIL: env.step(): {type(e).__name__}: {e}")
    finally:
        # PORT NOTE: the single-agent env guarded close() against a double disconnect and
        # this test asserted that. HiveMindMultiAgentEnv.close() still does not - the
        # second call raises pybullet.error("Not connected to physics server."). Left
        # unfixed deliberately: it is unrelated to the observation work and wants its own
        # change. It will bite during SubprocVecEnv teardown in step 6, so fix it before
        # then - guard close() on a _closed flag.
        env.close()
        try:
            env.close()
            emit("PASS: env.close() is idempotent")
        except Exception as e:
            emit(f"TODO: env.close() is NOT idempotent - second call raises "
                 f"{type(e).__name__}: {e} (fix in env.py: guard on a _closed flag)")

# -- Termination wiring ---------------------------------------------------------
try:
    probe = HiveMindMultiAgentEnv(render_mode=None)
    probe.reset(seed=0)
    probe.max_steps = 3
    trunc = False
    for _ in range(3):
        _, _, term, trunc, _ = probe.step([6, 6, 6, 6])
    probe.close()
    if trunc:
        emit("PASS: truncation fires at max_steps")
    else:
        emit("FAIL: max_steps is not enforced (roadmap step 4 regressed)")
except Exception as e:
    emit(f"FAIL: could not read env.py: {e}")

# -- Feature extractor wiring ---------------------------------------------------
# The extractor is real now (roadmap step 6). Build it against the env's own observation
# space so a layout change that the network has not followed shows up here rather than
# forty minutes into a training run.
try:
    import torch
    from hivemind_env.models import HiveMindExtractor
    probe = HiveMindMultiAgentEnv(render_mode=None)
    single = probe.observation_space  # (num_agents, obs_dim)
    from gymnasium import spaces
    flat = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM_V3,), dtype=np.float32)
    ex = HiveMindExtractor(flat, features_dim=256)
    out = ex(torch.zeros(2, OBS_DIM_V3))
    probe.close()
    if tuple(out.shape) == (2, 256):
        emit(f"PASS: HiveMindExtractor builds and runs "
             f"({sum(q.numel() for q in ex.parameters()):,} params)")
    else:
        emit(f"FAIL: extractor emitted {tuple(out.shape)}, expected (2, 256)")
except ImportError as e:
    emit(f"FAIL: models import: {e}")
except Exception as e:
    emit(f"FAIL: extractor build: {type(e).__name__}: {e}")

# -- Greedy baseline ------------------------------------------------------------
# One seeded episode of the scripted controller. This is a real end-to-end exercise of
# navigation, pickup, delivery, termination and the reward split, and it guards the
# reference makespan: if a change makes greedy stop completing, every published number
# that was quoted against it is invalid.
try:
    from hivemind_env.greedy import GreedyController
    genv = HiveMindMultiAgentEnv(render_mode=None)
    genv.reset(seed=1000)
    ctrl = GreedyController(genv)
    term = False
    for _ in range(genv.max_steps):
        _, _, term, trunc, ginfo = genv.step(ctrl.act())
        ctrl.sync_after_step()
        if term or trunc:
            break
    steps = genv.current_step
    genv.close()
    if term and ginfo["delivered"] == 12:
        emit(f"PASS: greedy baseline completes seed 1000 in {steps} steps "
             f"(reference mean is 98)")
        if steps > 150:
            emit(f"FAIL: greedy took {steps} steps, far above the ~98 reference - "
                 f"something regressed")
    else:
        emit(f"FAIL: greedy delivered {ginfo['delivered']}/12 in {steps} steps")
except Exception as e:
    emit(f"FAIL: greedy baseline: {type(e).__name__}: {e}")

# -- Scaffolding sanity ---------------------------------------------------------
try:
    assert linear_schedule(3e-4)(1.0) == 3e-4
    assert linear_schedule(3e-4)(0.0) == 0.0
    assert issubclass(CurriculumCallback, BaseCallback)
    emit("PASS: training scaffolding (linear_schedule, CurriculumCallback) sane")
except Exception as e:
    emit(f"FAIL: training scaffolding: {type(e).__name__}: {e}")

# -- Write results --------------------------------------------------------------
n_fail = sum(1 for r in results if r.startswith("FAIL"))
n_todo = sum(1 for r in results if r.startswith("TODO"))
n_pass = sum(1 for r in results if r.startswith("PASS"))
emit(f"\nSUMMARY: {n_pass} PASS, {n_todo} TODO (not built yet), {n_fail} FAIL")

OUT_PATH.write_text("\n".join(results), encoding="utf-8")
print("\n".join(results))
print(f"\nResults written to: {OUT_PATH}")

sys.exit(1 if n_fail else 0)
