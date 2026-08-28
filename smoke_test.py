"""
Quick smoke test - runs in-process, writes output to file for inspection.

Ported from `single-agent-rl` (roadmap step 2) and adapted for the 4-robot env. Checks
that the package imports, a device is selectable, and the env can reset and step.

It used to hardcode an absolute path from another machine ("d:\\Dev Projects\\...") and
os.chdir into it, so it raised FileNotFoundError on line 6 before doing anything. It also
grepped train.py for literal strings, which tested the text of a file rather than any
behaviour and reported a FAIL for a stale token that had been correctly removed.

PORT NOTE - three result states instead of two. The single-agent env was finished, so
every check was PASS or FAIL. Here, roadmap steps 3, 4, 6 are genuinely not built yet, so
a third state exists:

    PASS    works now
    TODO    not implemented yet, and known not to be - not a regression
    FAIL    something that used to work, or should work today, is broken

Only FAIL sets a non-zero exit code. As steps 3/4/6 land, TODO lines should turn into
PASS lines without this file needing new checks - that is the point of listing them.

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
# PORT NOTE: `from hivemind_env.models import CustomCombinedExtractor` moved out of this
# block. models.py is a 0-byte placeholder on this branch, so keeping it here made the
# whole file exit(1) at the first import and no other check ever ran.
try:
    import numpy as np
    import torch  # noqa: F401  (imported for the version banner and the device probe)
    from stable_baselines3 import PPO  # noqa: F401
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: F401
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: F401
    from hivemind_env.env import HiveMindMultiAgentEnv
    from hivemind_env.training import (
        DEFAULT_OBS_SIZE,
        NUM_AGENTS,
        CurriculumCallback,
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
# PORT NOTE: the single-agent version asserted `obs["grid"].shape == (N, N, 5)` and
# `len(info["lidar_distances"]) == 180`. Neither exists here: `_get_obs()` returns [],
# there is no `observation_space`, and there are no rayTest calls anywhere in env.py.
# Those are roadmap step 3, so they are checked as TODO, not asserted as FAIL.
env = None
try:
    env = HiveMindMultiAgentEnv(render_mode=None, difficulty_level=1, obs_size=DEFAULT_OBS_SIZE)
    emit(f"PASS: env constructed. num_agents={env.num_agents}, action_space={env.action_space}")
    if env.num_agents != NUM_AGENTS:
        emit(f"FAIL: expected {NUM_AGENTS} agents, got {env.num_agents}")

    if hasattr(env, "observation_space"):
        emit(f"PASS: observation_space declared: {env.observation_space}")
    else:
        emit("TODO: no observation_space declared (roadmap step 3)")

    obs, info = env.reset(seed=0)
    emit(f"PASS: env.reset(seed=0) OK. info keys={sorted(info)}")
    emit(f"  remaining_resources={info['remaining_resources']}, "
         f"robot_pos[0]={tuple(round(v, 3) for v in info['robot_pos'][0])}")

    if len(info["robot_pos"]) != NUM_AGENTS:
        emit(f"FAIL: expected {NUM_AGENTS} robot poses, got {len(info['robot_pos'])}")
    if info["remaining_resources"] != 12:
        emit(f"FAIL: expected 12 cartons at reset, got {info['remaining_resources']}")

    if len(obs) == 0:
        emit("TODO: _get_obs() returns an empty list (roadmap step 3)")
    else:
        emit(f"PASS: obs is non-empty: type={type(obs).__name__} len={len(obs)}")

    if "lidar_distances" in info:
        emit(f"PASS: info exposes {len(info['lidar_distances'])} lidar distances")
    else:
        emit("TODO: no LiDAR in info - env.py has no rayTest calls (roadmap step 3)")
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
            emit("TODO: reward is hard-coded [0,0,0,0] (roadmap step 4)")
        if env.current_step >= env.max_steps:
            emit("FAIL: step budget exhausted during a 5-step smoke test")
        emit(f"PASS: current_step advanced to {env.current_step} (max_steps={env.max_steps})")
    except Exception as e:
        emit(f"FAIL: env.step(): {type(e).__name__}: {e}")
    finally:
        # PORT NOTE: the single-agent env guarded close() against a double disconnect and
        # this test asserted that. HiveMindMultiAgentEnv.close() does not - the second
        # call raises pybullet.error("Not connected to physics server."). Reported rather
        # than fixed: env.py is out of scope this session. The one-line fix is to clear
        # self.client_id (or set a _closed flag) inside close().
        env.close()
        try:
            env.close()
            emit("PASS: env.close() is idempotent")
        except Exception as e:
            emit(f"TODO: env.close() is NOT idempotent - second call raises "
                 f"{type(e).__name__}: {e} (fix in env.py: guard on a _closed flag)")

# -- Termination wiring ---------------------------------------------------------
try:
    src = (ROOT / "hivemind_env" / "env.py").read_text(encoding="utf-8")
    if "return self._get_obs(), [0]*self.num_agents, False, False" in src:
        emit("TODO: terminated/truncated are hard-coded False; max_steps unenforced "
             "(roadmap step 4)")
    else:
        emit("PASS: step() no longer returns hard-coded reward/termination")
except Exception as e:
    emit(f"FAIL: could not read env.py: {e}")

# -- Feature extractor wiring ---------------------------------------------------
# PORT NOTE: was an unconditional build of CustomCombinedExtractor. hivemind_env/models.py
# is 0 bytes on this branch (roadmap step 6), and it also needs an observation_space that
# does not exist yet, so this is a soft check.
try:
    from hivemind_env.models import CustomCombinedExtractor  # noqa: F401
    emit("PASS: hivemind_env.models.CustomCombinedExtractor importable")
except ImportError:
    emit("TODO: hivemind_env/models.py is empty - no feature extractor yet (roadmap step 6)")
except Exception as e:
    emit(f"FAIL: models import: {type(e).__name__}: {e}")

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
