"""
Quick smoke test - runs in-process, writes output to file for inspection.

Checks that the package imports, a device is selectable, and the env can reset and step.
It used to hardcode an absolute path from another machine ("d:\\Dev Projects\\...") and
os.chdir into it, so it raised FileNotFoundError on line 6 before doing anything. It also
grepped train.py for literal strings, which tested the text of a file rather than any
behaviour and reported a FAIL for a stale token that had been correctly removed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "smoke_test_results.txt"

results = []

# -- Imports ------------------------------------------------------------------
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from hivemind_env.env import OBS_SIZE_V1, HiveMindSingleAgentEnv
    from hivemind_env.models import CustomCombinedExtractor
    from hivemind_env.training import CurriculumCallback, get_device, linear_schedule
    results.append("PASS: All imports successful")
except Exception as e:
    results.append(f"FAIL: Import error: {e}")
    OUT_PATH.write_text("\n".join(results), encoding="utf-8")
    print("\n".join(results))
    sys.exit(1)

# -- Device check --------------------------------------------------------------
try:
    device = get_device()
    results.append(f"PASS: device={device}")
except Exception as e:
    results.append(f"FAIL: Device check: {e}")

# -- Environment ---------------------------------------------------------------
env = None
try:
    env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=1, obs_size=OBS_SIZE_V1)
    obs, info = env.reset(seed=0)
    grid_shape = obs["grid"].shape
    results.append(f"PASS: env.reset() OK. grid={grid_shape}, is_carrying={obs['is_carrying']}")
    if grid_shape != (OBS_SIZE_V1, OBS_SIZE_V1, 5):
        results.append(f"FAIL: expected grid ({OBS_SIZE_V1}, {OBS_SIZE_V1}, 5), got {grid_shape}")
    if len(info["lidar_distances"]) != env.lidar_num_rays:
        results.append(f"FAIL: expected {env.lidar_num_rays} lidar rays, "
                       f"got {len(info['lidar_distances'])}")
    else:
        results.append(f"PASS: info exposes {env.lidar_num_rays} lidar distances")
except Exception as e:
    results.append(f"FAIL: env.reset(): {e}")

# -- Steps ---------------------------------------------------------------------
if env is not None:
    try:
        rewards = []
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(int(action))
            rewards.append(reward)
            results.append(
                f"  Step {i+1}: requested={info['requested_action']} "
                f"executed={info['executed_action']} reward={reward:.4f} "
                f"term={term} trunc={trunc}"
            )
            if term or trunc:
                obs, info = env.reset()
        results.append(f"PASS: 5 steps OK. Rewards: {[round(r, 4) for r in rewards]}")
    except Exception as e:
        results.append(f"FAIL: env.step(): {e}")
    finally:
        env.close()
        env.close()  # idempotent - close() is guarded against double disconnect
        results.append("PASS: env.close() is idempotent")

# -- Feature extractor wiring ---------------------------------------------------
try:
    probe = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=1, obs_size=OBS_SIZE_V1)
    extractor = CustomCombinedExtractor(probe.observation_space, features_dim=256)
    results.append(f"PASS: extractor built for {OBS_SIZE_V1}x{OBS_SIZE_V1} "
                   f"(linear in_features={extractor.linear[0].in_features})")
    probe.close()
except Exception as e:
    results.append(f"FAIL: extractor build: {e}")

# -- Write results --------------------------------------------------------------
OUT_PATH.write_text("\n".join(results), encoding="utf-8")
print("\n".join(results))
print(f"\nResults written to: {OUT_PATH}")

sys.exit(1 if any(r.startswith("FAIL") for r in results) else 0)
