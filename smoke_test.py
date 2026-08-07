"""Quick smoke test - runs in-process, writes output to file for inspection."""
import sys
import os

sys.path.insert(0, r'd:\Dev Projects\Hivemind\Single-Agent-implementation')
os.chdir(r'd:\Dev Projects\Hivemind\Single-Agent-implementation')

results = []

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from hivemind_env.env import HiveMindSingleAgentEnv
    from hivemind_env.models import CustomCombinedExtractor
    results.append("PASS: All imports successful")
except Exception as e:
    results.append(f"FAIL: Import error: {e}")

# ── Device check ─────────────────────────────────────────────────────────────
try:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        device = "cuda" if cap[0] >= 7 else "cpu"
        results.append(f"PASS: GPU={name} sm_{cap[0]}{cap[1]} -> device={device}")
    else:
        device = "cpu"
        results.append("PASS: No GPU detected -> device=cpu")
except Exception as e:
    results.append(f"FAIL: Device check: {e}")

# ── Environment ───────────────────────────────────────────────────────────────
try:
    env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=1)
    obs, info = env.reset()
    grid_shape = obs["grid"].shape
    is_carrying = obs["is_carrying"]
    results.append(f"PASS: env.reset() OK. grid={grid_shape}, is_carrying={is_carrying}")
except Exception as e:
    results.append(f"FAIL: env.reset(): {e}")
    env = None

# ── Steps ─────────────────────────────────────────────────────────────────────
if env is not None:
    try:
        import numpy as np
        rewards = []
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(int(action))
            rewards.append(reward)
            results.append(f"  Step {i+1}: action={action} reward={reward:.4f} term={term} trunc={trunc}")
        env.close()
        results.append(f"PASS: 5 steps OK. Rewards: {[round(r,4) for r in rewards]}")
    except Exception as e:
        results.append(f"FAIL: env.step(): {e}")
        try:
            env.close()
        except:
            pass

# ── PBRS scale check - verify env.py change ──────────────────────────────────
try:
    with open(r"hivemind_env\env.py", "r") as f:
        src = f.read()
    if "* 5.0" in src and "* 1.0" not in src.split("PBRS")[1].split("terminated")[0]:
        results.append("PASS: PBRS scale = 5.0 confirmed in env.py")
    else:
        results.append("WARN: Check PBRS scale in env.py manually")
    if "APF Repulsive Force removed" in src:
        results.append("PASS: APF removal comment confirmed in env.py")
    else:
        results.append("WARN: APF may not be removed - check env.py")
except Exception as e:
    results.append(f"WARN: Could not verify env.py: {e}")

# ── train.py checks ───────────────────────────────────────────────────────────
try:
    with open("train.py", "r") as f:
        train_src = f.read()
    checks = {
        "import torch": "import torch",
        "VecMonitor": "VecMonitor(env)",
        "linear_schedule": "def linear_schedule",
        "LR reset in callback": "self.model.lr_schedule = linear_schedule",
        "70% threshold": "CURRICULUM_THRESH = 0.70",
        "10M steps": "10_000_000",
        "Isolated log dir": "tensorboard_logs_ppo_v1",
        "Isolated checkpoint dir": "checkpoints_ppo_v1",
        "v1 final model": "ppo_hivemind_v1_final",
        "cudnn.benchmark": "cudnn.benchmark = True",
        "n_steps=2048": "N_STEPS           = 2048",
        "batch_size=512": "BATCH_SIZE        = 512",
    }
    for name, token in checks.items():
        if token in train_src:
            results.append(f"PASS: train.py has '{name}'")
        else:
            results.append(f"FAIL: train.py MISSING '{name}' (token: {token})")
except Exception as e:
    results.append(f"FAIL: Could not read train.py: {e}")

# ── Write results ─────────────────────────────────────────────────────────────
out_path = r"d:\Dev Projects\Hivemind\Single-Agent-implementation\smoke_test_results.txt"
with open(out_path, "w") as f:
    f.write("\n".join(results))

print("\n".join(results))
print(f"\nResults written to: {out_path}")
