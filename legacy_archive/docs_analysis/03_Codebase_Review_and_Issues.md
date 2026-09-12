# Codebase Review and Issues

A review of the `multi-agent-rl` branch codebase reveals a highly structured and thoroughly documented architecture. However, several critical flaws, legacy artifacts, and potentially dangerous edge-cases exist.

## 1. PyBullet Connection Leaks (`env.close()`)

**Issue**: `HiveMindMultiAgentEnv.close()` is not idempotent. Calling it twice raises `pybullet.error("Not connected to physics server.")`. 
**Impact**: When training completes or interrupts, `SubprocVecEnv` attempts to tear down all parallel environments. If a failure triggers a cascading close, the entire teardown halts with a disconnect error, masking the original stack trace.
**Resolution**: Wrap the PyBullet disconnect call with a boolean flag.

```python
# In env.py -> HiveMindMultiAgentEnv
def close(self):
    if hasattr(self, "_closed") and self._closed:
        return
    try:
        pb.disconnect(physicsClientId=self.client_id)
    except Exception:
        pass
    self._closed = True
```

## 2. Evaluation Script Harness Overrides (`run_evaluation.py`)

**Issue**: The warning banners and assertions in `run_evaluation.py` and `smoke_test.py` contain stale language regarding roadmap steps.
- The script checks `len(obs) == 0` to throw an error: `WARNING: _get_obs() returns an empty list`. However, since observation V3 was merged, `_get_obs()` natively returns a `(4, 177)` array. Thus, `obs_missing` evaluates to False, bypassing the warning safely. Still, maintaining stale warning text causes developer confusion.
**Impact**: False assumptions for anyone onboarding into the project.

## 3. Disconnected / Legacy Reward Logic

**Issue**: `R_REPLAN_PENALTY = -0.1`
- The spec defines a penalty when $A^*$ triggers a replan. However, this environment enforces a discrete grid-teleport motion model; there is no $A^*$ running under the hood. The penalty is correctly not applied, but keeping the constant in the active reward block without clear warnings could mislead future researchers into thinking replanning is penalized.

## 4. `IDLE_SPEED_THRESHOLD` and Turning Penalties

**Issue**: The specification dictates an idle penalty (`-0.02`) if $v < 0.1$ m/s. Because the environment enforces a discrete grid, a turn-in-place action has $v=0$. 
**Impact**: A robot executing `TURN LEFT` is penalized exactly the same as a robot sitting completely idle. While currently controlled via the `idle_penalises_turning` flag, this fundamentally changes the mathematical specification of the reward function. Turning is a required action to solve the maze; taxing it inherently slows down exploration and promotes straight-line collisions.

## 5. Potential Batch Size Mismatches in PPO

**Issue**: `train.py` calculates rollouts based on `slots`. `slots = worlds * NUM_AGENTS`.
- `args.n_steps` is the rollout length *per slot*. The final buffer size sent to PPO is `args.n_steps * slots`. If `n_steps=512` and `worlds=8`, the buffer is `16384`. 
- `args.batch_size` defaults to 1024. `16384 / 1024 = 16` batches per epoch. 
**Impact**: If a user runs `train.py` with `worlds=1` (e.g., debugging) -> `slots=4`. Buffer is `512 * 4 = 2048`. `batch_size=1024`. This results in only 2 batches per epoch, severely hindering PPO's gradient variance stability.
**Resolution**: Automatically adjust `batch_size` relative to `worlds` inside `train.py`.
