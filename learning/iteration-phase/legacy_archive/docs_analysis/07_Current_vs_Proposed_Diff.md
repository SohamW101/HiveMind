# Current vs Proposed: Complete Diff Analysis

> This document is the authoritative reference for every change planned for the `multi-agent-rl` branch. Each section shows the **current code**, the **exact proposed change**, the **design decision** behind it, the **tradeoffs**, and the **alternatives that were rejected**. No change is made without being documented here first.

---

## Table of Contents

1. [Reward Engineering](#1-reward-engineering)
   - 1.1 Idle Turning Penalty Default
2. [Emergent Communication](#2-emergent-communication)
   - 2.1 Action Space Encoding (Two Approaches to Test)
   - 2.2 Message Slot Population
   - 2.3 Message Delivery Timing
   - 2.4 Greedy Controller Compatibility
3. [VecEnv Wrappers](#3-vecenv-wrappers)
   - 3.1 `vec_env.py` Action Space
   - 3.2 `subproc_vec_env.py` Action Space
4. [Training Script](#4-training-script)
   - 4.1 Batch Size Auto-Scaling
   - 4.2 Communication CLI Flag
5. [Post-Training Test Suite](#5-post-training-test-suite)
   - 5.1 `tests/test_post_training.py`
   - 5.2 `tests/test_environment.py`
6. [File-by-File Summary](#6-file-by-file-summary)

---

## 1. Reward Engineering

### 1.1 Idle Turning Penalty Default

**File**: `hivemind_env/env.py`, line ~434

#### Current Code
```python
def __init__(self, render_mode=None, difficulty_level=1, obs_dim=DEFAULT_OBS_DIM,
             show_lidar=None, obs_size=None, idle_penalises_turning=True,
             lidar_noise=True, substeps=None, max_steps=None,
             num_cartons=None, shaping=True,
             shaping_scale=SHAPING_SCALE_DEFAULT, gamma=0.99):
```

#### Proposed Change
```diff
 def __init__(self, render_mode=None, difficulty_level=1, obs_dim=DEFAULT_OBS_DIM,
-             show_lidar=None, obs_size=None, idle_penalises_turning=True,
+             show_lidar=None, obs_size=None, idle_penalises_turning=False,
              lidar_noise=True, substeps=None, max_steps=None,
              num_cartons=None, shaping=True,
              shaping_scale=SHAPING_SCALE_DEFAULT, gamma=0.99):
```

#### Design Decision
The spec says "v < 0.1 m/s → idle penalty". In this grid environment, `TURN_LEFT` and `TURN_RIGHT` have linear velocity = 0. So every turn triggers the -0.02 idle penalty (weighted to -0.002). Over a 400-step episode with ~100 turns, that's -0.2 cumulative — small individually, but in PPO's gradient landscape it teaches the policy that turning is costly.

#### Why Selected
- The **greedy baseline** turns freely and is the benchmark. Penalizing the learned policy for something the benchmark does for free is an unfair comparison.
- The flag `idle_penalises_turning` already exists in the codebase (line 1317), so this change is a **one-character default flip**, not new logic.
- The idle penalty still applies to `STAY` and `PICKUP`/`DROP` at empty locations, which are genuinely wasteful actions.

#### Tradeoff
| Factor | `True` (current) | `False` (proposed) |
|--------|-------------------|---------------------|
| Spec compliance | Literal interpretation | Loosened: turns exempt |
| Exploration | Slightly suppressed | Unpenalized |
| Impact per step | -0.002 per turn | 0.0 |
| Cumulative (400 steps) | -0.2 to -0.4 | 0.0 |
| Greedy baseline fairness | Tilted against RL | Level playing field |

#### Alternatives Rejected
- **Reduce penalty magnitude**: Setting `R_IDLE_PENALTY = -0.005` instead. Rejected because the penalty is already tiny; the issue is that it exists at all for a mandatory navigation action.
- **Positive turn reward**: Adding `+0.001` per turn. Rejected because it introduces a farming opportunity (spin forever for free reward) that the shaping cannot cancel.

---

## 2. Emergent Communication

### 2.1 Action Space Encoding (Two Approaches to Test)

**File**: `hivemind_env/env.py`, line ~523

#### Current Code
```python
# Actions: 0: Forward, 1: Backward, 2: Turn Left, 3: Turn Right,
#          4: Pick Up, 5: Drop Off, 6: Stay
self.action_space = spaces.MultiDiscrete([7] * self.num_agents)
```

#### Approach A: `MultiDiscrete([7, 16] * 4)`
```python
if self.communication:
    # Each robot outputs [movement_action, message_token]
    self.action_space = spaces.MultiDiscrete([7, MSG_TOKENS] * self.num_agents)
else:
    self.action_space = spaces.MultiDiscrete([7] * self.num_agents)
```

**How SB3 sees it**: The policy outputs an 8-element vector `[move0, msg0, move1, msg1, move2, msg2, move3, msg3]`. SB3 handles `MultiDiscrete` natively — each element gets its own categorical distribution head, trained independently.

**Pros**:
- Clean separation of movement and communication in the action vector.
- SB3's PPO natively supports `MultiDiscrete` — no custom policy needed.
- Each head has its own entropy, so `ent_coef` applies to both, encouraging message exploration.

**Cons**:
- Doubles the action dimensionality from 4 to 8 elements.
- The message head trains on every step even when the movement head is the bottleneck.

#### Approach B: `MultiDiscrete([112] * 4)`
```python
if self.communication:
    # Each robot outputs a single action encoding both movement and message:
    #   action = movement * MSG_TOKENS + message_token
    self.action_space = spaces.MultiDiscrete(
        [7 * MSG_TOKENS] * self.num_agents  # [112, 112, 112, 112]
    )
else:
    self.action_space = spaces.MultiDiscrete([7] * self.num_agents)
```

**How SB3 sees it**: The policy outputs a 4-element vector where each element is a single integer in [0, 111]. The movement is `action // 16` and the message is `action % 16`.

**Pros**:
- Keeps the action vector the same length (4 elements).
- Jointly samples movement and message, capturing correlations (e.g., "when I move forward, I should say token 5").

**Cons**:
- 112 categories is a large discrete space — takes longer to explore.
- The entropy of movement and message are entangled, making `ent_coef` tuning harder.
- Decoding requires `divmod` in `step()`.

#### Plan
Both approaches will be implemented behind a flag. The training script will accept `--comm-encoding multi` (Approach A) or `--comm-encoding merged` (Approach B). Smoke tests will verify both. The one that converges faster on the 4-carton curriculum wins.

### 2.2 Message Slot Population

**File**: `hivemind_env/env.py`, in `step()` method and `_get_obs()`

#### Current Code (messages are always zero)
```python
# In reset():
self.messages = np.zeros((self.num_agents, MSG_TOKENS), dtype=np.float32)

# In _get_obs():
row[OBS_SLICES["messages"]] = np.concatenate([self.messages[j] for j in others])
```

#### Proposed Change
```python
# In step(), BEFORE computing observations (same-step delivery):
if self.communication:
    for i in range(self.num_agents):
        if self.comm_encoding == "multi":
            msg_token = int(actions[i * 2 + 1])
        else:  # merged
            msg_token = int(actions[i]) % MSG_TOKENS
        self.messages[i] = np.zeros(MSG_TOKENS, dtype=np.float32)
        self.messages[i][msg_token] = 1.0

# _get_obs() is UNCHANGED — it already reads self.messages and writes them
# into the observation. The wiring is already complete.
```

#### Design Decision: Same-Step Delivery (from Previous Step's Action)

Messages are written into `self.messages` at step $t$ and observed at step $t+1$. This happens automatically because:
1. `step()` updates `self.messages` from the current actions.
2. `_get_obs()` reads `self.messages` to build the observation for the NEXT call.
3. The observation returned by `step()` at time $t$ uses the messages set by actions at time $t$, which were decided based on observations from time $t-1$.

So from each robot's perspective: "I see what others *said* based on their actions this step, alongside the physical state that resulted from those actions." This is effectively a 0-step communication latency on physical state but represents intentions decided simultaneously.

#### Tradeoff: One-Hot vs Single-Float Encoding

| Factor | One-Hot (16 floats) | Single Float (`token/15`) |
|--------|---------------------|---------------------------|
| Message slots used | 16 per robot (fits exactly) | 1 per robot (wastes 15 slots) |
| Network learning | Categorical — each token is equidistant | Ordinal — token 7 is "close" to 8 |
| Compositionality | High — tokens are independent symbols | Low — imposes false ordering |
| Observation width | No change (48 slots already reserved) | Would require layout change |

**Selected**: One-hot. The 48 reserved slots were designed for exactly this — `3 robots × 16 tokens = 48`. It fits perfectly and treats tokens as the categorical symbols they are.

### 2.3 Message Delivery Timing Diagram

```
Step t:
  1. Robot receives obs_t (contains messages from step t-1 actions)
  2. Policy outputs [movement_t, message_t] 
  3. env.step() processes movements → physics settles
  4. env.step() writes message_t into self.messages  ← HERE
  5. env.step() calls _get_obs() which reads self.messages
  6. Returns obs_{t+1} (contains messages from step t actions)

Robot A at step t+1 sees what Robot B *said* at step t.
This is a 1-logical-step delay: B decided what to say based on obs_t,
and A sees it in obs_{t+1}.
```

### 2.4 Greedy Controller Compatibility

**File**: `hivemind_env/greedy.py`

#### Current Code (returns list of ints)
```python
def act(self, env=None):
    ...
    actions = [STAY] * self.n
    ...
    return actions  # e.g., [0, 6, 4, 2]
```

#### Proposed Change
```python
def act(self, env=None):
    ...
    actions = [STAY] * self.n
    ...
    if getattr(self.env, "communication", False):
        # Greedy doesn't communicate — always sends token 0
        if self.env.comm_encoding == "multi":
            joint = []
            for a in actions:
                joint.extend([a, 0])
            return joint  # e.g., [0, 0, 6, 0, 4, 0, 2, 0]
        else:  # merged
            return [a * MSG_TOKENS + 0 for a in actions]  # e.g., [0, 96, 64, 32]
    return actions
```

**Design Decision**: Greedy always sends token 0. This is intentional: the greedy baseline is a *no-communication* reference. If the learned policy beats greedy AND uses non-trivial messages, we can measure whether communication contributed.

---

## 3. VecEnv Wrappers

### 3.1 `vec_env.py` Action Space

**File**: `hivemind_env/vec_env.py`, line ~86

#### Current Code
```python
single_act = spaces.Discrete(int(self.envs[0].action_space.nvec[0]))
```

#### Proposed Change
```python
if getattr(self.envs[0], "communication", False):
    # MultiDiscrete([7, 16]) per slot — movement + message
    if self.envs[0].comm_encoding == "multi":
        single_act = spaces.MultiDiscrete([7, MSG_TOKENS])
    else:
        single_act = spaces.Discrete(7 * MSG_TOKENS)
else:
    single_act = spaces.Discrete(7)
```

The `step_async` / `step_wait` methods must also be updated to reshape the action array correctly for the underlying multi-agent env.

### 3.2 `subproc_vec_env.py` Action Space

Same pattern as `vec_env.py`. The worker function `_worker` receives actions from the pipe and passes them to `env.step()` — the action format is transparent as long as the pipe carries the right shape.

---

## 4. Training Script

### 4.1 Batch Size Auto-Scaling

**File**: `train.py`, after line ~175

#### Current Code
```python
worlds = args.worlds or max(2, min(8, num_parallel_envs(cap=8)))
slots = worlds * NUM_AGENTS
```
*(No validation of batch_size vs buffer size)*

#### Proposed Change
```python
worlds = args.worlds or max(2, min(8, num_parallel_envs(cap=8)))
slots = worlds * NUM_AGENTS

# Auto-scale batch_size to prevent micro-batches or overflow
buffer = args.n_steps * slots
if args.batch_size > buffer:
    old = args.batch_size
    args.batch_size = buffer
    print(f"  [auto] batch_size {old} > buffer {buffer}, clamped to {buffer}")
elif buffer // args.batch_size < 4:
    old = args.batch_size
    args.batch_size = max(64, buffer // 8)
    print(
        f"  [auto] batch_size {old} gives only {buffer // old} mini-batches, "
        f"adjusted to {args.batch_size} ({buffer // args.batch_size} mini-batches)"
    )
```

#### Why This Matters

| Scenario | worlds | slots | buffer (n_steps=512) | batch_size | mini-batches |
|----------|--------|-------|----------------------|------------|--------------|
| Normal   | 8      | 32    | 16,384               | 1024       | 16 ✓         |
| Debug    | 1      | 4     | 2,048                | 1024       | 2 ✗ (too few)|
| Smoke    | 2      | 8     | 1,024                | 1024       | 1 ✗ (no shuffle)|

PPO needs ≥4 mini-batches per epoch for gradient variance reduction. The auto-scale catches the debug case.

### 4.2 Communication CLI Flag

**File**: `train.py`

#### Proposed Addition
```python
p.add_argument(
    "--communication",
    action="store_true",
    help="Enable emergent communication (roadmap step 7). Each robot "
    "selects a 16-token message alongside its movement action. "
    "The no-communication run is the baseline this must beat.",
)
p.add_argument(
    "--comm-encoding",
    choices=["multi", "merged"],
    default="multi",
    help="How the communication token is encoded in the action space. "
    "'multi': MultiDiscrete([7, 16]) per slot. "
    "'merged': Discrete(112) per slot.",
)
```

These flags are passed through to `build_env()` → `HiveMindSubprocVecEnv` → `HiveMindMultiAgentEnv`.

---

## 5. Post-Training Test Suite

### 5.1 `tests/test_post_training.py`

**File**: [NEW] `tests/test_post_training.py`

A standalone script (not pytest — runs from CLI with a model path) that loads a trained checkpoint and evaluates it against concrete, measurable criteria.

#### Test Matrix

| # | Test | Threshold | Source | PASS condition |
|---|------|-----------|--------|----------------|
| 1 | Completion rate | ≥50% | Greedy = 100% | `completed / episodes >= 0.50` |
| 2 | Mean makespan | < 120 steps | Greedy = 97.6 | `mean(steps_when_complete) < 120` |
| 3 | Action diversity | No action > 60% | Greedy: fwd=42%, turn=28% | `max(action_counts) / total < 0.60` |
| 4 | Pickup rate | ≥ 1.0 / episode | Greedy = 12.0 | `total_pickups / episodes >= 1.0` |
| 5 | Collision rate | < 10.0 / episode | Greedy = 6.7 | `total_collisions / episodes < 10.0` |
| 6 | Det vs Stoch gap | Both complete ≥1 | — | `det_completions >= 1 AND stoch_completions >= 1` |
| 7 | Comm entropy | > 1.0 bit | — | `H(token_distribution) > 1.0` (comm only) |

#### Why These Thresholds
- **Completion ≥50%**: A random policy completes 0% at 12 cartons. 50% demonstrates real learning. Greedy hits 100%, so 50% is a minimum bar, not a goal.
- **Makespan < 120**: Greedy averages 97.6. A policy that completes in 120 has learned routing but not coordination. Below 97.6 means it's beating greedy.
- **Action diversity**: The failure mode documented in `test_run.py` (deterministic argmax collapsing to one action) would show as >80% on a single action. 60% catches it early.
- **Comm entropy > 1.0 bit**: With 16 tokens, max entropy is 4.0 bits. If all messages are token 0, entropy is 0.0. Above 1.0 means the policy is using at least 3-4 tokens meaningfully.

### 5.2 `tests/test_environment.py`

**File**: [NEW] `tests/test_environment.py`

Pytest-compatible unit tests for the environment mechanics. These run without a trained model.

#### Test List

```python
class TestObservation:
    def test_shape_and_bounds()       # obs.shape == (4, 177), all in [-1, 1]
    def test_lidar_range()            # lidar values in [0, 1]
    def test_message_slots_zero()     # without communication, messages are all 0
    def test_message_slots_nonzero()  # with communication, messages reflect sent tokens

class TestReward:
    def test_shaping_telescopes()     # sum(F) over episode = scale * (Phi_end - Phi_start)
    def test_pickup_reward_positive() # pickup step has net positive reward
    def test_delivery_reward_positive() # delivery step has net positive reward
    def test_collision_penalty()      # collision gives negative shared reward
    def test_idle_penalty_exempt_turn() # with idle_penalises_turning=False, turn is not idle

class TestMechanics:
    def test_close_idempotent()       # env.close(); env.close() — no error
    def test_curriculum_cartons()     # setting num_cartons=4 yields only 4 active
    def test_blocked_cells()          # shelf cells are blocked, gap cells are open
    def test_pickup_in_range()        # PICKUP succeeds within 1.5 cells
    def test_pickup_out_of_range()    # PICKUP fails beyond 1.5 cells → invalid action

class TestCommunication:
    def test_action_space_multi()     # comm_encoding="multi" → MultiDiscrete([7,16]*4)
    def test_action_space_merged()    # comm_encoding="merged" → MultiDiscrete([112]*4)
    def test_message_propagation()    # token sent at step t appears in obs at step t+1
    def test_self_message_excluded()  # robot does not see its own message
    def test_greedy_compat()          # greedy controller works with communication enabled
```

---

## 6. File-by-File Summary

| File | Change Type | What Changes |
|------|-------------|--------------|
| `hivemind_env/env.py` | MODIFY | `idle_penalises_turning` default → `False`; add `communication` and `comm_encoding` params to `__init__`; update `action_space`; populate `self.messages` in `step()` |
| `hivemind_env/vec_env.py` | MODIFY | Update `single_act` space for communication; reshape actions in `step_async` |
| `hivemind_env/subproc_vec_env.py` | MODIFY | Same action space update as `vec_env.py` |
| `hivemind_env/models.py` | NO CHANGE | Architecture is already communication-ready |
| `hivemind_env/training.py` | NO CHANGE | Curriculum and scaffolding unchanged |
| `hivemind_env/greedy.py` | MODIFY | Output `[movement, 0]` pairs when communication is enabled |
| `train.py` | MODIFY | Add `--communication`, `--comm-encoding` flags; batch size auto-scaling |
| `test_run.py` | MODIFY | Handle new action format when loading communication-trained checkpoints |
| `smoke_test.py` | MODIFY | Add communication-mode smoke test section |
| `tests/test_post_training.py` | NEW | Post-training evaluation with 7 concrete tests |
| `tests/test_environment.py` | NEW | Unit tests for environment mechanics and communication wiring |

---

## Design Principles Applied

1. **No observation width change**: The 48 message slots were reserved precisely for this. Communication fills them without bumping OBS_DIM.
2. **Flag-gated**: Every change is behind `--communication` / `comm_encoding`. The no-communication baseline remains the default and is untouched.
3. **Architecture-stable**: `models.py` is unchanged. The message MLP branch already processes `[129:177]`. It was learning on zeros; now it learns on real signals.
4. **Greedy-compatible**: The greedy controller adapts to the new action format automatically. Its benchmark numbers remain valid.
5. **Checkpoint-aware**: Communication-enabled and no-communication checkpoints are incompatible (different action heads). The `--communication` flag makes this explicit rather than silent.
