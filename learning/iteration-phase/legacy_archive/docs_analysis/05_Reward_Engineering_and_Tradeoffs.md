# Reward Engineering and Tradeoffs

Reward engineering is arguably the most complex component of the HiveMind multi-agent environment. The baseline mathematical specification is sparse, meaning agents only receive rewards when events happen (deliveries, pickups, collisions). 

## 1. The Core Failure of Sparse S3.3 Rewards

The S3.3 formula defined in the original specs heavily penalizes collisions (`-5.0` per event shared) and time (`-0.05` per step shared). 
Under random actions, collisions are frequent. As documented in `env.py`, a baseline run converged to a mean reward of `-103` and essentially learned to freeze. **Standing perfectly still costs `-0.045` per step, which is mathematically optimal compared to moving and risking a `-5.0` collision penalty before finding a `+1.0` pickup.**

## 2. Potential-Based Reward Shaping

To fix this, the implementation relies on Potential-Based Reward Shaping (Ng, Harada & Russell). 

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

Where $\Phi$ is the potential of a state. 

### What is currently implemented:
The system uses $\Phi = -(\text{work remaining})$, effectively rewarding the robot for getting closer to the objective without allowing it to "farm" points by moving back and forth (since $F(s, s')$ telescopes perfectly).

### Tradeoff Analysis
- **Scale Factor**: The scale is set to `30.0`. This ensures that the Expected Value (EV) of a `MOVE` action becomes strictly positive ($+0.221$ gain per cell) even when factoring in the high probability of random-walk collisions. 
- **The Risk**: If the scale is too high, the magnitude of the shaping reward heavily overrides the actual task rewards. The policy might learn to rapidly grab cartons but ignore the `+100` completion bonus entirely because the gradients are dominated by the shaping scale.

## 3. Necessary Code Changes to the Reward System

If we want to enforce the best possible emergent behavior, we need to balance individual exploration with global coordination.

### Modification 1: Fix the Idle Turning Penalty
Currently, turning in place penalizes the robot because $v < 0.1$. 

**Code Change in `env.py` (`step` function processing)**:
```python
# Old:
# if current_speed < IDLE_SPEED_THRESHOLD and not at_depot:
#     R_individual += R_IDLE_PENALTY

# New: (Do not penalize if action was turn left/right)
if current_speed < IDLE_SPEED_THRESHOLD and not at_depot:
    # Action 2 is Turn Left, Action 3 is Turn Right
    if action[agent_idx] not in [2, 3] or self.idle_penalises_turning:
        R_individual += R_IDLE_PENALTY
```
*Why?* Navigation inherently requires turning. Taxing a turn suppresses exploration.

### Modification 2: Cooperative Collision Shielding (New Concept)
A flat `-5.0` penalty for a collision creates a Nash Equilibrium where both robots are terrified to enter an aisle. We should apply the penalty only to the robot moving, or implement a "Yield" reward.

**Code Change in `env.py` (`_calculate_collisions`)**:
```python
# Old:
# Both robots get charged -5.0 shared

# New Proposed Tradeoff:
# If Robot A was moving and hit stationary Robot B:
# Robot A (Instigator) gets -5.0 individual
# Shared pool gets -1.0
```
*Why?* This allows the stationary robot (which might be correctly performing a pickup) to not be punished for the random walk of an untrained agent next to it, stabilizing the critic's value estimation.

### Modification 3: Removing Dead Code
Remove the `R_REPLAN_PENALTY` entirely. Leaving it in implies future functionality that conflicts with the fundamental discrete grid architecture of this environment.
