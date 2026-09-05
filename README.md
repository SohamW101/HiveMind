# HiveMind Warehouse

HiveMind is a four-robot multi-agent reinforcement-learning environment in PyBullet. The
robots collectively pick up twelve resources from a randomized warehouse and deliver all
of them to a depot. The research question is whether they learn useful communication from
discrete broadcast tokens, rather than receiving hand-written roles or routes.

The implementation-plan-compatible default is a 177-value observation per robot. An
explicit V4 mode provides the decentralized sensor-derived map observation. Simulator
ground truth is used only for physics, rewards, and diagnostics in that mode.

## Task definition

The world is a 13 x 13 grid with 1 m cells. Six horizontal shelf rows are generated at
rows 1, 3, 5, 7, 9, and 11. Each row contains three shelf segments with positive lengths
whose total is nine cells. The two one-cell gaps hold resources, giving twelve resources.
Shelf lengths and gap positions are randomized on every reset while preserving this
structure.

The depot is corner cell `(0, 0)`. Four robots spawn in the literal corner cells:

```text
agent 0: (0, 0)    agent 1: (0, 12)
agent 2: (12, 0)   agent 3: (12, 12)
```

The depot marker is visual-only, so agent 0 can safely start there. Robots, shelves, shelf
posts, cartons, and boundary walls have collision geometry. A robot may not enter a wall or
shelf cell. Cartons are pickup targets and are not permanent navigation obstacles.

An episode terminates when every active resource has been delivered and truncates at the
configured step limit. Curriculum episodes can activate fewer resources while preserving
the twelve-slot resource identity used by diagnostics.

The depot is an interaction target, not a place where a delivery robot needs to stand. A
robot must perform `drop` from exactly one cardinal grid cell away from `(0, 0)`, such as
`(0, 1)` or `(1, 0)`. The drop action never moves a robot into the depot cell.

## Decentralized state and maps

The environment separates simulator truth, agent belief, and policy input. The actor never
receives the global carton table, all robot poses, or the privileged static occupancy grid.

Each robot owns an independent six-layer 13 x 13 map:

1. unknown
2. free
3. static obstacle
4. resource
5. other robot
6. depot

The map is fixed in world orientation and centered on that robot's spawn cell. Its spawn
cell is the map origin; turning does not rotate the map. At reset, cells are unknown except
for the known depot and the robot origin. After each settled physics step, that robot's
LiDAR updates only its own map. Ray traversals mark free cells and the first hit marks a
shelf, wall, resource, or nearby robot. Dynamic robot evidence is range-limited and may
become stale; it is never copied between agents.

The LiDAR mount is fixed at chassis level in the robot URDF and never raises when carrying.
The arm yaw joint is attached to the fixed mast rod at an elevated Z, above the LiDAR link.
During pickup, the carton is moved onto the arm's forward centreline and lifted to a fixed
high Z position; its bottom clears the constant LiDAR scan plane. This keeps perception
geometrically consistent while preventing the carried carton from occluding the chassis-level
scan or hanging at the side of the robot.

## Pickup arm sequence

The arm is attached to the mast through two joints:

```text
lidar_post_link -> arm_lift_joint -> arm_lift_link -> arm_yaw_joint -> arm_base_link
```

`arm_lift_joint` is prismatic, moves only on the Z axis, and has a lower position of `0.0`
and an upper position of `0.38 m`. The fixed mast rod is `0.48 m` long and terminates at
that upper slide position; no mast extends above the arm mechanism. The LiDAR remains a
separate fixed joint at chassis level.

One `pickup` action executes this sequence across the physics substeps:

1. Rotate the arm yaw toward the resource.
2. Lower the arm slide to the resource center height.
3. Lift the arm slide to its carried height, above the LiDAR plane.
4. Return arm yaw to `0`, the robot-forward position.

The resource follows the gripper during the sequence. After pickup, its XY position is
always directly in front of the robot and its center is raised to `z = 0.7 m`. Drop lowers
the resource to depot height and releases it; it must be executed from exactly one cardinal
cell away from the depot and never moves the robot into the depot.

The implementation-plan default V3 observation is a flat vector of 177 floats: 105 state
values and message values, with the 72-ray LiDAR at indices `[57:129]`. The custom CNN
extractor in `hivemind_env/models.py` processes those LiDAR values separately. Explicit V4
mode (`obs_dim=1143, decentralized=True`) provides the six-layer local-map observation for
the decentralized map experiment.

## Actions and collision safety

Each robot chooses its movement action simultaneously with one communication token:

| Action | Meaning |
| --- | --- |
| `0` | move forward one cell |
| `1` | move backward one cell |
| `2` | turn left |
| `3` | turn right |
| `4` | pick up the nearest resource in range |
| `5` | drop the carried resource at the depot |
| `6` | stay |

Pickup and drop use the same strict interaction configuration:

```text
INTERACTION_DISTANCE_CELLS = 1
```

The robot must occupy a cardinal neighbor of the target cell. Diagonal distance, zero
distance, and two-or-more-cell distance are invalid. Pickup is therefore performed from a
cell directly adjacent to the resource, and drop is performed from a cell directly
adjacent to the depot. The action masks use the same rule as `step()`, so masked pickup
and drop actions cannot select a position that the environment would reject.

Movement proposals are resolved before physics. A move is rejected when it would enter an
occupied cell, collide with another proposal, swap positions with another robot, leave the
grid, or enter a shelf cell. Conflicting moves are rejected together so agent ordering
cannot permanently privilege one robot. Rejected moves are reported as blocked or invalid;
valid evaluation episodes must have zero physical robot-robot contacts.

## Communication

Communication is global, discrete, and delayed by one environment step. A robot hears the
other three robots in fixed speaker order and never hears itself. Every token is a learned
symbol; the environment assigns no meanings such as “claim resource 3” or “go left”. The
first observation is silent because no token has yet been emitted.

Training can apply independent dropout to every listener-speaker link. Evaluation supports
`learned`, `silent`, `shuffled`, and `random` channel modes. These interventions test
whether tokens carry coordination information rather than adding unused entropy. The
environment records emitted tokens, received tokens, dropped links, and message diagnostics
in `info`.

## Reward and metrics

The reward combines shared and individual terms:

- shared completion: `+100` once
- shared delivery: `+10` per delivered resource
- shared makespan bonus: `+50 * (max_steps - finish_step) / max_steps`
- shared time cost: `-0.05` per step
- individual pickup: `+1`
- individual delivery: `+2`
- invalid action: `-0.5`
- optional idle cost: `-0.02`
- physical collision event: `-5`

Shared reward has weight `0.90` and individual reward has weight `0.10`. Optional
potential-based shaping is added separately and reported separately. It uses reachable grid
distance rather than straight-line distance, so shelves do not create false local minima.

The important evaluation metrics are completion rate, makespan, delivered fraction,
physical collisions, blocked-by-agent moves, invalid actions, distance travelled, and
message dependence. Reward alone is not a success criterion.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Python 3.10 or newer is supported. The runtime uses Gymnasium, NumPy, and PyBullet.
Stable-Baselines3, sb3-contrib, and PyTorch are included for training integrations.

## Minimal environment check

```python
from hivemind_env.env import HiveMindMultiAgentEnv

env = HiveMindMultiAgentEnv(render_mode=None, comms=True, lidar_noise=False)
try:
    observation, info = env.reset(seed=0)
    print(observation.shape)  # (4, 177)
    observation, reward, terminated, truncated, info = env.step(
        [[6, 0], [6, 0], [6, 0], [6, 0]]
    )
finally:
    env.close()
```

Use `[6, 6, 6, 6]` with `comms=False`. Use a `(4, 2)` array with `comms=True`; column
zero is movement and column one is the token. Flattened communication actions are rejected
to prevent silent wiring bugs.

## Environment verifier

Run the deterministic integration test from the repository root:

```bash
.venv/bin/python verify_environment.py
```

It checks the warehouse structure, all four corner spawns, LiDAR range and obstacle
evidence, private map updates and anchoring, observation bounds, obstacle rejection,
robot movement arbitration, pickup, depot delivery, delayed broadcast tokens, dropout
configuration, exact one-cell pickup/drop distance, and clean environment shutdown. It uses `seed=123`/`seed=9` and disables
LiDAR noise so failures are reproducible. It validates environment mechanics; it does not
claim that a learned policy has solved the task.

For a visual hard-coded warehouse walkthrough, run:

```bash
.venv/bin/python verify_environment.py --gui
```

The walkthrough resets with seed `2026`, verifies the fixed twelve-resource layout, then
moves agent 0 through every resource in order, executing pickup, return-to-depot, and drop
actions. Agents 1-3 remain visible and stationary so the full four-robot scene can be
inspected. The normal environment remains randomized; this fixed seed belongs only to the
demonstration script. On machines without a display, run the identical route with:

```bash
.venv/bin/python verify_environment.py --headless-demo
```

## Experimental protocol

The required comparison is:

```text
local maps + no communication
versus
local maps + learned communication
```

Both arms must use identical layouts, seeds, episode budgets, curriculum, network capacity,
and optimizer settings. Also evaluate the communication policy with messages silenced,
randomized, and speaker-shuffled. A communication claim is supported only when channel
corruption harms coordination while sensing and action conditions remain unchanged.

Recommended progression:

1. Verify one-resource pickup and delivery.
2. Train with one resource.
3. Promote through two, four, eight, and twelve resources.
4. Establish the local-map no-communication baseline.
5. Train the local-map communication policy.
6. Run channel ablations and fixed-seed evaluation.
7. Compare against greedy and centralized oracle baselines.

Before trusting results, verify four-corner spawns, shelf geometry, map independence, LiDAR
map updates, delayed messages, action arbitration, zero-contact safety, resource
conservation, and reproducibility across fixed seeds. V3 is only an oracle/control
condition; it is not evidence of decentralized coordination.