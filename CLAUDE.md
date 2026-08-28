# HiveMind — working notes for Claude Code

Read this before touching anything. It captures decisions already made, the state of the
code as of 2026-08-28, and the order work should happen in. Full audit with reasoning:
https://claude.ai/code/artifact/b8f764e5-a1bd-4db0-b922-773267c81d7a

---

## What this project is

Four warehouse robots learn to divide up a delivery job between themselves — and to
develop a communication protocol that helps them do it — using multi-agent reinforcement
learning in PyBullet. Twelve cartons sit in shelf aisles; the robots must collect them all
and deliver them to a depot, as fast as possible, without a central controller telling
them who does what.

The research claim is about **emergent communication**, not about navigation. Keep that in
mind when trading off effort.

---

## Branch map

| Branch | State | Contents |
|---|---|---|
| `single-agent-rl` | **Finished, Aug 4–9 2026.** Do not develop here. | One robot, 180-ray LiDAR, PPO trained 10M steps, 79.2% success over 120 episodes, 12 analysis docs, trained model, TensorBoard logs |
| `multi-agent-rl` | **Active.** All work happens here. | The 4-robot warehouse simulator. World is built; learning system is not |

**These branches share no git history** — `multi-agent-rl` is an orphan, not a
continuation. That means it inherited none of the working code from Phase 1. Several
hundred lines of already-debugged infrastructure sit one branch away and should be
**ported, not rewritten** (see step 2 below).

---

## Decisions already made — do not relitigate these

1. **Simulation only.** No TurtleBot3, no real hardware target. URDF fidelity to the spec
   does not matter.
2. **Route C motion model.** Keep the existing grid-teleport movement for now so the
   learning loop can be built quickly — but isolate *all* movement into a single method
   (`_execute_motion()`) so a velocity-controlled physics version can replace it later
   without touching reward, observation or training code.
3. **Out of scope as a consequence:** EKF localisation, DWA local planning, pure-pursuit
   wheel control. These require robots that are *driven* rather than *placed*. They are
   in the MAWC spec; they are not in this plan. Revisit only if the hardware decision
   changes.
4. **Gripper stays as-is.** "Within range + pickup action = attached" works, looks right,
   and never fails for physics reasons. Do not build constraint-based grasping.
5. **Communication is unlimited range for v1.** A distance cutoff is a later ablation, not
   a starting condition — it makes failures impossible to diagnose.
6. **Collisions penalise and continue.** Do not terminate an episode on contact.

---

## Current state of `multi-agent-rl`

### What works and should not be broken

- `hivemind_env/env.py` — `HiveMindMultiAgentEnv`, ~400 lines
  - 13x13 m arena on a 1 m grid, procedurally regenerated every `reset()`
  - 6 shelf rows (grid rows 1,3,5,7,9,11), each randomly cut into 3 segments with 2 gaps;
    cut points change every reset
  - Exactly 12 cartons, one per gap. Boundary walls, floor, depot at grid `(0,0)`
  - 4 robots spawn at cells `(0,1) (1,0) (0,2) (2,0)`
  - `MultiDiscrete([7,7,7,7])`: 0 fwd, 1 back, 2 turn L, 3 turn R, 4 pickup, 5 drop, 6 stay
  - Pickup/drop are fully working — arm swings to face the target, gripper closes, LiDAR
    mast rises to 0.5 m while carrying, carton travels with the gripper
  - Motion is interpolated over 30 physics substeps; pose is snapped back to exact grid
    centres and 90-degree headings each step to prevent drift
- `hivemind_env/assets/` — robot URDF, carton, and pre-generated shelves at every length
  1 m–7 m (`generate_shelves.py` regenerates them)
- `play_multi.py` — scripted GUI demo: BFS to the nearest carton, turn, pick up, stop

### What does not exist yet

- `_get_obs()` returns `[]`. There is **no `observation_space` declared at all**
- Reward is hard-coded `[0,0,0,0]` every step
- `terminated` and `truncated` are hard-coded `False`. `self.current_step` increments and
  nothing ever reads it — the 2000-step limit is not enforced
- Zero LiDAR ray-casting (no `rayTest` calls anywhere)
- Zero collision detection (no `getContactPoints` calls) — robots pass through each other
- No communication channel
- `train.py` and `hivemind_env/models.py` are **0-byte placeholders**

---

## Roadmap, in order

Each step should end with something runnable. Nothing here depends on a later step.

### 1. Line endings (do first, once)

Every file shows as modified in `git status` because of CRLF vs LF, which makes real
diffs unreadable. Add `.gitattributes` containing `* text=auto`, commit it, then
`git add --renormalize .` and commit again. Expect the second commit to touch every file.

### 2. Port infrastructure from `single-agent-rl` — do not rewrite

    git checkout origin/single-agent-rl -- hivemind_env/training.py
    git checkout origin/single-agent-rl -- scripts/run_evaluation.py
    git checkout origin/single-agent-rl -- smoke_test.py

Then adapt for 4 agents. `training.py` has the curriculum callback, linear LR schedule,
parallel-env factory and device probe. `run_evaluation.py` has a fixed-seed scored
evaluation harness. Also worth reading (do not necessarily copy): the `_get_lidar_scan`
method in the single-agent `env.py`, and `models.py`'s `CustomCombinedExtractor`.

### 3. Observations

Declare `observation_space` and make `_get_obs()` return one vector per robot:
own pose (3) + velocity (2) + carrying flag (1) + other robots' poses (9) + other
carrying flags (3) + status of all 12 cartons (12: available / claimed-by-me /
claimed-by-other / delivered) + depot direction (2) + elapsed time (1). Leave message
slots as zeros for now.

**Pin the dimension and write it down.** The MAWC spec says 132 but its own component
list sums to 130. Phase 1 was bitten by exactly this — the CNN flatten width gets baked
into saved weights, so a model can only be loaded into an env of the same size.

### 4. Rewards and termination

Implement the 90/10 shared/individual split:
- shared: +10 per delivery, +100 when all 12 done, makespan bonus
  `50 * (T_max - T_actual) / T_max`, -5 per collision, -0.05 per step
- individual: +1 own pickup, +2 own delivery, -0.02 idle
- `R_total_i = 0.90 * R_shared + 0.10 * R_individual_i`
- `terminated = True` when all 12 delivered; `truncated = True` at `max_steps`
- collisions via `pb.getContactPoints` between robot bodies

Verify with `play_multi.py` before any training — drive one bot through a full delivery
and print the reward each step.

### 5. Greedy baseline

Each robot claims the nearest unclaimed carton, drives there, delivers, repeats. Measure
makespan over 30 fixed seeds. **This is the number the learned policy has to beat.**
Build it before training so "it trains" is never mistaken for "it works".

### 6. Train without communication

Fill `models.py` (shared actor, centralised critic) and `train.py`. All four robots share
one set of weights; no messaging yet. Get it approaching or beating the greedy baseline.

Practical note: Stable-Baselines3 does not do MAPPO out of the box. **Try the cheap option
first** — wrap the env so the 4 robots look like 4 parallel single-agent envs sharing a
policy. Only reach for a dedicated multi-agent PPO implementation if that plateaus.

### 7. Add communication

Only now. 16-token broadcast, message slots filled in the observation. Because step 6 gave
a no-comms baseline, the before/after comparison is clean — that comparison *is* the
contribution. Include ~10% message dropout during training so the protocol is not brittle.

### 8. Scale and evaluate

Parallel envs, the 4 -> 8 -> 12 carton curriculum, then the ported evaluation harness with
fixed seeds. Track makespan (headline), distance travelled, collision count, and message
entropy — entropy is how you show a real protocol emerged rather than noise.

---

## Conventions

- Run everything from the repo root; asset paths are relative
- **No conda.** Use the repo-root `.venv` (Python 3.14, created 2026-08-28):
  `.venv\Scripts\python.exe <script>` — there is no activate step in the documented
  workflow. `hivemind_env` is installed editable, so imports resolve from any directory.
  `environment.yml` is kept as an alternative for conda users but is not the path in use.
- Regenerate shelves after changing shelf/carton geometry:
  `python hivemind_env/assets/generate_shelves.py`
- `reset(seed=...)` seeds both `random` and `numpy`, so shelf layouts are reproducible
- PyBullet GUI exits with code 130 on Ctrl+C — that is expected, not a crash
- Commit messages: use the conventional style from `single-agent-rl`
  (`fix:`, `feat:`, `docs:`, `chore:`), not the six commits named "base" on this branch

---

## Known issues worth fixing while nearby

- `play_multi.py` is misnamed — it drives one robot and never delivers. Rename or finish it
- `pyproject.toml` still describes "A Single-Agent PyBullet Environment for RL Foraging"
- `README.md` describes the environment accurately but predates all of the above
- MAWC spec inconsistencies: observation components sum to 130 not 132; resources R11/R12
  are placed at x=12.0 and x=14.5 but labelled "Row 4 aisle", whose shelf only spans
  x=5.0–9.0. The spec was AI-generated and should be treated as a proposal, not a contract
- The spec's section 10 proposes changing the research question — the old plan's value was
  emergent roles under partial observability, and MAWC gives every robot full map
  knowledge. That trade was accepted, but be aware it was a trade

## Documentation habit worth keeping

Phase 1's `docs_analysis/` numbered documents are the most credible thing in this
repository — particularly doc 12, which openly reports that the environment forces the
pickup action so the policy's measured pickup usage is 0.0%, and that the model was
trained before that change and evaluated after it. Keep that standard here. Honest
reporting of what a result does *not* show is worth more than the headline number.
