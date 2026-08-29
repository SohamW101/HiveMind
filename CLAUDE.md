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

- ~~`_get_obs()` returns `[]`~~ — **done**, see step 3 below: pinned at 81 floats/robot
- ~~Reward is hard-coded `[0,0,0,0]`~~ — **done**, see step 4 below
- ~~`terminated` and `truncated` are hard-coded `False`~~ — **done**: terminates on all
  12 delivered, truncates at `max_steps`
- ~~Zero LiDAR ray-casting~~ — **done**: 72 rays per robot, in observation V3
- ~~Zero collision detection~~ — **done** for robot-robot pairs. Robots still pass
  *under* shelves (geometry, not detection — see step 4)
- No communication channel
- ~~`train.py` and `hivemind_env/models.py` are 0-byte placeholders~~ — **done**, step 6

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

### 3. Observations — DONE, V3 (2026-08-29)

**The observation is pinned at `OBS_DIM_V3 = 177` floats per robot.**
`observation_space` is `Box(-1.0, 1.0, (4, 177), float32)` — one row per robot, every
value normalised into `[-1, 1]`.

Two earlier widths were **superseded before anything was trained against either**, and
both are refused at construction with the reason:
- `OBS_DIM_V1 = 81` — no carton positions, so five warehouse layouts produced one
  byte-identical observation.
- `OBS_DIM_V2 = 105` — no LiDAR. Added when shelves became solid obstacles, which a
  robot otherwise had no way to perceive.

| slice | size | component |
|---|---|---|
| `[ 0: 3]` | 3 | own pose — x, y over arena half-extent; heading wrapped to `[0,1)` |
| `[ 3: 5]` | 2 | own velocity — last-step displacement in cells/step |
| `[ 5: 6]` | 1 | own carrying flag |
| `[ 6:15]` | 9 | other robots' poses (3 × 3, fixed agent order, self skipped) |
| `[15:18]` | 3 | other robots' carrying flags |
| `[18:30]` | 12 | carton status — 0.00 available / 0.33 mine / 0.67 other's / 1.00 delivered |
| `[30:54]` | 24 | carton positions — 12 × (x, y), same index as the status slots |
| `[54:56]` | 2 | depot direction — offset normalised by arena span |
| `[56:57]` | 1 | elapsed time — `current_step / max_steps` |
| `[57:129]` | 72 | LiDAR — 270° arc, 0.1–10 m, normalised, Gaussian noise |
| `[129:177]` | 48 | message slots — 3 others × 16 tokens, **all zero until step 7** |

129 world features + 48 reserved message slots = 177.

The **message slots are reserved now, zeroed**. This is the whole point of pinning: if
they were added at step 7 instead, the observation would grow 129 → 177 and every no-comms
checkpoint from step 6 would become unloadable — destroying exactly the before/after
comparison that is the contribution.

Three defences live in `hivemind_env/env.py`, and none of them is a comment:
1. Per-component constants are the single source of truth; nothing hard-codes 81 or 33.
2. `OBS_SLICES` is checked **at import** to tile `[0, 81)` with no gap or overlap — add a
   component without updating the total and the module refuses to import.
3. `__init__` rejects any `obs_dim` but the pinned one, and rejects the old `obs_size`
   argument with a sentence rather than a `TypeError`.

Verify with `.venv\Scripts\python.exe scripts/verify_observations.py` — it drives a
robot through a full pickup and delivery and checks all 52 assertions against PyBullet
ground truth — including a permanent regression test that distinct layouts produce
distinct observations, which is what V1 failed.

**A width change is a new `OBS_DIM_V4`, never an edit to V3.** The MAWC spec's own
numbers disagree (it says 132, its list sums to 130), which is why none of them was
adopted verbatim.

Decisions taken here that the roadmap left open, all recorded in env.py's header: poses
are the *snapped* grid poses (raw base transforms drift during substeps, which made a
one-cell move measure 0.992 cells); velocity is displacement, not `getBaseVelocity`;
carton status is one ordinal float, not a 4-way one-hot; other poses are absolute, not
relative. LiDAR **is** part of this observation as of V3 — 72 rays rather than the spec's
720, which would be seven times the rest of the vector; the FOV, range and noise
model are the spec's. The beam is cast at a fixed `LIDAR_BEAM_Z = 0.17`, chosen to
sit inside both the chassis band and the bottom shelf plate so the sensor sees
exactly what the body collides with.

### 4. Rewards and termination — DONE (2026-08-28)

Transcribed from `MAWC_Technical_Specification.pdf` §3, not from this file's earlier
summary. The constants live at the top of `hivemind_env/env.py` with the spec table
beside them.

**Shared (90% weight), identical for all four agents**

| Event | Reward |
|---|---|
| All 12 delivered | `+100.0` once per episode |
| Per successful delivery (any bot) | `+10.0` |
| Makespan bonus | `+50 × (T_max − T_actual) / T_max`, once per episode |
| Collision (any pair) | `−5.0` per event |
| Time penalty | `−0.05` per step |

**Individual (10% weight), per agent**

| Event | Reward |
|---|---|
| Own pickup | `+1.0` |
| Own delivery | `+2.0` |
| Idle (`v < 0.1`, not at depot) | `−0.02` per step |
| Invalid action | `−0.5` |
| Replanning (A* re-triggered) | `−0.1` — **not implemented, see below** |

`R_total_i = 0.90 × R_shared + 0.10 × R_individual_i`

`terminated` when all 12 are delivered; `truncated` at `max_steps`.

**Three places the spec needed a decision, all recorded in env.py:**

1. **The replanning penalty has no trigger here and is not implemented.** It fires when
   A* re-runs, and decision 3 above puts A*, DWA and EKF out of scope — robots are
   placed on a grid, not driven along a planned path. `R_REPLAN_PENALTY` is defined and
   deliberately unused so the omission is visible rather than silent.
2. **"Per collision event" is billed on contact onset, not per step.** Robots are
   teleported, so an overlapping pair stays overlapped until one moves; charging every
   step would bill `−5.0` repeatedly for one mistake.
3. **The idle penalty is on linear velocity, so a turning robot counts as idle** — that
   is the spec's letter (its velocity component is `(v, ω)`) and it is implemented
   literally. Pass `idle_penalises_turning=False` to exempt turns if step 6 shows the
   policy avoiding them.

`T_max` is `max_steps`; this env counts steps, not the spec's seconds. The formula is
otherwise unchanged.

Verify with `.venv\Scripts\python.exe scripts/verify_rewards.py` — it drives a robot
through a full delivery printing the reward and its breakdown **every step**, then
checks all 29 assertions, including that `R_total_i` equals the weighted sum recomputed
by hand, and that termination, truncation and the makespan bonus behave.

**Shelves are solid as of 2026-08-29.** The bottom plate was lowered from 0.30 to 0.18
so it spans 0.14–0.22 and overlaps the chassis (0.094–0.194) and wheels. Robot-vs-shelf
and robot-vs-wall contacts are now charged as collisions alongside robot-vs-robot —
"let the collision penalty do the work" rather than blocking the move, so a wedged robot
must learn its way out. Regenerate the URDFs after touching `SHELF_HEIGHTS`.

**Also fixed while nearby:** the chassis was sinking ~0.17 mm per step because `step()`
snapped x, y and yaw but never z — 0.051 m per 300 steps, ~0.34 m over a full episode,
i.e. through the floor. It went unnoticed until the LiDAR beam depended on height. `z` is
now snapped to the settled spawn height like the other axes.

### 5. Greedy baseline

Each robot claims the nearest unclaimed carton, drives there, delivers, repeats. Measure
makespan over 30 fixed seeds. **This is the number the learned policy has to beat.**
Build it before training so "it trains" is never mistaken for "it works".

### 6. Train without communication — PIPELINE DONE (2026-08-29), unmeasured

`hivemind_env/vec_env.py`, `hivemind_env/models.py` and `train.py` are built and the
loop runs end to end: train → save → load → evaluate.

- **`HiveMindSharedPolicyVecEnv`** presents N four-robot worlds as 4N single-agent slots
  sharing one policy — the roadmap's "cheap option first". Slot ordering is world-major;
  all four slots of a world go `done` together, with SB3's auto-reset contract honoured
  (`terminal_observation` and `TimeLimit.truncated` per slot).
- **`HiveMindExtractor`** splits the flat vector by `OBS_SLICES` into three branches:
  world features → MLP, the 72 LiDAR returns → 1-D CNN (adjacent rays are adjacent
  bearings, so a conv learns "obstacle to my left" once instead of 72 times), messages →
  MLP. 227k params. The message branch runs on zeros today so step 7 changes nothing
  architectural.
- **`train.py --smoke`** does 4096 robot-steps in ~2 min and exercises every path.

**This is parameter sharing with a DECENTRALISED critic, not MAPPO.** The critic sees
one robot's observation, not joint state. Tolerable here only because the observation is
already close to global (all four poses, all twelve carton statuses and positions); what
it misses is the others' LiDAR and, later, their messages. If step 6 plateaus below the
greedy baseline, replace this with a real asymmetric critic rather than tuning it.

**Nothing has been measured.** The only run so far is a 4096-step smoke run, which
produces a policy that collides 14.5 times per episode and delivers nothing — as it
should at that budget. Step 5 has to exist before any curve here means anything.

**Throughput is the practical constraint: ~65 robot-steps/s, ~34/s with PPO updates.**
2M robot-steps is roughly 16 hours. The 30 physics substeps per step dominate (LiDAR is
~4 ms of a 61 ms step). Note `--worlds` buys batch diversity but **no** speed: worlds
step sequentially in one process. Real parallelism needs subprocess workers; lowering
`num_substeps` is the other obvious lever and is a motion change, so it wants its own
decision.

TensorBoard is optional and not installed here — `train.py` detects that and runs
without curves rather than dying inside `learn()`.

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
