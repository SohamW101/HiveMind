# Training Guide

How to train, evaluate and interpret a policy for the HiveMind multi-agent warehouse.

Four robots must collect twelve cartons from shelf aisles and deliver them to a depot,
as fast as possible, with no central controller telling them who does what. This guide
covers the no-communication run; the communication phase builds on it unchanged.

---

## 1. The number that matters

Everything is measured against one figure.

> **Greedy baseline: makespan 97 steps at 12 cartons, 100% completion over 30 fixed seeds.**

A scripted controller — each robot claims the nearest unclaimed carton, delivers it,
repeats — clears the warehouse in a mean of 97 steps, travelling 233 m and collecting
6.7 collisions per episode.

There is a baseline for every curriculum level, all at 30/30 completion:

| Cartons | Makespan | Distance | Collisions | Reward per agent |
| --- | --- | --- | --- | --- |
| 4 | 23 | 40.5 m | 2.8 | 175.3 |
| 8 | 58 | 122.4 m | 4.5 | 222.9 |
| 12 (full task) | **97** | 232.6 m | 6.7 | 271.1 |

Quote a policy against the level it was trained on. Reward per agent includes the shaping
term, which is an addition to the specification's table — see §11 — so quote **makespan**,
not reward, when reporting.

**A policy that does not beat 97 has not demonstrated anything**, however smoothly its
reward curve rises. Reproduce the baseline on any new machine before trusting a result
from it:

```powershell
.venv\Scripts\python.exe scripts\run_evaluation.py --baseline greedy --episodes 30
```

Expect `Makespan : 23 / 58 / 97` across the three levels and `Completed : 100%`
throughout. Takes about two minutes. If your machine produces different numbers,
something differs in the physics build and no result from it is comparable to
`docs_analysis/greedy_baseline.json`.

The baseline is deliberately made a *fair* opponent — it reverses instead of turning
twice, never wastes a turn lining up a grab, and stops as soon as a target is in range.
A hobbled baseline would quietly flatter everything measured against it.

---

## 2. Quick start

```powershell
# 1. are the incentives sane? six gates, under a minute. never skip this
.venv\Scripts\python.exe scripts\diagnose_incentives.py --num-cartons 1

# 2. does the pipeline run on this machine? prints the device and the fps
.venv\Scripts\python.exe train.py --smoke --worlds 4

# 3. the real run - starts at ONE carton and promotes 1 -> 2 -> 4 -> 8 -> 12
.venv\Scripts\python.exe -u train.py --run-name main --num-cartons 1 --curriculum ^
    --timesteps 5000000 --worlds <physical cores> --checkpoint-every 25000

# 4. what is the checkpoint actually doing?
.venv\Scripts\python.exe scripts\probe_policy.py models\checkpoints\<file>.zip --num-cartons 1

# 5. score it against the baseline
.venv\Scripts\python.exe scripts\run_evaluation.py --model models\main_final.zip --episodes 30
```

**Start at one carton, not four.** An episode ends only when *every* carton is delivered,
so `success_rate` is a conjunction over four robots. At 4 cartons an untrained policy
never satisfies it — 0 completions in 15 random episodes — and therefore never observes
the +100 completion bonus at all. At 1 carton a random policy completes 33% of the time.
Section 11 has the full story; it is the single reason four earlier runs learned nothing.

---

## 2b. Running the full campaign on another machine

The plan is a **5,000,000-step curriculum run**. Everything here is what actually matters
when the run is on a machine you are not sitting in front of.

**Pick `--worlds` from physical cores, not from the GPU.** Each world is one PyBullet
process on one core. Match the physical core count — oversubscribing threads makes
throughput worse, not better. On 16 cores, expect 5M steps in roughly 3-5 hours.

**Checkpoint often. This is not optional.**

```powershell
--checkpoint-every 25000
```

Two runs were interrupted at 215k and 102k steps and both lost *everything*, because the
checkpoint interval was longer than the run survived. 25k costs a second of disk per save
and buys back hours.

**Install TensorBoard there.** It is deliberately not in `requirements.txt`, and training
runs silently without curves when it is missing — exactly what you do not want on a
machine you are checking in on remotely:

```powershell
.venv\Scripts\python.exe -m pip install tensorboard
tensorboard --logdir tensorboard_logs
```

**Run it detached and unbuffered.** Without `-u` the output sits in a pipe buffer and an
empty log looks identical to a hung run. It has happened here.

**What to check, in order.** Reward is the *last* thing to look at — a rising reward curve
is what fooled every earlier run:

| signal | healthy | dead |
|---|---|---|
| `rollout/ep_len_mean` | falls below the cap early | pinned at the cap |
| `rollout/success_rate` | leaves zero, then climbs | flat 0 |
| `curriculum/target_cartons` | steps 1 → 2 → 4 → 8 → 12 | never moves |
| `rollout/ep_rew_mean` | rises **and** the three above are moving | rises alone |

**Kill rules.** If `ep_len_mean` has not left the cap by 100k steps, stop the run; it does
not recover, and four runs proved that the expensive way. If the curriculum has not
promoted past 1 carton by around 500k steps, stop and re-run the diagnostic in step 1
before spending more compute on it.

**Where it should get to.** At 1 carton, `success_rate` reached 0.40 with `ep_len` 44.1
inside 92k steps on a modest CPU box, with mean episode reward positive for the first
time. Promotion needs 70% rolling success over the last 100 episodes. **Levels 2 through
12 have never been reached by any run**, so treat everything above 1 carton as genuinely
unknown — that is what this 5M run exists to find out.

---

## 3. Setting up a fresh machine

```powershell
git clone <repository-url>
cd HiveMind
git checkout multi-agent-rl

py -3.14 -m venv .venv                       # any Python >= 3.10 works
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

There is no `activate` step in the documented workflow — call the interpreter by path.

**TensorBoard is optional** and not in `requirements.txt`. Training detects its absence
and runs without curves rather than failing. Install it if you want them:

```powershell
.venv\Scripts\python.exe -m pip install tensorboard
```

**CUDA is optional too** — see §7 before assuming you need it. If you do want it, take
the exact index URL from pytorch.org for your CUDA version:

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url <url from pytorch.org>
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Verify before training

Run all six. They take about three minutes together and they are the difference between
debugging a bad checkpoint and never creating one. `diagnose_incentives.py` is the one
added after three training runs completed zero episodes; do not skip it.

| Command | Expected |
|---|---|
| `.venv\Scripts\python.exe smoke_test.py` | 16 PASS, 0 TODO, 0 FAIL |
| `.venv\Scripts\python.exe scripts\verify_observations.py` | 55 passed, 0 failed |
| `.venv\Scripts\python.exe scripts\verify_rewards.py` | 38 passed, 0 failed |
| `.venv\Scripts\python.exe scripts\verify_comms.py` | 31 passed, 0 failed |
| `.venv\Scripts\python.exe scripts\diagnose_incentives.py --num-cartons 1` | 6 gates PASS |
| `.venv\Scripts\python.exe scripts\run_evaluation.py --baseline greedy --episodes 30` | makespan 23 / 58 / 97, 100% |

`verify_comms.py` is worth running even for a no-communication run: its first section is
the assertion that `comms=False` is unchanged, which is what keeps that run a valid
control arm.

---

## 4. What is actually being trained

**One policy, shared by all four robots.** Each four-robot warehouse is presented to the
learner as four independent slots that happen to share weights, so experience from every
robot in every world pools into one training batch. The robots still act simultaneously
on the same physics step, so collisions, races for the same carton and the shared 90% of
the reward are all real.

**The network** splits the 177-float observation into three branches rather than
flattening it: world features through an MLP, the 72 LiDAR returns through a 1-D
convolution — adjacent rays are adjacent bearings, so a wall is a run of neighbouring
slots and a convolution learns "obstacle to my left" once instead of 72 times — and the
message block through its own small MLP. 227k parameters.

**The message branch already runs, on zeros.** The 48 message slots are reserved in the
observation and the network reads them, so adding communication later changes neither the
observation width nor the architecture. That is what keeps the eventual before-and-after
comparison clean.

### One simplification, stated plainly

This is **parameter sharing with a decentralised critic, not MAPPO**. The critic values a
state from one robot's observation rather than the joint state of all four.

That is tolerable here for a specific reason rather than as a shortcut: this observation
is already close to global — every robot sees all four poses, all twelve carton statuses
and positions, and the clock. What it misses is the other robots' LiDAR returns and,
later, the messages they are about to send.

If the policy plateaus below the baseline, this is the first thing to replace, and the
honest replacement is a real asymmetric critic — not more hyperparameter tuning.

---

## 5. Training options

```
--timesteps N       Total robot-steps (not world-steps). Default 2,000,000.
--worlds N          Parallel warehouses. Each contributes 4 policy slots.
--backend           subproc (default) or inprocess.
--substeps N        Physics substeps per env step. Default 5 headless, 30 GUI.
--lr FLOAT          Initial learning rate, decayed linearly to zero. Default 3e-4.
--n-steps N         Rollout length per slot. Buffer is n-steps x 4 x worlds.
--batch-size N      Minibatch size for the update. Default 1024.
--seed N            Seeds the policy and the per-world warehouse generation.
--max-steps N       Episode cap. Defaults per carton count: 150 / 250 / 400 at 4 / 8 / 12.
--num-cartons N     Cartons in play. Default 12. Start at 1 - see section 2.
--shaping-scale F   Strength of the potential-based shaping. Default 60.0.
--no-shaping        Train the specification's sparse reward exactly. Ablations only.
--gamma FLOAT       Discount. Default 0.99.
--curriculum        Promote 4 -> 8 -> 12 cartons on rolling success.
--comms             Turn on the 16-token broadcast channel (step 7). Off by default.
--masked            Train with MaskablePPO; invalid actions cannot be selected. A
                    DIFFERENT ALGORITHM - keep both arms of the comms comparison on
                    the same setting.
--msg-dropout F     Per listener-speaker link dropout. Default 0.10. Needs --comms.
--ent-coef F        PPO entropy bonus. Default 0.01. Read the note in section 14 before
                    changing it - with --comms it is a standing pressure toward a
                    meaningless token distribution.
--checkpoint-every  Save every N robot-steps. Default 250,000.
--run-name NAME     Names the checkpoint and the TensorBoard run.
--smoke             4,096 steps. Exercises every code path in about a minute.
```

`--timesteps` counts **robot-steps**, not world-steps. With 12 worlds each step of the
simulation produces 48 robot-steps, so 2,000,000 robot-steps is about 41,000 steps of
each warehouse — roughly 20 full episodes per world.

`--curriculum` sets `num_cartons` on every environment as the rolling success rate
crosses its threshold, walking 4 → 8 → 12. It used to promote a `difficulty_level` that
nothing in the world read, so it did nothing at all; that is fixed.

---

## 6. Throughput

Two settings dominate, and both are already at sensible defaults.

**`--backend subproc`** puts each warehouse in its own process. Training is dominated by
single-threaded rigid-body physics, so this is the difference between using one core and
using the machine. `--backend inprocess` runs everything sequentially in one process:
slower, but a traceback inside a worker process is much harder to read, so use it when
something is broken.

**`--substeps 5`** controls how finely a one-cell move is interpolated. It is an
interpolation, not the motion model — the final pose is the snapped grid target and
collisions are read from that pose, so **makespan, collisions, deliveries and completion
are identical at any value**. Verified across the full 30-seed baseline at 30, 10, 5 and
1 substeps: identical to the decimal every time.

Measured on a 16-thread / 8-core machine:

| Configuration | robot-steps/s | 2M steps |
|---|---|---|
| in-process, 30 substeps | 34 | ~16 hours |
| 12 subprocess workers, 30 substeps | 133 | ~4 hours |
| **12 workers, 5 substeps (defaults)** | **844** | **~40 minutes** |

Scaling is sub-linear in worker count because physical cores, not threads, are the limit.
Start with `--worlds` equal to your physical core count and adjust from there.

Lowering substeps to 1 gains a further 2.6× but makes the robot jump a full metre in one
go, which could tunnel past the 6 cm shelf posts. There is no evidence it does — collision
counts are identical — but it is a thin margin, and thinner still once a
velocity-controlled motion model replaces the current teleport.

---

## 7. Does a GPU help?

**Not as much as you would expect, and it depends on what else you have configured.**

The workload is rigid-body physics, not matrix multiplication. Before the parallelism and
substep work, a rollout split like this:

```
env stepping   95.0%     physics, CPU only
policy forward  1.3%
PPO update      3.8%     the only part a GPU touches
```

At that point an *infinitely fast* GPU would have bought about 5%.

With the current defaults the environment is roughly 5× faster, so the balance is now
close to half physics and half policy update — and a CUDA GPU is genuinely worth having.
The policy is small (227k parameters on a 177-float vector), so measure rather than
assume: run `--smoke` on both devices and compare the reported `fps`. Small networks are
sometimes faster on CPU because transfer overhead outweighs the compute saved.

Two details in the device probe worth knowing: it requires compute capability 7.0 or
higher and falls back to CPU otherwise, and it **disables cuDNN** — a stability workaround
inherited from the single-agent phase that may hurt the LiDAR convolution on modern
hardware. Benchmark both ways.

**Prioritise core count over GPU** when choosing a machine for this. If you are picking
between a 32-core box with no GPU and an 8-core box with an A100, take the 32 cores.

### Checklist for a GPU machine

Run this before committing hours to it:

```powershell
.venv\Scripts\python.exe train.py --smoke --worlds 4
```

The first line of output is the device probe. Three ways it disappoints:

1. **`No CUDA GPU detected`** — the CPU wheel of PyTorch is installed. Reinstall from the
   index URL pytorch.org gives for that CUDA version (section 3).
2. **`is below sm_70. Falling back to CPU`** — the card is too old. A GTX 1080 is sm_61
   and gets nothing here. The run still works, just on CPU.
3. **It says CUDA and the fps is no better than CPU.** Entirely possible and not a bug:
   the policy is 227k parameters on a 177-float vector, so transfer overhead can exceed
   the compute saved. Compare the `fps` line from both and use whichever wins.

Note the probe **disables cuDNN** when it selects CUDA — a stability workaround inherited
from the single-agent phase. It may now be costing throughput on the LiDAR convolution.
If you are benchmarking anyway, that line in `hivemind_env/training.py` is worth toggling.

---

## 8. Monitoring a run

```powershell
tensorboard --logdir tensorboard_logs
```

What to watch, in order of usefulness. **`ep_rew_mean` is last, not first** — it rose
steadily through all four failed runs and is how three of them were misread.

| Metric | What it should do |
|---|---|
| `rollout/ep_len_mean` | **Fall, and leave the cap.** This is makespan. Pinned at exactly `max_steps` is the single clearest sign of a dead run. |
| `rollout/success_rate` | Rise from zero. All-or-nothing over every carton, so it cannot tell "delivers none" from "delivers all but one". |
| `rollout/delivered_fraction` | Rise. The metric `success_rate` cannot give you — read the two together (§11b). |
| `curriculum/target_cartons` | Step 1 → 2 → 3 → 4 → 8 → 12. Never moving, or moving and then never recovering, are both failures. |
| `curriculum/steps_at_level` | Reset on each change. Growing past ~30% of the run triggers an automatic demotion. |
| `train/entropy_loss` | Fall from its maximum as the policy commits. Stuck at −1.94 means it has not committed to anything. |
| `train/explained_variance` | Rise towards 1. Near zero means the critic is not learning. |
| `train/approx_kl` | Stay under about 0.02. Consistently higher means the learning rate is too high. |
| `rollout/ep_rew_mean` | Rise — but only meaningful **against the do-nothing floor** and the greedy reference. On its own it says nothing. |

With `--comms`, add `comms/token_entropy_bits` and `comms/mi_carrying_bits`, and read
them **together** — see §14.

Console output reports `fps`; compare it against §6 to confirm you are getting the
throughput your machine should give.

Checkpoints land in `models/checkpoints/` every `--checkpoint-every` robot-steps, and the
final model in `models/<run-name>_final.zip`. Interrupting with Ctrl+C still saves.

---

## 9. Evaluating a checkpoint

```powershell
.venv\Scripts\python.exe scripts\run_evaluation.py --model models\<run>_final.zip --episodes 30
```

The same harness, the same 30 fixed seeds and the same makespan definition the baseline
was scored with. That identity is deliberate: a policy and a baseline measured different
ways are not a comparison.

Useful flags:

```
--episodes N      Episodes per level. 30 for a real result, fewer to smoke-test.
--policy-mode     shared (default) queries the policy once per robot. joint expects a
                  single policy whose action space is already MultiDiscrete.
--max-steps N     Per-episode budget enforced by the harness itself.
--out PATH        Where the JSON goes. Refuses to overwrite a larger run without --force.
```

Read `avg_makespan` first, then `completion_rate`. **A low makespan with poor completion
is not a win** — makespan is only defined over completed episodes, so a policy that
finishes two easy seeds quickly and times out on the rest will show a flattering number.
Always read them together.

---

## 10. Interpreting the result

| Outcome | Reading |
|---|---|
| Makespan < 98 at 100% completion | **A real result.** The policy beats greedy. |
| Makespan ≈ 98 | Learned the task, found no better division of labour than "nearest carton". |
| Makespan > 98 at 100% | Learned the task but coordinates worse than greedy. |
| Completion < 100% | Has not reliably learned to finish. Makespan is not meaningful yet. |
| Reward rising, completion at zero | Collecting shaping reward without delivering. Check delivery counts in the JSON. |

Some caveats worth carrying into any write-up:

- **98 is a greedy number, not an optimal one.** It says what an unsophisticated
  controller achieves — the right bar for "has the policy learned anything useful", not a
  ceiling. Beating it does not mean the schedule is optimal.
- **The reward structure implements nine of its ten specified terms.** The replanning
  penalty is deliberately absent because there is no planner in this environment to
  re-trigger.
- **Robots can push cartons.** They are solid bodies; a robot driving into one moves it.
  The observation reports live positions so the policy is unaffected, but it means the
  world is not static.
- **Shelves are solid and cost −5.0 on contact**, so routing through an aisle row is
  penalised rather than free.

---

## 11. The failure modes, and how each was found

Four runs have failed here in three distinct ways. Every one looked healthy on at least
one curve while it happened, so read this before you start.

### 11a. "Learned to stand still" (runs 1-3, Aug 2026)

`ep_len_mean` pinned at exactly `max_steps`, `success_rate` flat 0, and `ep_rew_mean`
rising steeply while `explained_variance` sat near 0.9 — every optimisation signal saying
the run was healthy. It was: PPO was correctly optimising a reward maximised by doing
nothing.

**The tell is arithmetic, not intuition.** Compute what a do-nothing policy scores and
compare. If the plateau sits at or below that floor, the reward is the bug. At 400 steps
the floor is &minus;18.8 and one run finished at &minus;7.3; at 2000 steps it was
&minus;94 and the run converged to &minus;103 — *below* doing nothing.

Fixed by blocking moves into shelves rather than charging for them, making a pickup pay,
lifting shaping outside the 90/10 split, and setting the scale from `EV(move)`.

### 11b. "Working but not finishing" (run 4, `nocomm2`, 20M steps, Sep 2026)

The opposite signature, and the one you are most likely to hit next. **`ep_rew_mean` rose
80 → 100 across the entire dead region** — well above the do-nothing floor — while
`success_rate` sat at exactly 0.00 for 16.4M steps and `ep_len_mean` at exactly 150.

The policy was working. It just never closed an episode:

```
nocomm2_final at 4 cartons, 10 episodes
  argmax     delivered 1.9/4   pickups 2.6   completed 0/10
  sampled    delivered 2.8/4   pickups 3.3   completed 0/10
  greedy     delivered 4.0/4   makespan 23   completed 30/30
```

Termination is an **AND over every carton**, so delivering 3 of 4 earns nearly all the
reward and zero success. **`success_rate` cannot distinguish "delivers nothing" from
"delivers all but the last one".** That is why `rollout/delivered_fraction` now exists —
watch it beside `success_rate`, because the two failures need opposite fixes:

| delivered_fraction | success_rate | meaning |
|---|---|---|
| rising | flat 0 | competent; something stops it closing. Look at the last carton |
| flat low | flat 0 | not working at all. Go back to 11a |

### 11c. The three defects run 4 exposed

**Euclidean distance in a warehouse full of shelves.** Φ measured straight-line distance
to the target, but robots have to drive around six shelf rows. Measured over seeds
1000-1009, **10.2% of free cells were local minima** — every legal move increased the
distance, so the shaping punished all of them and the robot was paid to stand still.
Φ is now geodesic (BFS over enterable cells), which has no local minima by construction.

**Idle robots had no gradient at all.** Φ's target search skipped cartons already in
someone's gripper, so a robot with nothing left to claim got a constant Φ. Since only
movement can collide and an invalid PICKUP cannot, its cheapest action was to grab at
thin air: **1,480 of 6,000 robot-steps were PICKUP with nothing in reach**, 25% of the
episode, with movement at 21%. Φ now falls back to the nearest undelivered carton
wherever it is.

**The curriculum was one-way.** It promoted to 4 cartons at 3.6M and had no route back,
so 82% of the run trained against an all-zero success signal. That is not neutral — it
**destroyed** the 2-carton ability the policy already had, from ~55% success down to
0/10. The ladder now demotes on sustained failure or on spending 30% of the run at one
level, and gained a 3-carton rung so the 2 → 4 doubling is no longer where it breaks.

### 11d. Check the incentives before you train — it costs seconds

```powershell
.venv\Scripts\python.exe scripts\diagnose_incentives.py --num-cartons 4
```

Six gates. The one that matters most is **`EV(move) > 0`**: only movement can collide, so
the collision penalty is a risk premium on the single action class that makes progress,
and when it exceeds the shaping gain the optimal policy is to turn, grab and freeze.

Passing every gate does **not** promise the policy will learn. It means the reward is not
the reason if it does not.

### 11e. And check what a checkpoint actually does

```powershell
.venv\Scripts\python.exe scripts\probe_policy.py models\run_final.zip --num-cartons 4
.venv\Scripts\python.exe scripts\probe_pickup.py models\run_final.zip --num-cartons 4
```

`probe_policy` states a verdict — standing still / thrashing / moving but not working /
working but not finishing / completing — which `ep_rew_mean` cannot, and three runs were
misread because of it. `probe_pickup` conditions on the one state that decides the task:
a carton in reach and an empty gripper. Run both argmax and sampled; they disagree
sharply, and a policy is not broken merely because its argmax is.

---

## 12. Troubleshooting

**`ImportError: Trying to log data to tensorboard`** — should not happen; training probes
for TensorBoard and disables logging if absent. If you see it, you are on an older
revision.

**Workers hang or the run will not start** — anything constructing the subprocess backend
must sit behind `if __name__ == "__main__":`. The spawn start method re-imports the module
in every child, so without the guard each worker starts its own training run. `train.py`
is already guarded; custom scripts need the same.

**Throughput far below §6** — check `--backend` is `subproc`, and that `--worlds` is not
far above your physical core count. Oversubscribing threads makes it worse, not better.

**Baseline makespan is not 97** — do not train until this is resolved. Something differs
in the physics build, and nothing measured on that machine is comparable. Per level the
reference is **23 / 58 / 97** at 4 / 8 / 12 cartons, all 30/30 over seeds 1000-1029,
2000-2029 and 3000-3029.

**Every evaluation level reports the same makespan and "Delivered 12/12"** — you are on a
revision before 2026-08-31. `run_evaluation.py` passed `difficulty_level` to the world,
which stores it and never reads it, and never passed `num_cartons` — so all three levels
ran the full task while the summary table labelled them 4, 8 and 12. Any per-level number
from that harness is a 12-carton number whatever its row said.

**A shape error loading a checkpoint** — the observation width is baked into the policy's
input layer. The environment pins it at 177 and refuses any other value with an
explanation naming the superseded width. A model trained against an older width cannot be
loaded; retrain it.

**Policy plateaus below the baseline** — first check §11: if `ep_len_mean` is pinned at
`max_steps` and `success_rate` is zero, it has learned to stand still and no amount of
extra training will help. Score a hard-coded `stay` policy on the same settings before
concluding anything from a reward curve: the second failed run converged to -16.5 against
a `stay` score of -17.2, and the 29x reward "improvement" that produced was entirely the
policy learning to freeze. `scripts/diagnose_incentives.py` prints both numbers. If it is completing episodes but slowly, confirm
`explained_variance` is rising (flat means the critic is the problem, and the
decentralised critic in §4 is the first thing to replace), then try a longer run.

---

## 13. Where this actually stands

Worth being blunt about, because the numbers above describe a system that is fixed but
not yet finished.

**Established.** The environment, the reward, and the diagnostics are verified: 38 reward
checks, 53 observation checks, 16 smoke checks, and a greedy baseline of 23 / 58 / 97
steps at 4 / 8 / 12 cartons with 30/30 completion at every level. Seven defects that made
learning impossible have been found, fixed, and each has a measurement attached.

**Demonstrated once, briefly.** At 1 carton a policy reached `success_rate` 0.40 with
`ep_len` 44.1 in 92k steps, mean episode reward positive. That run was interrupted and
checkpointed nothing, so the artefact does not exist any more — only the log does.

**Not established at all.** No policy has been trained past 1 carton. The curriculum has
never promoted. Nothing has been scored against the greedy baseline. **No learned policy
in this repository has ever completed a 12-carton episode**, and the headline claim —
beating makespan 97 — has not been attempted, let alone met.

Any write-up should say that plainly. Phase 1's `docs_analysis` set its standard with a
document that openly reported the environment forcing an action the policy was then
measured on; matching that standard matters more here than a confident-sounding number.

---

## 14. After this run: the communication comparison

The no-communication policy is the control condition. The channel it is compared against
was built on 2026-09-02 and is off by default, so the two arms of the experiment are the
same command differing in one flag:

```powershell
# control
.venv\Scripts\python.exe -u train.py --run-name nocomm --num-cartons 1 --curriculum ^
    --timesteps 5000000 --worlds <cores> --checkpoint-every 25000

# treatment - identical but for --comms
.venv\Scripts\python.exe -u train.py --run-name comms --num-cartons 1 --curriculum ^
    --timesteps 5000000 --worlds <cores> --checkpoint-every 25000 --comms
```

With `--comms` each robot emits one of 16 tokens per step and hears the other three one
step later, through 10% per-link dropout. The observation width (177) and the network are
identical either way, which is the entire reason the slots shipped reserved and zeroed
rather than being added now.

**Run the control first and to completion.** A communicating run with no baseline to
compare against measures nothing.

### What to watch, and the trap in it

`--comms` adds four TensorBoard series. Read the first and the last **together**:

| series | healthy | what it looks like when the channel is unused |
|---|---|---|
| `comms/token_entropy_bits` | anywhere below the ceiling | pinned at **4.0**, its maximum |
| `comms/tokens_used` | some subset of 16 | all 16, evenly |
| `comms/top_token_share` | one or two symbols carrying real mass | ~6% each, flat |
| `comms/mi_carrying_bits` | rises above 0 | **0.00** forever |

**High entropy is the null result, not the positive one.** PPO's entropy bonus is summed
over both action heads, and the token head's maximum (ln 16 = 2.77) is larger than the
movement head's (ln 7 = 1.95), so an ignored token head drifts to uniform on its own.
Measured on the callback directly: a perfect protocol scored **2.78 bits of entropy with
0.97 bits of mutual information**, while pure noise scored **4.00 bits with 0.00**. The
informative channel had the *lower* entropy.

If entropy pins at 4.0 while `mi_carrying_bits` stays at zero past ~1M steps, lower the
entropy bonus: `--ent-coef 0.003`.

### Reporting the result

```powershell
.venv\Scripts\python.exe scripts\analyse_messages.py models\comms_final.zip ^
    --num-cartons 4 --episodes 30 --out docs_analysis\messages.json
```

It answers three questions and a protocol claim needs all three: is there information in
the tokens (mutual information against a shuffled floor), does the listener react (KL on
the movement distribution when the message slots are zeroed), and does breaking the
channel cost makespan (the same seeds re-run under `learned` / `silent` / `shuffled` /
`random`).

`shuffled` is the one that decides it — real tokens from the same step attributed to the
wrong speakers, so the input distribution is untouched and only the meaning is destroyed.

Report makespan as the headline, with distance travelled, collision count and the message
analysis alongside. If the channel turns out to be unused, report that: "the robots were
given a channel and did not use it" is a real finding, and it is the more likely one on a
first run.
