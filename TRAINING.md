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
`docs_analysis/greedy_baseline_blocked.json`.

The baseline is deliberately made a *fair* opponent — it reverses instead of turning
twice, never wastes a turn lining up a grab, and stops as soon as a target is in range.
A hobbled baseline would quietly flatter everything measured against it.

---

## 2. Quick start

```powershell
# 1. sanity run at the easiest curriculum level - start here, see section 11
.venv\Scripts\python.exe train.py --timesteps 400000 --worlds 12 --num-cartons 4

# 2. score the checkpoint against the baseline
.venv\Scripts\python.exe scripts\run_evaluation.py --model models\<run>_final.zip --episodes 30

# 3. the real run, with the curriculum walking 4 -> 8 -> 12
.venv\Scripts\python.exe train.py --timesteps 2000000 --worlds 12 --num-cartons 4 --curriculum
```

Do step 1 before step 3, and read §11 first. Four cartons is a 23-step task for greedy,
so `success_rate` should lift off zero early. If it does not at that difficulty, the
problem is not the difficulty and a longer run will not fix it.

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

Run all five. They take about three minutes together and they are the difference between
debugging a bad checkpoint and never creating one. `diagnose_incentives.py` is the one
added after three training runs completed zero episodes; do not skip it.

| Command | Expected |
|---|---|
| `.venv\Scripts\python.exe smoke_test.py` | 16 PASS, 0 TODO, 0 FAIL |
| `.venv\Scripts\python.exe scripts\verify_observations.py` | 53 passed, 0 failed |
| `.venv\Scripts\python.exe scripts\verify_rewards.py` | 38 passed, 0 failed |
| `.venv\Scripts\python.exe scripts\diagnose_incentives.py --num-cartons 4` | 5 gates PASS |
| `.venv\Scripts\python.exe scripts\run_evaluation.py --baseline greedy --episodes 30` | makespan 23 / 58 / 97, 100% |

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
--num-cartons N     Cartons in play. Default 12; start at 4.
--shaping-scale F   Strength of the potential-based shaping. Default 6.0.
--no-shaping        Train the specification's sparse reward exactly. Ablations only.
--gamma FLOAT       Discount. Default 0.99.
--curriculum        Promote 4 -> 8 -> 12 cartons on rolling success.
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

**Prioritise core count over GPU** when choosing a machine for this.

---

## 8. Monitoring a run

```powershell
tensorboard --logdir tensorboard_logs
```

What to watch, in order of usefulness:

| Metric | What it should do |
|---|---|
| `rollout/ep_rew_mean` | Rise. Flat after ~500k steps means something is wrong. |
| `rollout/ep_len_mean` | **Fall.** This is makespan. It is the metric that matters. |
| `rollout/success_rate` | Rise from zero. Fed by the `is_success` flag the env sets. |
| `train/explained_variance` | Rise towards 1. Near zero means the critic is not learning. |
| `train/entropy_loss` | Rise slowly (less negative) as the policy commits. A collapse to zero early means premature convergence — raise `--lr` decay or `ent_coef`. |
| `train/approx_kl` | Stay under about 0.02. Consistently higher means the learning rate is too high. |

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

## 11. The failure mode you will hit first

The first serious run — 5,013,504 steps, 1.19 hours — **completed zero episodes** and
converged to a mean episode reward of &minus;103. Worth understanding before you start,
because the curves look deceptively healthy while it happens.

**How to recognise it.** `rollout/ep_len_mean` pinned flat at exactly `max_steps`, and
`rollout/success_rate` flat at 0. Meanwhile `ep_rew_mean` rises steeply and plateaus,
`explained_variance` sits around 0.9, `approx_kl` stays near 0.005 and `clip_fraction`
declines — every optimisation signal says the run is healthy. It is: PPO is correctly
optimising a reward that is maximised by doing nothing.

**Do the arithmetic on the plateau.** A robot that does nothing for a full episode scores:

```
time penalty     steps × −0.05  shared  × 0.90
idle penalty     steps × −0.02  own     × 0.10
```

At 2000 steps that is &minus;94. The run converged to &minus;103, i.e. *below* the
do-nothing floor — the remainder being collisions and invalid actions. For scale, greedy
scores +201 per agent. **If your plateau is near the do-nothing number, the policy has
learned to stand still, not to work.**

**Why it happens.** Two independent causes, both structural:

1. **Collisions dominate delivery during exploration.** Under random actions the world
   produces roughly 71 collisions per 300 steps against about one delivery. That is
   `0.24 × −4.5 = −1.06` per step from collisions against `+0.03` from deliveries —
   moving is about 35× worse than standing still. The policy found the correct answer to
   the question it was being asked.
2. **Long episodes make the terminal rewards invisible.** With `gamma = 0.99`,
   `0.99^98 = 0.37` but `0.99^2000 = 1.9e-9`. The +100 completion bonus and the makespan
   bonus — the two largest terms in the whole table — discount to nothing.

**What was changed as a result**, all of it in the defaults:

- **`max_steps` 2000 → 400.** Greedy's worst observed episode is 123, so this is 3.2×
  headroom. Terminal rewards land inside the discount horizon, and a fixed compute budget
  buys five times as many episodes. Costs about 25% throughput to extra resets.
- **Potential-based reward shaping, on by default.** `F = γΦ(s′) − Φ(s)` with
  `Φ = −distance to current objective` (the depot when carrying, the nearest carton
  otherwise). This form is provably policy-invariant — it changes which policies are
  *findable*, never which is optimal, and cannot be farmed by hovering because the sum
  telescopes. `--no-shaping` restores the specification's reward exactly, for ablations.
- **The curriculum now does something.** It used to promote a `difficulty_level` that
  nothing in the world read. It now sets `num_cartons`, so `--curriculum` genuinely
  starts the task at 4 cartons and promotes on rolling success.

**Those changes were not enough, and the second failure is more instructive than the
first.** `fix_test` — 409,600 steps at 4 cartons with shaping on and `max_steps=400` —
also completed zero episodes. `ep_len_mean` was exactly 400.0 for all 39 iterations.
Reward climbed &minus;478 → &minus;16.5, a 29× improvement that means nothing:

```
   stay  : reward/agent  -17.2   ep_len 400.0   completed  0/10
   policy: reward/agent  -16.5   ep_len 400.0   completed  0/10   <- converged here
   greedy: reward/agent +155.3   ep_len  23.7   completed 10/10
```

Landing 0.7 above a hard-coded `stay` is not learning. **Always score `stay` on the same
settings before believing a reward curve** — it takes thirty seconds and it is the only
thing that tells you what the plateau means.

Four further defects, each measured:

1. **Shelf collisions were an exploration tax the optimal policy never pays.** A random
   policy took 105.8 collision events per episode, **93.8 of them against shelving**;
   greedy took 0.0 across 10 episodes. Predicted cost of a random episode: &minus;476.
   Measured reward at iteration 1: &minus;478. The entire initial reward was collisions.
   *Fixed by refusing the move* — driving into a shelf is now an invalid action costing
   the specification's &minus;0.5, exactly as driving off the grid always was. **No
   reward constant changed.**
2. **Picking up a carton was punished.** Φ switched from "nearest carton" to "depot" when
   `is_carrying` flipped, so it fell off a cliff at the key transition. On a *perfect*
   greedy episode, all four pickups scored negative: &minus;0.469, &minus;0.107,
   &minus;0.399, &minus;0.698. *Fixed* with a remaining-work Φ measured in cartons, which
   is continuous across both pickup and delivery.
3. **Shaping was diluted 10× and went flat.** It was added inside the 0.10 individual
   bucket, so `shaping_scale=15` delivered 1.5. And Φ returned 0 for a non-carrying robot
   once every carton was claimed — at 4 cartons with 4 robots, from step 6 onward.
   *Fixed*: `F_i` now sits outside the 90/10 split, so `R_i = 0.90·shared +
   0.10·individual_i + F_i`, and `n_undelivered` keeps Φ moving all episode.
4. **Discovery was never the bottleneck.** A *random* policy already achieves 3.0 pickups
   and 1.3 deliveries per episode. The reward was reachable and simply outweighed 476:12.
   This is why a behaviour-cloning warm start also failed (0/10 completed, 76.4% action
   match): it solved a problem the run did not have.

After those fixes, at 4 cartons: a pickup pays **+2.06** (was &minus;0.70), shelf
collisions are **0.0** (was 93.8), and random's collision cost fell from &minus;476 to
&minus;44.

### And a fifth defect, which the first canary found

That canary - 153,600 steps at 4 cartons with `shaping_scale=6.0` - **still failed**.
`ep_len_mean` was 150.0 every iteration and `success_rate` 0.00, while reward rose
&minus;65 to &minus;16.7 and entropy barely moved. `scripts/probe_policy.py` on the
checkpoint says what the curve could not:

```
deterministic: stay 100%                                    -> STANDING STILL
stochastic   : turnL 18%  turnR 23%  PICKUP 21%  DROP 10%  stay 26%
               fwd 2%   back 1%                             <- it will not drive
```

It learned to turn and grab but never to move. That is correct play, because **only
movement can collide** - turning, staying, PICKUP and DROP cannot - so the collision
penalty is a risk premium on the one action class that makes progress:

```
EV(move) = shaping gain  -  P(collision | move) x 4.5  -  time penalty
         =    +0.150     -        0.1076 x 4.5         -     0.045      =  -0.227
```

P(collision | move) measures 0.108 under random play, 0.146 mid-episode when random
walkers jam the aisles, 0.053 once dispersed, and 0.031 for greedy - elevated throughout,
not a spawn-cluster artifact.

`shaping_scale` is therefore **30.0**, giving EV(move) = **+0.221** under random play. A
scale that large is safe because with the shaping gamma at 1 the episode total telescopes
exactly to `scale x (Phi_end - Phi_start)`, independent of path: it changes gradient
magnitude and nothing else, and no trajectory can farm it.

**The lesson generalises.** An earlier version of this guide compared the shaping gain to
the time penalty and called 3x healthy. Wrong competitor - the time penalty is charged
whether or not the robot moves, so it can never be the reason a policy refuses to move.

### Check the incentives before you train — it costs seconds

```powershell
.venv\Scripts\python.exe scripts\diagnose_incentives.py --num-cartons 4
```

It prints the reward budget for `stay`, `random` and `greedy` side by side and gates on
the six things that actually went wrong: *a pickup pays*, *a delivery pays*, *exploring
is survivable*, *greedy dominates*, *greedy does not hit shelves*, and ***moving is worth
it***. Every failure above would have been caught by it in under a minute instead of
after half an hour of training.
Re-run it after any change to Φ or to `shaping_scale` — the arithmetic is not stable
across a redefinition.

### And check what a checkpoint actually does

```powershell
.venv\Scripts\python.exe scripts\probe_policy.py models\<run>_final.zip --num-cartons 4
```

Action mix, pickups, deliveries and collisions, deterministic and stochastic, with a
stated verdict: *standing still*, *thrashing*, *moving but not working*, *working but not
finishing*, or *completing episodes*. `ep_rew_mean` cannot distinguish the first from the
fourth and both look like a smoothly rising curve - which is how three runs were misread.

Deterministic and stochastic can disagree sharply, so both are reported. The
behaviour-cloned policy scored 0.0 deliveries under argmax and 2.1 under sampling: greedy
uses a single `backward` for a 180-degree turn, so the demonstrator is bimodal from one
observation and the argmax flips between modes. A policy is not broken merely because its
argmax is.

### Then run a canary, not a campaign

```powershell
.venv\Scripts\python.exe -u train.py --run-name canary --timesteps 150000 --num-cartons 4 --worlds 10
```

Four cartons is a 23-step task for greedy, and the cap there is 150 steps. **Kill rule:
if `ep_len_mean` has not dropped below the cap by 100k steps, the run is dead — stop it
rather than letting it finish.** Three runs were allowed to finish before that rule
existed. If `success_rate` does not lift off the floor at 4 cartons, the problem is not
the difficulty.

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

## 13. After this run

The no-communication policy is the control condition. The contribution is the comparison
against a communicating policy: a 16-token broadcast filling the reserved message slots,
with around 10% message dropout so the protocol does not become brittle.

Because the observation width and the network architecture are identical across both
conditions, that comparison is clean — which is the entire reason the slots ship reserved
and zeroed rather than being added later.

Report makespan as the headline, with distance travelled, collision count and message
entropy alongside. Entropy is how you show a real protocol emerged rather than noise.
