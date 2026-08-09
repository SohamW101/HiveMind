# PPO v1 — Current Evaluation (authoritative)

**Model:** `models/ppo_hivemind_v1_final.zip` (10M steps, 2.6 MB, 15×15×5 observation)
**Date:** 2026-08-09
**Method:** `scripts/run_evaluation.py --episodes 30` — 30 episodes per level, 120 total,
`deterministic=True`, CPU, fixed per-episode seeds (`level*1000 + index`) so the run
reproduces exactly.
**Raw data:** `docs_analysis/evaluation_results.json`

> **This supersedes `09_PPO_v1_5_Demo_Run_Analysis.md`.** That document measured the code
> as it stood on 2026-08-07, before the pickup/drop takeover landed in commit `5b8c84b`
> ("fixed pickup/drop mechanics"). The takeover forces action 4/5 once the robot is within
> 0.25 m of its target, which roughly halved episode length and removed collisions almost
> entirely. Doc 09's numbers were never wrong — they describe a version of the environment
> that no longer exists.

> ### ⚠️ Train/test environment mismatch — state this when presenting
>
> | | |
> |:---|:---|
> | PPO_v1_5 training run started | 2026-08-07 00:20 |
> | Takeover added (`5b8c84b`) | 2026-08-08 14:22 |
> | This evaluation | 2026-08-09 |
>
> **The model was trained without the takeover and is evaluated with it.** The evaluated
> MDP is not the one the policy learned in, so most of the improvement over doc 09 is
> attributable to the environment change rather than to the policy.
>
> Concretely, the policy never emits a pickup or a drop — measured Pick Up usage is
> **0.0% at every level**, because the environment supplies those actions. The network
> does the navigation; the environment closes the last 25 cm.
>
> The 79.2% below is a fair measure of **the system as it exists today**, which is also
> what `play_demo.py` demonstrates. It is *not* a measure of what the training run
> achieved, and it should not be compared directly against the training curves in
> `docs_analysis/08` or slide 6. For a like-for-like figure, the pre-takeover environment
> is recoverable at commit `feab93f` and reproduces doc 09's 67.5%.

---

## Results

| Level | Environment | Success | 95% CI | Collision | Timeout | Pickup | Avg reward | Steps (mean / median) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **1** | fixed positions, no obstacles | **100%** | 89–100% | 0% | 0% | 100% | 42.62 ± 0.00 | 34 / 34 |
| **2** | random spawns, no obstacles | **80%** | 63–90% | 0% | 20% | 100% | 25.76 ± 8.55 | 120 / 25 |
| **3** | random spawns + obstacles | **73%** | 56–86% | 0% | 27% | 97% | 28.41 ± 12.17 | 171 / 46 |
| **4** | random spawns + obstacles¹ | **63%** | 46–78% | 0% | 37% | 90% | 21.52 ± 9.90 | 208 / 31 |

**Overall: 79.2% success across 120 episodes.**

¹ Levels 3 and 4 draw from the **same** generator — `_generate_valid_map` only
special-cases level 2, so both sample `integers(3, 9)` obstacles. The L3/L4 difference is
sampling noise, and the confidence intervals overlap heavily (56–86% vs 46–78%).

### What changed against doc 09

| | Doc 09 (2026-08-07, n=40) | Now (2026-08-09, n=120) |
|:---|:---|:---|
| Overall success | 67.5% | **79.2%** |
| L1 steps | 64 | **34** |
| L2 / L3 / L4 success | 70% / 60% / 40% | **80% / 73% / 63%** |
| Collisions | 4/40 (10%) | **0/120 (0%)** |
| Dominant failure mode | collisions at L4 | **timeouts, all levels** |

---

## The failure mode has changed completely

**Zero collisions in 120 episodes.** Doc 09's headline weakness — a 30% collision rate at
level 4 — no longer reproduces at all. Every single failure is now a timeout.

More usefully: **21 of the 25 failures (84%) had already picked up the resource** and then
ran out of steps trying to reach the depot.

| Level | Failures | Of which post-pickup |
|:---|:---|:---|
| 2 | 6 | 6 (100%) |
| 3 | 8 | 7 (88%) |
| 4 | 11 | 8 (73%) |

So the agent reliably finds and grabs the resource — pickup rate is 90–100% at every
level. What it cannot always do is complete the return leg. That is a **navigation and
memory** problem, not an obstacle-perception problem.

### Outcomes are strongly bimodal

Mean steps badly misrepresents the distribution. Successful episodes are fast; failures
burn the full 500-step budget:

| Level | Median steps (successes only) | Slowest success | Mean over all episodes |
|:---|:---|:---|:---|
| 1 | 34 | 34 | 34 |
| 2 | 22 | 55 | 120 |
| 3 | 32 | 350 | 171 |
| 4 | 20 | 327 | 208 |

A level-4 success typically takes **20 steps**. The 208-step mean is an artifact of
averaging 20-step wins with 500-step timeouts.

---

## An oddity worth knowing before someone asks

The policy **never selects Pick Up (action 4)** — 0.0% at every level. It does not need
to: the environment's takeover fires action 4 automatically inside the 0.25 m radius.

More awkwardly, the policy emits **Drop Off (action 5) on 40% of steps at level 2 and 36%
at level 3**, while not carrying anything. Out of range, action 5 is a no-op — the agent
has learned to spend a large share of its steps doing nothing. Those wasted steps
plausibly contribute to the timeout failures.

The environment overrode the policy's chosen action on 5.9% / 14.8% / 10.2% / 22.9% of
steps at L1–L4. Any action-distribution analysis needs to say which of the two it is
reporting; `test_ppo_demo.py` now prints both.

---

## Reproducing

```bash
python scripts/run_evaluation.py --episodes 30
```

Writes `docs_analysis/evaluation_results.json`, the single source for every number in
this document and for the slide deck (which is maintained outside this repo). Re-running
with the same episode count and model reproduces the table exactly — episodes use fixed
seeds.

A smaller run must write elsewhere, or `run_evaluation.py` will refuse to overwrite the
better data:

```bash
python scripts/run_evaluation.py --episodes 5 --out docs_analysis/eval_smoke.json
```

---

## What this means for the mid-eval narrative

1. **The headline improves**: 79.2% over 120 episodes, up from 67.5% over 40.
2. **"30% collision rate at level 4" is no longer true** and should not be presented as
   the outstanding weakness — it is 0%.
3. **The real remaining weakness is the return leg**: 84% of failures happen after a
   successful pickup. The stated v2 fix (a 21×21 window to stop walls and boxes aliasing)
   was aimed at collisions. It may still help the agent locate the depot, but the
   *justification* needs restating — the aliasing-causes-collisions story is not supported
   by the current data.
4. **Error bars matter at this sample size.** Even at n=30 the L3 and L4 intervals overlap
   heavily, which is the honest answer to "why is level 4 worse than level 3" — and it is
   consistent with them being the same environment.
