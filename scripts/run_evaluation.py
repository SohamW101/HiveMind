"""
Reproducible evaluation of a trained policy across the carton-count curriculum levels.

Ported from `single-agent-rl` (roadmap step 2) and adapted for the 4-robot warehouse.
Writes a JSON summary plus a formatted text report. Episodes use fixed per-episode seeds
(level * 1000 + episode index), so a rerun reproduces the same maps and the same numbers.

Why the original exists: the figures in docs_analysis/09 were produced before the
pickup/drop takeover landed in `5b8c84b`, so they describe an environment the code no
longer has. Fixed seeds and a written-down harness are how that stops happening.

WHAT CHANGED IN THE PORT (each site is marked "PORT NOTE"):
  - the headline metric is MAKESPAN (steps to deliver all 12), not success rate.
    CLAUDE.md step 8: "Track makespan (headline), distance travelled, collision count,
    and message entropy." Success rate is still reported, but makespan is the number the
    greedy baseline (step 5) is compared against.
  - levels are carton counts (4 / 8 / 12), not obstacle-density levels 1-4
  - --policy-mode selects how a policy is turned into a joint MultiDiscrete action
  - the single-agent info keys (requested_action / executed_action / action_overridden)
    are gone; this env never overrides the policy, so there is no override rate
  - --baseline runs the harness with a random policy and no model, so the harness itself
    is testable before step 6 produces anything to load

STATUS: roadmap steps 3 and 4 have landed, so episodes now end on their own and rewards
are real. What is still missing is a policy to evaluate (step 6) and the greedy baseline
(step 5) that gives the makespan number meaning.

Usage:
    .venv\\Scripts\\python.exe scripts/run_evaluation.py --baseline random --episodes 5
    .venv\\Scripts\\python.exe scripts/run_evaluation.py --model models/xxx.zip --episodes 30
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import date

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hivemind_env.env import DEFAULT_OBS_DIM, OBS_DIM_V3, HiveMindMultiAgentEnv
from hivemind_env.training import (
    CURRICULUM_CARTONS,
    NUM_AGENTS,
    SUCCESS_REWARD_THRESHOLD,
    load_policy,
)

# PORT NOTE: the single-agent branch imported its feature extractor at module scope
# because SB3 unpickles it by name. models.py was empty when this file was ported, so the
# import moved into _load_model(). It stays there: evaluation of a --baseline run should
# not need the network module at all.

ACTION_NAMES = ["Forward", "Backward", "Turn Left", "Turn Right", "Pick Up", "Drop Off", "Stay"]

# PORT NOTE: LEVEL_LABELS described obstacle density on the single-agent branch. Here the
# ladder is carton count, mirroring hivemind_env.training.CURRICULUM_CARTONS so the
# curriculum the policy trained under and the levels it is scored on cannot drift apart.
LEVEL_LABELS = {
    level: f"{cartons} cartons, 4 robots, randomised shelf gaps"
    for level, cartons in CURRICULUM_CARTONS.items()
}


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval - honest error bars for small-n success rates."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(terminated, info, final_reward):
    """
    Outcome from the terminal event, not the episode total.

    PORT NOTE: the single-agent version thresholded the final reward because distance
    shaping dominated the total. Here `remaining_resources` is an unambiguous statement
    of whether the job finished, so it is preferred; the reward threshold is the fallback
    for when step 4 changes the info dict.
    """
    if isinstance(info, dict) and "remaining_resources" in info:
        if int(info["remaining_resources"]) == 0:
            return "COMPLETE"
        return "TIMEOUT" if not terminated else "TERMINATED_INCOMPLETE"
    if terminated and float(np.max(final_reward)) >= SUCCESS_REWARD_THRESHOLD:
        return "COMPLETE"
    return "TIMEOUT"


def _joint_action(model, obs, policy_mode, recurrent, lstm_states, episode_start, action_space):
    """
    Turn whatever the policy is into the MultiDiscrete([7,7,7,7]) the env wants.

    PORT NOTE: this function is entirely new, and it is the one real design decision in
    the port. The single-agent harness called `model.predict(obs)` and cast to int.

    CLAUDE.md step 6 recommends training a *shared* policy over 4 single-agent slots, so
    the saved model's action space will be Discrete(7), not MultiDiscrete([7]*4). Two
    modes cover both outcomes:

      shared : one Discrete(7) policy queried once per robot, results stacked. Matches
               the step-6 "cheap option" wrapper. Default.
      joint  : one policy whose action space is already MultiDiscrete([7]*4, e.g. a
               dedicated MAPPO implementation. Queried once.
      random : no model; samples the action space. Exercises the harness itself.

    Returns (action, lstm_states).
    """
    if policy_mode == "random":
        return action_space.sample(), None

    def _predict(single_obs, state):
        if recurrent:
            return model.predict(
                single_obs, state=state,
                episode_start=np.array([episode_start]), deterministic=True,
            )
        act, _ = model.predict(single_obs, deterministic=True)
        return act, None

    if policy_mode == "joint":
        action, lstm_states = _predict(obs, lstm_states)
        return np.asarray(action, dtype=int).reshape(NUM_AGENTS), lstm_states

    # shared: obs is expected to be indexable per agent (obs[i] is agent i's vector).
    if lstm_states is None:
        lstm_states = [None] * NUM_AGENTS
    actions = []
    for i in range(NUM_AGENTS):
        act, lstm_states[i] = _predict(obs[i], lstm_states[i])
        actions.append(int(np.asarray(act).reshape(-1)[0]))
    return np.array(actions, dtype=int), lstm_states


def _distance_travelled(prev_positions, positions):
    """Sum of per-robot XY displacement this step. CLAUDE.md step 8 metric."""
    total = 0.0
    for (px, py, _), (x, y, _) in zip(prev_positions, positions):
        total += math.hypot(x - px, y - py)
    return total


def evaluate_level(model, difficulty, num_episodes, obs_dim, recurrent, policy_mode,
                   max_steps=None, verbose=True):
    env = HiveMindMultiAgentEnv(
        render_mode=None, difficulty_level=difficulty, obs_dim=obs_dim
    )
    # PORT NOTE - this guard is new and it is not optional.
    #
    # The single-agent harness looped `while True` and trusted the env to raise
    # `truncated` at max_steps. This env returns hard-coded `False, False` from step()
    # (roadmap step 4), so that loop never exits: the first smoke run hung until it was
    # killed at the two-minute mark rather than finishing two episodes.
    #
    # The harness therefore enforces its own budget. Keep it even after step 4 lands - an
    # evaluation script that can hang on a buggy env is a bad evaluation script.
    step_budget = max_steps if max_steps is not None else env.max_steps

    episodes = []
    action_counts = np.zeros(7, dtype=int)
    total_steps = 0

    for ep in range(num_episodes):
        seed = difficulty * 1000 + ep
        obs, info = env.reset(seed=seed)
        lstm_states = None
        episode_start = True

        # PORT NOTE: per-agent reward sums replace the single scalar. The 90/10 split
        # (CLAUDE.md step 4) gives each robot a different total for the same shared
        # outcome, and the spread across robots is itself a signal about role division.
        agent_rewards = np.zeros(NUM_AGENTS, dtype=float)
        steps = 0
        distance = 0.0
        collisions = 0
        reward = np.zeros(NUM_AGENTS)
        terminated = False
        prev_positions = list(info["robot_pos"])
        cartons_at_start = info["remaining_resources"]

        while True:
            action, lstm_states = _joint_action(
                model, obs, policy_mode, recurrent, lstm_states, episode_start,
                env.action_space,
            )
            episode_start = False

            obs, reward, terminated, truncated, info = env.step(action)
            agent_rewards += np.asarray(reward, dtype=float)
            steps += 1
            for a in np.asarray(action).reshape(-1):
                action_counts[int(a)] += 1

            distance += _distance_travelled(prev_positions, info["robot_pos"])
            prev_positions = list(info["robot_pos"])
            collisions += int(info.get("collisions", 0))  # step 4 should provide this

            if terminated or truncated:
                break
            if steps >= step_budget:
                truncated = True
                break

        total_steps += steps
        outcome = classify(terminated, info, reward)
        delivered = cartons_at_start - info["remaining_resources"]

        # Makespan is only defined for a completed job. Recording None (rather than
        # max_steps) keeps timeouts out of the mean instead of silently flattering it -
        # report completion rate alongside makespan, never makespan alone.
        makespan = steps if outcome == "COMPLETE" else None

        episodes.append({
            "episode": ep + 1, "seed": seed, "outcome": outcome,
            "makespan": makespan, "steps": steps,
            "delivered": int(delivered), "cartons": int(cartons_at_start),
            "distance": round(distance, 2), "collisions": collisions,
            "reward_total": round(float(agent_rewards.sum()), 2),
            "reward_per_agent": [round(float(r), 2) for r in agent_rewards],
        })
        if verbose:
            ms = f"{makespan:4d}" if makespan is not None else "   -"
            print(f"  Ep {ep+1:02d} | seed {seed:<5} | {outcome:<20} | "
                  f"Makespan: {ms} | Steps: {steps:4d} | "
                  f"Delivered: {delivered:2d}/{cartons_at_start} | "
                  f"Dist: {distance:6.1f}m | Coll: {collisions:3d}", flush=True)

    env.close()

    n = len(episodes)
    completed = [e for e in episodes if e["outcome"] == "COMPLETE"]
    makespans = [e["makespan"] for e in completed]
    lo, hi = wilson_interval(len(completed), n)
    delivered_frac = [e["delivered"] / e["cartons"] if e["cartons"] else 0.0 for e in episodes]

    return {
        "difficulty": difficulty,
        "label": LEVEL_LABELS.get(difficulty, f"level {difficulty}"),
        "num_episodes": n,
        # Headline.
        "completion_rate": len(completed) / n,
        "completion_ci95": [round(lo, 4), round(hi, 4)],
        "avg_makespan": round(float(np.mean(makespans)), 1) if makespans else None,
        "median_makespan": float(statistics.median(makespans)) if makespans else None,
        "min_makespan": min(makespans) if makespans else None,
        "max_makespan": max(makespans) if makespans else None,
        # Supporting metrics (CLAUDE.md step 8).
        "avg_delivered_fraction": round(float(np.mean(delivered_frac)), 4),
        "avg_distance": round(float(np.mean([e["distance"] for e in episodes])), 1),
        "avg_collisions": round(float(np.mean([e["collisions"] for e in episodes])), 2),
        "avg_steps": round(float(np.mean([e["steps"] for e in episodes])), 1),
        "avg_reward_total": round(float(np.mean([e["reward_total"] for e in episodes])), 2),
        "std_reward_total": round(float(np.std([e["reward_total"] for e in episodes])), 2),
        "action_distribution": {ACTION_NAMES[i]: int(action_counts[i]) for i in range(7)},
        "episodes": episodes,
    }


def _load_model(path, policy_mode):
    if policy_mode == "random":
        return None, False
    try:
        # SB3 unpickles the feature extractor by name, so the module must be importable
        # before load. Lazy because models.py is empty until roadmap step 6.
        from hivemind_env.models import CustomCombinedExtractor  # noqa: F401
    except ImportError:
        print("  WARNING: hivemind_env/models.py is empty (roadmap step 6). If the saved\n"
              "           policy used a custom feature extractor, SB3 will fail to unpickle it.")
    return load_policy(path, device="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on the multi-agent warehouse across curriculum levels"
    )
    parser.add_argument("--model", default=None,
                        help="Path to a saved SB3 policy. Omit and pass --baseline instead.")
    parser.add_argument("--baseline", choices=["random"], default=None,
                        help="Run without a model. 'random' samples the action space - "
                             "useful for checking the harness itself.")
    parser.add_argument("--policy-mode", choices=["shared", "joint"], default="shared",
                        help="shared: one Discrete(7) policy queried per robot (roadmap "
                             "step 6 default). joint: one MultiDiscrete policy.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM,
                        help="Asserted against the pinned width in env.py; env.py "
                             "rejects a mismatch rather than building a bad env.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Hard per-episode step budget enforced by the harness. "
                             "Defaults to env.max_steps (2000). Lower it for smoke runs - "
                             "the env does not raise truncated yet (roadmap step 4), so "
                             "without this the harness would run forever.")
    parser.add_argument("--levels", type=int, nargs="+", default=sorted(CURRICULUM_CARTONS))
    parser.add_argument("--out", default="docs_analysis/evaluation_results.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --out even if it holds a larger, more reliable run.")
    args = parser.parse_args()

    if (args.model is None) == (args.baseline is None):
        parser.error("pass exactly one of --model PATH or --baseline random")

    policy_mode = args.baseline if args.baseline else args.policy_mode

    if args.model and not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    # Guard: --out is the source of truth for the docs and the figures. A quick smoke run
    # (--episodes 5) must not silently replace a 30-episode result set with weaker
    # numbers.
    if os.path.exists(args.out) and not args.force:
        try:
            with open(args.out, encoding="utf-8") as f:
                existing = json.load(f).get("episodes_per_level", 0)
        except (OSError, ValueError):
            existing = 0
        if existing > args.episodes:
            sys.exit(
                f"ERROR: {args.out} already holds a {existing}-episode-per-level run, and\n"
                f"this one is only {args.episodes}. Refusing to overwrite better data.\n\n"
                f"  Write elsewhere:  --out docs_analysis/eval_smoke.json\n"
                f"  Or overwrite:     --force"
            )

    print("=" * 84)
    print("  HiveMind Multi-Agent Policy Evaluation")
    print("=" * 84)
    if args.model:
        print(f"  Model      : {args.model} ({os.path.getsize(args.model)/1024/1024:.1f} MB)")
    else:
        print(f"  Model      : none - {args.baseline} baseline")
    print(f"  Policy mode: {policy_mode}")
    print(f"  Agents     : {NUM_AGENTS}")
    print(f"  Episodes   : {args.episodes} per level (fixed seeds, reproducible)")
    print(f"  Date       : {date.today().isoformat()}", flush=True)

    # Honest banner: refuse to imply these numbers mean anything they cannot yet mean.
    probe = HiveMindMultiAgentEnv(render_mode=None)
    obs, _ = probe.reset(seed=0)
    obs_missing = len(obs) == 0
    probe.close()
    if obs_missing:
        print("\n  !! WARNING: _get_obs() returns an empty list (roadmap step 3 not done),")
        print("     and reward/termination are hard-coded (step 4 not done). Every episode")
        print("     will run to max_steps with zero reward. These numbers measure nothing")
        print("     about a policy - they only prove the harness executes.")
        if args.model:
            sys.exit("\nERROR: refusing to evaluate a model against an empty observation.\n"
                     "       Finish roadmap step 3 first, or use --baseline random.")

    model, recurrent = _load_model(args.model, policy_mode)
    if model is not None:
        print(f"  Algorithm  : {'RecurrentPPO' if recurrent else 'PPO'}", flush=True)

    started = time.time()
    results = {}
    for level in args.levels:
        label = LEVEL_LABELS.get(level, f"level {level}")
        print(f"\n{'='*84}\n  Level {level} - {label}\n{'='*84}", flush=True)
        results[level] = evaluate_level(
            model, level, args.episodes, args.obs_dim, recurrent, policy_mode,
            max_steps=args.max_steps,
        )
        r = results[level]
        lo, hi = r["completion_ci95"]
        ms = f"{r['avg_makespan']:.0f}" if r["avg_makespan"] is not None else "n/a (no completions)"
        print(f"\n  Makespan  : {ms}   (headline - lower is better)")
        print(f"  Completed : {r['completion_rate']*100:.0f}%  (95% CI {lo*100:.0f}-{hi*100:.0f}%)"
              f"   Delivered: {r['avg_delivered_fraction']*100:.0f}% of cartons")
        print(f"  Distance  : {r['avg_distance']:.1f} m    Collisions: {r['avg_collisions']:.1f}")
        print(f"  Reward    : {r['avg_reward_total']:.2f} +/- {r['std_reward_total']:.2f} "
              f"(summed over {NUM_AGENTS} agents)")
        print(f"  Steps     : mean {r['avg_steps']:.0f}", flush=True)

    elapsed = time.time() - started
    overall_eps = sum(results[l]["num_episodes"] for l in results)
    overall_done = sum(results[l]["completion_rate"] * results[l]["num_episodes"] for l in results)

    print(f"\n{'='*84}\n  CROSS-LEVEL SUMMARY\n{'='*84}")
    print(f"  {'Level':<7}{'Cartons':>9}{'Makespan':>10}{'Complete':>10}{'95% CI':>14}"
          f"{'Dist(m)':>9}{'Coll':>7}{'AvgRew':>9}")
    print(f"  {'-'*75}")
    for level in args.levels:
        r = results[level]
        lo, hi = r["completion_ci95"]
        ms = f"{r['avg_makespan']:.0f}" if r["avg_makespan"] is not None else "-"
        print(f"  {level:<7}{CURRICULUM_CARTONS.get(level, '?'):>9}{ms:>10}"
              f"{r['completion_rate']*100:>9.0f}%{f'{lo*100:.0f}-{hi*100:.0f}%':>14}"
              f"{r['avg_distance']:>9.1f}{r['avg_collisions']:>7.1f}"
              f"{r['avg_reward_total']:>9.2f}")
    print(f"  {'-'*75}")
    print(f"  Overall: {overall_done/overall_eps*100:.1f}% of {overall_eps} episodes completed")
    print(f"  Wall clock: {elapsed/60:.1f} min")
    print("\n  Compare avg_makespan against the greedy baseline (CLAUDE.md roadmap step 5).")
    print("  A learned policy that does not beat greedy makespan has not shown anything.")

    payload = {
        "model": args.model,
        "policy_mode": policy_mode,
        "num_agents": NUM_AGENTS,
        "obs_dim": args.obs_dim,
        "episodes_per_level": args.episodes,
        "date": date.today().isoformat(),
        "observations_implemented": not obs_missing,
        "overall_completion_rate": round(overall_done / overall_eps, 4),
        "total_episodes": overall_eps,
        "levels": {str(k): v for k, v in results.items()},
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  JSON written to {args.out}")


if __name__ == "__main__":
    main()
