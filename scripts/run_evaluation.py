"""
Reproducible evaluation of a trained policy across all four difficulty levels.

Writes a JSON summary plus a formatted text report. Episodes use fixed per-episode seeds
(level * 1000 + episode index), so a rerun reproduces the same maps and the same numbers.

Why this exists: the figures in docs_analysis/09 were produced before the pickup/drop
takeover landed in `5b8c84b`, so they describe an environment the code no longer has.

Usage:
    python scripts/run_evaluation.py --episodes 30
    python scripts/run_evaluation.py --model models/ppo_hivemind_v1_final.zip --episodes 30
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import date

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from hivemind_env.env import OBS_SIZE_V1, HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor  # noqa: F401  (SB3 unpickles by name)
from hivemind_env.training import SUCCESS_REWARD_THRESHOLD, load_policy

ACTION_NAMES = ["Forward", "Backward", "Turn Left", "Turn Right", "Pick Up", "Drop Off", "Stay"]
LEVEL_LABELS = {
    1: "fixed positions, no obstacles",
    2: "random spawns, no obstacles",
    3: "random spawns + obstacles",
    4: "random spawns + obstacles (same generator as L3)",
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


def classify(terminated, final_reward):
    """
    Outcome from the terminal event, not the episode total. Distance shaping dominates
    the total, so thresholding on it misreports outcomes.
    """
    if terminated and final_reward >= SUCCESS_REWARD_THRESHOLD:
        return "SUCCESS"
    if terminated and final_reward <= -1.0:
        return "COLLISION"
    return "TIMEOUT"


def evaluate_level(model, difficulty, num_episodes, obs_size, recurrent, verbose=True):
    env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=difficulty, obs_size=obs_size)

    episodes = []
    requested_counts = np.zeros(7, dtype=int)
    executed_counts = np.zeros(7, dtype=int)
    overrides = 0
    total_steps = 0

    for ep in range(num_episodes):
        seed = difficulty * 1000 + ep
        obs, info = env.reset(seed=seed)
        lstm_states = None
        episode_start = True

        total_reward, steps = 0.0, 0
        picked_up = False
        reward = 0.0
        terminated = False

        while True:
            if recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=np.array([episode_start]), deterministic=True,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            episode_start = False

            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            requested_counts[info["requested_action"]] += 1
            executed_counts[info["executed_action"]] += 1
            overrides += int(info["action_overridden"])
            picked_up = picked_up or bool(obs["is_carrying"])
            if terminated or truncated:
                break

        total_steps += steps
        outcome = classify(terminated, reward)
        episodes.append({
            "episode": ep + 1, "seed": seed, "outcome": outcome,
            "reward": round(total_reward, 2), "steps": steps, "picked_up": picked_up,
        })
        if verbose:
            print(f"  Ep {ep+1:02d} | seed {seed:<5} | {outcome:<9} | "
                  f"Reward: {total_reward:7.2f} | Steps: {steps:4d} | "
                  f"Pickup: {'Y' if picked_up else 'N'}", flush=True)

    env.close()

    n = len(episodes)
    successes = sum(1 for e in episodes if e["outcome"] == "SUCCESS")
    collisions = sum(1 for e in episodes if e["outcome"] == "COLLISION")
    timeouts = sum(1 for e in episodes if e["outcome"] == "TIMEOUT")
    pickups = sum(1 for e in episodes if e["picked_up"])
    rewards = [e["reward"] for e in episodes]
    steps_list = [e["steps"] for e in episodes]
    lo, hi = wilson_interval(successes, n)

    return {
        "difficulty": difficulty,
        "label": LEVEL_LABELS[difficulty],
        "num_episodes": n,
        "success_rate": successes / n,
        "success_ci95": [round(lo, 4), round(hi, 4)],
        "collision_rate": collisions / n,
        "timeout_rate": timeouts / n,
        "pickup_rate": pickups / n,
        "avg_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "min_reward": round(min(rewards), 2),
        "max_reward": round(max(rewards), 2),
        "avg_steps": round(float(np.mean(steps_list)), 1),
        "median_steps": float(statistics.median(steps_list)),
        "override_rate": round(overrides / total_steps, 4) if total_steps else 0.0,
        "requested_actions": {ACTION_NAMES[i]: int(requested_counts[i]) for i in range(7)},
        "executed_actions": {ACTION_NAMES[i]: int(executed_counts[i]) for i in range(7)},
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy across all levels")
    parser.add_argument("--model", default="models/ppo_hivemind_v1_final.zip")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--obs-size", type=int, default=OBS_SIZE_V1)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--out", default="docs_analysis/evaluation_results.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --out even if it holds a larger, more reliable run.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    # Guard: --out is the source of truth for the docs, the figures and the deck's
    # slide-7 table. A quick smoke run (--episodes 5) must not silently replace a
    # 30-episode result set with weaker numbers.
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

    print("=" * 74)
    print("  HiveMind Policy Evaluation")
    print("=" * 74)
    print(f"  Model      : {args.model} ({os.path.getsize(args.model)/1024/1024:.1f} MB)")
    print(f"  Obs window : {args.obs_size}x{args.obs_size}x5")
    print(f"  Episodes   : {args.episodes} per level (fixed seeds, reproducible)")
    print(f"  Date       : {date.today().isoformat()}", flush=True)

    model, recurrent = load_policy(args.model, device="cpu")
    print(f"  Algorithm  : {'RecurrentPPO' if recurrent else 'PPO'}", flush=True)

    started = time.time()
    results = {}
    for level in args.levels:
        print(f"\n{'='*74}\n  Level {level} - {LEVEL_LABELS[level]}\n{'='*74}", flush=True)
        results[level] = evaluate_level(
            model, level, args.episodes, args.obs_size, recurrent
        )
        r = results[level]
        lo, hi = r["success_ci95"]
        print(f"\n  Success  : {r['success_rate']*100:.0f}%  (95% CI {lo*100:.0f}-{hi*100:.0f}%)")
        print(f"  Collision: {r['collision_rate']*100:.0f}%   Timeout: {r['timeout_rate']*100:.0f}%"
              f"   Pickup: {r['pickup_rate']*100:.0f}%")
        print(f"  Reward   : {r['avg_reward']:.2f} +/- {r['std_reward']:.2f} "
              f"[{r['min_reward']:.2f}, {r['max_reward']:.2f}]")
        print(f"  Steps    : mean {r['avg_steps']:.0f}, median {r['median_steps']:.0f}")
        print(f"  Env overrode the policy on {r['override_rate']*100:.1f}% of steps", flush=True)

    elapsed = time.time() - started
    overall_eps = sum(results[l]["num_episodes"] for l in results)
    overall_succ = sum(results[l]["success_rate"] * results[l]["num_episodes"] for l in results)

    print(f"\n{'='*74}\n  CROSS-LEVEL SUMMARY\n{'='*74}")
    print(f"  {'Level':<7}{'Success':>9}{'95% CI':>14}{'Coll':>7}{'Time-out':>10}"
          f"{'AvgRew':>9}{'AvgSteps':>10}")
    print(f"  {'-'*66}")
    for level in args.levels:
        r = results[level]
        lo, hi = r["success_ci95"]
        print(f"  {level:<7}{r['success_rate']*100:>8.0f}%"
              f"{f'{lo*100:.0f}-{hi*100:.0f}%':>14}"
              f"{r['collision_rate']*100:>6.0f}%{r['timeout_rate']*100:>9.0f}%"
              f"{r['avg_reward']:>9.2f}{r['avg_steps']:>10.0f}")
    print(f"  {'-'*66}")
    print(f"  Overall: {overall_succ/overall_eps*100:.1f}% success across {overall_eps} episodes")
    print(f"  Wall clock: {elapsed/60:.1f} min")

    payload = {
        "model": args.model,
        "obs_size": args.obs_size,
        "episodes_per_level": args.episodes,
        "date": date.today().isoformat(),
        "overall_success_rate": round(overall_succ / overall_eps, 4),
        "total_episodes": overall_eps,
        "levels": {str(k): v for k, v in results.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  JSON written to {args.out}")


if __name__ == "__main__":
    main()
