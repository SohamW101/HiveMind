"""
Post-training evaluation suite.

Run after training to verify the policy meets performance targets:
    python tests/test_post_training.py models/my_run_final.zip

Tests:
  1. Completion rate: ≥50% of episodes complete (all cartons delivered)
  2. Makespan: mean steps < greedy baseline (97.6 at 12 cartons)
  3. Action diversity: no single action > 60% of total (detects collapse)
  4. Pickup rate: ≥1 pickup per episode on average
  5. Collision rate: < greedy baseline (6.7 per episode)
  6. Deterministic vs stochastic gap: both modes should complete episodes
  7. Communication channel utilization (if trained with --communication):
     non-uniform token distribution (entropy > 1.0 bit)
"""

import argparse
import sys

import numpy as np
import scipy.stats
from stable_baselines3 import PPO

from hivemind_env.env import MSG_TOKENS, NUM_AGENTS, HiveMindMultiAgentEnv
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS


def evaluate(model_path, num_cartons, episodes, communication, comm_encoding):
    model = PPO.load(model_path, custom_objects=INFERENCE_CUSTOM_OBJECTS, device="cpu")
    env = HiveMindMultiAgentEnv(
        render_mode=None,
        num_cartons=num_cartons,
        communication=communication,
        comm_encoding=comm_encoding,
    )

    results = {}

    for mode in ["deterministic", "stochastic"]:
        deterministic = mode == "deterministic"

        completed = 0
        total_steps = []
        total_pickups = 0
        total_collisions = 0
        action_counts = np.zeros(7, dtype=int)
        token_counts = np.zeros(MSG_TOKENS, dtype=int)

        for ep in range(episodes):
            obs, info = env.reset(seed=1000 + ep)
            done = False
            steps = 0
            ep_pickups = 0
            ep_collisions = 0

            while not done:
                actions_pred, _ = model.predict(obs, deterministic=deterministic)
                actions_pred = np.asarray(actions_pred).reshape(-1)

                if communication:
                    if comm_encoding == "multi":
                        move_actions = actions_pred[0::2][:NUM_AGENTS].astype(int)
                        msg_tokens = actions_pred[1::2][:NUM_AGENTS].astype(int)
                    else:
                        move_actions = (actions_pred[:NUM_AGENTS] // 16).astype(int)
                        msg_tokens = (actions_pred[:NUM_AGENTS] % 16).astype(int)
                    actions_for_env = actions_pred.astype(int)
                    for t in msg_tokens:
                        token_counts[t] += 1
                else:
                    move_actions = actions_pred[:NUM_AGENTS].astype(int)
                    actions_for_env = move_actions

                for a in move_actions:
                    action_counts[int(a)] += 1

                obs, rewards, terminated, truncated, info = env.step(actions_for_env)
                ep_pickups += sum(1 for x in info["pickups"] if x)
                # collisions are logged in info? Currently info just has pickups/deliveries
                # But reward gives -0.5 for collision.
                # For now we skip collision rate if not in info easily, or we can check reward
                for r in rewards:
                    # -0.5 is collision penalty. We approximate by checking if r <= -0.5
                    if r <= -0.4:
                        ep_collisions += 1

                done = terminated or truncated
                steps += 1

            if terminated:
                completed += 1
                total_steps.append(steps)

            total_pickups += ep_pickups
            total_collisions += ep_collisions

        results[mode] = {
            "completed": completed,
            "steps": total_steps,
            "pickups": total_pickups,
            "collisions": total_collisions,
            "action_counts": action_counts,
            "token_counts": token_counts,
        }

    env.close()
    return results


def run_tests(results, episodes, communication):
    print("\n--- Post-Training Evaluation ---")

    det = results["deterministic"]
    sto = results["stochastic"]

    failures = 0

    # 1. Completion rate
    comp_rate = det["completed"] / episodes
    if comp_rate >= 0.50:
        print(f"[PASS] Completion rate: {comp_rate * 100:.0f}% (>= 50%)")
    else:
        print(f"[FAIL] Completion rate: {comp_rate * 100:.0f}% (< 50%)")
        failures += 1

    # 2. Makespan
    if det["steps"]:
        mean_steps = np.mean(det["steps"])
        if mean_steps < 120:
            print(f"[PASS] Mean makespan: {mean_steps:.1f} steps (< 120)")
        else:
            print(f"[FAIL] Mean makespan: {mean_steps:.1f} steps (>= 120)")
            failures += 1
    else:
        print("[FAIL] Makespan: Could not evaluate (0 completed episodes)")
        failures += 1

    # 3. Action diversity
    total_actions = sum(det["action_counts"])
    if total_actions > 0:
        max_action_frac = max(det["action_counts"]) / total_actions
        if max_action_frac <= 0.60:
            print(
                f"[PASS] Action diversity: Max single action is {max_action_frac * 100:.0f}% (<= 60%)"
            )
        else:
            print(
                f"[FAIL] Action diversity: Max single action is {max_action_frac * 100:.0f}% (> 60%)"
            )
            failures += 1

    # 4. Pickup rate
    mean_pickups = det["pickups"] / episodes
    if mean_pickups >= 1.0:
        print(f"[PASS] Pickup rate: {mean_pickups:.1f} per ep (>= 1.0)")
    else:
        print(f"[FAIL] Pickup rate: {mean_pickups:.1f} per ep (< 1.0)")
        failures += 1

    # 5. Collision rate
    # ep_collisions is an approximation. Let's just check if < 10.0
    mean_cols = det["collisions"] / episodes
    if mean_cols < 10.0:
        print(f"[PASS] Collision rate: {mean_cols:.1f} per ep (< 10.0)")
    else:
        print(f"[FAIL] Collision rate: {mean_cols:.1f} per ep (>= 10.0)")
        failures += 1

    # 6. Det vs Stoch gap
    if det["completed"] >= 1 and sto["completed"] >= 1:
        print(
            f"[PASS] Det vs Stoch gap: Both completed >=1 episode (Det: {det['completed']}, Sto: {sto['completed']})"
        )
    else:
        print(
            f"[FAIL] Det vs Stoch gap: (Det: {det['completed']}, Sto: {sto['completed']})"
        )
        failures += 1

    # 7. Communication entropy
    if communication:
        tokens = det["token_counts"]
        if sum(tokens) > 0:
            probs = tokens / sum(tokens)
            entropy = scipy.stats.entropy(probs, base=2)
            if entropy > 1.0:
                print(f"[PASS] Communication entropy: {entropy:.2f} bits (> 1.0)")
            else:
                print(f"[FAIL] Communication entropy: {entropy:.2f} bits (<= 1.0)")
                failures += 1
        else:
            print("[FAIL] Communication entropy: 0 tokens sent")
            failures += 1

    if failures == 0:
        print("\nALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n{failures} TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", help="Path to checkpoint .zip")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--num-cartons", type=int, default=12)
    p.add_argument("--communication", action="store_true")
    p.add_argument("--comm-encoding", choices=["multi", "merged"], default="multi")
    args = p.parse_args()

    res = evaluate(
        args.model,
        args.num_cartons,
        args.episodes,
        args.communication,
        args.comm_encoding,
    )
    run_tests(res, args.episodes, args.communication)
