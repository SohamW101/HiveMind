"""
Standard PPO Test Demo Runner
Tests the feed-forward PPO model across all difficulty levels for comparison with RecurrentPPO.
"""
import os
import sys
import numpy as np
import time

from hivemind_env.env import HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor
from stable_baselines3 import PPO

def run_evaluation(model_path, difficulty, num_episodes=10, render=False):
    """Run the PPO model for N episodes at a given difficulty and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  Testing at Difficulty Level {difficulty} ({num_episodes} episodes)")
    print(f"{'='*60}")
    
    render_mode = "human" if render else None
    env = HiveMindSingleAgentEnv(render_mode=render_mode, difficulty_level=difficulty, obs_size=15)
    
    model = PPO.load(model_path, device="cpu")
    
    results = []
    action_counts = np.zeros(7, dtype=int)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        picked_up = False
        dropped_off = False
        collided = False
        ep_actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            ep_actions.append(int(action))
            action_counts[int(action)] += 1
            
            if reward >= 1.5:
                picked_up = True
            if reward >= 9.0:
                dropped_off = True
            if terminated and reward < -1.0:
                collided = True
            
            done = terminated or truncated
            
            if render:
                time.sleep(0.03)
        
        if dropped_off:
            outcome = "SUCCESS"
        elif collided:
            outcome = "COLLISION"
        else:
            outcome = "TIMEOUT"
        
        ep_action_dist = np.bincount(ep_actions, minlength=7)
        
        results.append({
            "episode": ep + 1,
            "outcome": outcome,
            "reward": total_reward,
            "steps": steps,
            "picked_up": picked_up,
            "dropped_off": dropped_off,
            "collided": collided,
            "action_dist": ep_action_dist,
        })
        
        print(f"  Ep {ep+1:02d} | {outcome:<9} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | Pickup: {'Y' if picked_up else 'N'} | Dropoff: {'Y' if dropped_off else 'N'}")
    
    env.close()
    
    rewards = [r["reward"] for r in results]
    steps_list = [r["steps"] for r in results]
    successes = sum(1 for r in results if r["outcome"] == "SUCCESS")
    collisions = sum(1 for r in results if r["outcome"] == "COLLISION")
    timeouts = sum(1 for r in results if r["outcome"] == "TIMEOUT")
    pickups = sum(1 for r in results if r["picked_up"])
    
    action_names = ["Forward", "Backward", "Turn Left", "Turn Right", "Pick Up", "Drop Off", "Stay"]
    total_actions = action_counts.sum()
    
    print(f"\n  --- Level {difficulty} Summary ---")
    print(f"  Success Rate : {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)")
    print(f"  Collisions   : {collisions}/{num_episodes}")
    print(f"  Timeouts     : {timeouts}/{num_episodes}")
    print(f"  Pickups      : {pickups}/{num_episodes}")
    print(f"  Avg Reward   : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Avg Steps    : {np.mean(steps_list):.0f}")
    print(f"  Reward Range : [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"\n  Action Distribution:")
    for i, name in enumerate(action_names):
        pct = 100 * action_counts[i] / total_actions if total_actions > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {name:<10}: {action_counts[i]:5d} ({pct:5.1f}%) {bar}")
    
    return {
        "difficulty": difficulty,
        "success_rate": successes / num_episodes,
        "collision_rate": collisions / num_episodes,
        "timeout_rate": timeouts / num_episodes,
        "pickup_rate": pickups / num_episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "action_distribution": {name: int(action_counts[i]) for i, name in enumerate(action_names)},
    }

def main():
    # Try all known PPO model paths in priority order
    candidates = [
        "models/ppo_hivemind_v1_final.zip",          # ← v1 final (10M steps)
        "models/checkpoints_ppo_v1/ppo_v1_8000000_steps.zip",  # ← 8M checkpoint fallback
        "models/ppo_hivemind_test.zip",               # ← old test model
    ]
    
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
    
    if model_path is None:
        print("ERROR: No PPO model found!")
        for c in candidates:
            print(f"  Checked: {c}")
        sys.exit(1)
    
    print("=" * 60)
    print("  Standard PPO (Feed-Forward) Model Evaluation")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Model Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    all_results = {}
    for level in [1, 2, 3, 4]:
        all_results[level] = run_evaluation(model_path, difficulty=level, num_episodes=10, render=False)
    
    print(f"\n{'='*60}")
    print(f"  CROSS-LEVEL SUMMARY (Standard PPO)")
    print(f"{'='*60}")
    print(f"  {'Level':<7} {'Success':<10} {'Collision':<11} {'Timeout':<9} {'Pickup':<8} {'Avg Reward':<12}")
    print(f"  {'-'*57}")
    for level in [1, 2, 3, 4]:
        r = all_results[level]
        print(f"  {level:<7} {r['success_rate']*100:5.0f}%     {r['collision_rate']*100:5.0f}%      {r['timeout_rate']*100:5.0f}%    {r['pickup_rate']*100:5.0f}%   {r['avg_reward']:8.2f}")
    
    print(f"\n  DONE. Full PPO evaluation complete.")

if __name__ == "__main__":
    main()
